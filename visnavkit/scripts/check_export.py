"""Compare one model's deployment artifacts on the same inputs: the checkpoint (PyTorch, the reference), the
exporter ``.pth``, the ONNX graph (ONNX Runtime) and the TensorRT engine.

    uv run visnavkit-check-export checkpoint=last.ckpt onnx=policy.onnx engine=policy.fp16.engine
    uv run visnavkit-check-export pth=policy.pth onnx=policy.onnx                # Lightning-free reference
    uv run visnavkit-check-export checkpoint=last.ckpt onnx=flowpilot_dst.onnx   # FlowPilot-DST window graph

The traced inputs ``<onnx>.inputs.npz`` are replayed when present (``inputs=`` names another ``.npz``), else the
reference model generates them (``seed``, ``batch_size``). Every artifact is checked against the reference at the
tolerance of its precision (``atol`` / ``rtol`` override), the sidecar metadata hashes are verified and each
backend is timed. Exit status 1 when anything fails.
"""

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import onnx
import torch
from omegaconf import DictConfig

from visnavkit.benchmark.runtime import create_session
from visnavkit.models.flowpilot_dst import FlowPilotDST
from visnavkit.utils.artifacts import (
    compare_outputs,
    describe,
    load_model,
    mha_fastpath_disabled,
    model_precision,
    onnx_precision,
    read_metadata,
    size_mb,
    tolerance,
)
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def _graph(model, cfg, *, batch_size, top_k, seed, export_heads):
    """``(wrapper, inputs, input_names, output_names)`` of the model's export family."""
    if isinstance(model, FlowPilotDST):
        from visnavkit.scripts.export_dst import prepare_dst_graph

        return prepare_dst_graph(model, cfg, batch_size=batch_size, top_k=top_k, seed=seed)
    from visnavkit.scripts.export import prepare_policy_graph

    return prepare_policy_graph(
        model, cfg, batch_size=batch_size, export_heads=export_heads, seed=seed, device="cpu"
    )


def _torch_runner(wrapper, input_names, output_names, device):
    wrapper = wrapper.to(device)

    @torch.no_grad()
    def run(feeds):
        with mha_fastpath_disabled():
            outputs = wrapper(*[torch.as_tensor(feeds[name]).to(device) for name in input_names])
        return {
            name: (value.float() if value.is_floating_point() else value).cpu().numpy()
            for name, value in zip(output_names, outputs)
        }

    return run


def _onnx_runner(path, provider, output_names):
    session = create_session(path, provider, threads=torch.get_num_threads())
    graph_inputs = {node.name for node in session.get_inputs()}
    graph_outputs = [node.name for node in session.get_outputs()]
    missing = [name for name in output_names if name not in graph_outputs]
    if missing:
        raise ValueError(f"{path} lacks outputs {missing}; it has {graph_outputs}")

    def run(feeds):
        values = session.run(output_names, {name: value for name, value in feeds.items() if name in graph_inputs})
        return dict(zip(output_names, values))

    return run


def _engine_runner(path, output_names):
    from visnavkit.utils.trt import EngineRunner, load_engine

    runner = EngineRunner(load_engine(path))
    missing = [name for name in output_names if name not in runner.output_names]
    if missing:
        raise ValueError(f"{path} lacks outputs {missing}; it has {runner.output_names}")
    return lambda feeds: {name: value for name, value in runner(feeds).items() if name in output_names}


def _decision(model, outputs):
    """The newest decision in one line, and its endpoint (x, y) in metres."""
    if isinstance(model, FlowPilotDST):
        modes, probs = np.asarray(outputs["modes"]), np.asarray(outputs["probs"])
        return f"top plan p={probs[0, 0]:.3f}", modes[0, 0, -1, :2].astype(np.float64)
    from visnavkit.scripts.export import parse_plan_output

    decoder = model.action_decoder
    parsed = parse_plan_output(
        outputs["plan"][:1], M=decoder.num_modes, num_pts=decoder.num_pts, pose_width=decoder.pose_size
    )
    best = int(np.argmax(parsed["pred_logits"]))
    return f"mode {best} logit={parsed['pred_logits'][best]:.3f}", parsed["best_plan"][-1].astype(np.float64)


def _latency_ms(run, feeds, iterations, warmup=2):
    if iterations < 1:
        return None
    for _ in range(warmup):
        run(feeds)
    start = time.perf_counter()
    for _ in range(iterations):
        run(feeds)
    return (time.perf_counter() - start) * 1e3 / iterations


def check_export(
    *,
    checkpoint=None,
    pth=None,
    onnx_path=None,
    engine=None,
    inputs=None,
    seed=0,
    batch_size=1,
    top_k=None,
    device="cpu",
    provider="CPUExecutionProvider",
    atol=None,
    rtol=None,
    iterations=10,
):
    """Run every artifact on the same feeds and report per-output errors against the PyTorch reference.

    ``report["ok"]`` is False when an output leaves its precision's tolerance or is not finite, or a sidecar
    metadata hash does not match the file it describes.
    """
    if not (checkpoint or pth):
        raise ValueError("checkpoint= or pth= is required as the PyTorch reference")
    if not (onnx_path or engine or (checkpoint and pth)):
        raise ValueError("Nothing to compare: give onnx=, engine= or both checkpoint= and pth=")
    paths = (("checkpoint", checkpoint), ("pth", pth), ("onnx", onnx_path), ("engine", engine))
    files = {kind: describe(path) for kind, path in paths if path}
    onnx_meta = read_metadata(Path(onnx_path).with_suffix(".metadata.json")) if onnx_path else None
    engine_meta = read_metadata(Path(engine).with_suffix(".metadata.json")) if engine else None

    reference_kind = "checkpoint" if checkpoint else "pth"
    model, cfg = load_model(checkpoint or pth)
    export_heads = list(cfg.get("export_heads") or (onnx_meta or {}).get("config", {}).get("export_heads") or [])
    top_k = int(top_k or (onnx_meta or {}).get("top_k") or 6)
    graph_options = dict(batch_size=batch_size, top_k=top_k, seed=seed, export_heads=export_heads)
    wrapper, generated, input_names, output_names = _graph(model, cfg, **graph_options)

    feeds_source = Path(inputs) if inputs else Path(onnx_path).with_suffix(".inputs.npz") if onnx_path else None
    if feeds_source is not None and feeds_source.exists():
        feeds = dict(np.load(feeds_source))
        missing = [name for name in input_names if name not in feeds]
        if missing:
            raise ValueError(f"{feeds_source} lacks inputs {missing}; the graph takes {input_names}")
    else:
        if inputs:
            raise FileNotFoundError(inputs)
        feeds_source = f"synthetic (seed {seed}, batch {batch_size})"
        feeds = {name: value.cpu().numpy() for name, value in zip(input_names, generated)}
    logger.info(f"Inputs from {feeds_source}: " + ", ".join(f"{name}{value.shape}" for name, value in feeds.items()))

    reference = _torch_runner(wrapper, input_names, output_names, device)
    reference_outputs = reference(feeds)
    reference_decision, reference_endpoint = _decision(model, reference_outputs)
    report = {
        "reference": {
            "kind": reference_kind,
            **files[reference_kind],
            "precision": model_precision(model),
            "model": type(model).__name__,
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "device": device,
            "decision": reference_decision,
            "latency_ms": _latency_ms(reference, feeds, iterations),
        },
        "inputs": {
            "source": str(feeds_source),
            "shapes": {name: list(value.shape) for name, value in feeds.items()},
            "dtypes": {name: str(value.dtype) for name, value in feeds.items()},
        },
        "artifacts": {},
        "provenance": [],
    }

    artifacts = []
    if checkpoint and pth:
        pth_model, _ = load_model(pth, cfg)
        pth_wrapper = _graph(pth_model, cfg, **graph_options)[0]
        runner = _torch_runner(pth_wrapper, input_names, output_names, device)
        artifacts.append(("pth", pth, model_precision(pth_model), runner, f"PyTorch on {device}"))
    if onnx_path:
        stored, _ = onnx_precision(onnx.load(str(onnx_path), load_external_data=False))
        artifacts.append(("onnx", onnx_path, stored, _onnx_runner(onnx_path, provider, output_names), provider))
    if engine:
        precision = (engine_meta or {}).get("precision")
        if precision is None:
            logger.warning(f"{engine} has no metadata; checking it at the fp16 tolerance")
            precision = "fp16"
        backend = (engine_meta or {}).get("gpu") or "TensorRT"
        artifacts.append(("engine", engine, precision, _engine_runner(engine, output_names), backend))

    for kind, path, precision, run, backend in artifacts:
        outputs = run(feeds)
        r, a = tolerance(precision, rtol, atol)
        expected = [reference_outputs[name] for name in output_names]
        errors = compare_outputs(output_names, expected, [outputs[name] for name in output_names], r, a)
        decision, endpoint = _decision(model, outputs)
        report["artifacts"][kind] = {
            **files[kind],
            "precision": precision,
            "backend": backend,
            "tolerance": {"rtol": r, "atol": a},
            "outputs": errors,
            "decision": decision,
            "endpoint_delta_m": float(np.linalg.norm(endpoint - reference_endpoint)),
            "latency_ms": _latency_ms(run, feeds, iterations),
            "pass": all(item["pass"] and item["finite"] for item in errors.values()),
        }

    recorded = []
    if onnx_meta:
        recorded.append(("onnx metadata: onnx_sha256", onnx_meta.get("onnx_sha256"), files["onnx"]["sha256"]))
        for kind in ("checkpoint", "pth"):
            if kind in files:
                expected = onnx_meta.get(f"{kind}_sha256")
                recorded.append((f"onnx metadata: {kind}_sha256", expected, files[kind]["sha256"]))
    if engine_meta:
        recorded.append(("engine metadata: engine_sha256", engine_meta.get("engine_sha256"), files["engine"]["sha256"]))
        if onnx_path:
            recorded.append(("engine metadata: onnx_sha256", engine_meta.get("onnx_sha256"), files["onnx"]["sha256"]))
    report["provenance"] = [
        {"check": name, "expected": expected, "actual": actual, "pass": expected == actual}
        for name, expected, actual in recorded
        if expected is not None
    ]
    report["ok"] = all(row["pass"] for row in report["artifacts"].values()) and all(
        item["pass"] for item in report["provenance"]
    )
    print_report(report)
    return report


def _ms(value):
    return f"{value:.1f} ms" if value is not None else "-"


def print_report(report):
    ref = report["reference"]
    print("=" * 40 + " EXPORT CHECK " + "=" * 40)
    print(
        f"reference : {ref['kind']} {ref['path']} ({size_mb(ref['path'])}, {ref['precision']}, {ref['model']}"
        f" {ref['parameters_total'] / 1e6:.1f}M params) PyTorch on {ref['device']}, {_ms(ref['latency_ms'])}"
    )
    inputs = report["inputs"]
    print(
        f"inputs    : {inputs['source']}: "
        + ", ".join(f"{name}{tuple(shape)} {inputs['dtypes'][name]}" for name, shape in inputs["shapes"].items())
    )
    print(f"{'artifact':9} {'precision':9} {'size':>9} {'sha256':13} {'latency':>10}  file (backend)")
    for kind, row in report["artifacts"].items():
        print(
            f"{kind:9} {row['precision']:9} {size_mb(row['path']):>9} {row['sha256'][:12]:13}"
            f" {_ms(row['latency_ms']):>10}  {row['path']} ({row['backend']})"
        )
    print(f"{'output':12} {'artifact':9} {'max_abs':>10} {'max_rel':>10} {'rtol':>8} {'atol':>8}  result")
    for kind, row in report["artifacts"].items():
        for name, item in row["outputs"].items():
            result = "PASS" if item["pass"] and item["finite"] else "FAIL" + ("" if item["finite"] else " (not finite)")
            print(
                f"{name:12} {kind:9} {item['max_abs']:10.3g} {item['max_rel']:10.3g}"
                f" {row['tolerance']['rtol']:8.0e} {row['tolerance']['atol']:8.0e}  {result}"
            )
    deltas = " | ".join(
        f"{kind} {row['decision']}, endpoint delta {row['endpoint_delta_m']:.3f} m"
        for kind, row in report["artifacts"].items()
    )
    print(f"decision  : reference {ref['decision']} | {deltas}")
    for item in report["provenance"]:
        print(f"provenance: {item['check']} {'matches' if item['pass'] else 'MISMATCH (stale or foreign sidecar)'}")
    failed = [kind for kind, row in report["artifacts"].items() if not row["pass"]]
    failed += [item["check"] for item in report["provenance"] if not item["pass"]]
    print("RESULT: PASS" if report["ok"] else f"RESULT: FAIL ({', '.join(failed)})")


@hydra.main(version_base=None, config_path="../configs", config_name="check_export")
def main(cfg: DictConfig):
    report = check_export(
        checkpoint=cfg.checkpoint,
        pth=cfg.pth,
        onnx_path=cfg.onnx,
        engine=cfg.engine,
        inputs=cfg.inputs,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        top_k=cfg.top_k,
        device=cfg.device,
        provider=cfg.provider,
        atol=cfg.atol,
        rtol=cfg.rtol,
        iterations=cfg.iterations,
    )
    if not report["ok"]:
        sys.exit(1)
    return report


if __name__ == "__main__":
    main()
