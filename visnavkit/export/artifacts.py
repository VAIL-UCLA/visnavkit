"""What every export family shares: loading the weights, tracing helpers, ONNX Runtime runs, output parity,
the ``.pth`` / ``.metadata.json`` / ``.inputs.npz`` sidecars and the closing summary."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from visnavkit.benchmark.export import sha256_file
from visnavkit.export.precision import convert_onnx, onnx_precision, tolerance
from visnavkit.models.lit_model import disable_pretrained_downloads
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


# ---- files -----------------------------------------------------------------------------------
def describe(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def size_mb(path):
    return f"{Path(path).stat().st_size / 2**20:.1f}MB"


def write_metadata(path, meta):
    Path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return Path(path)


def read_metadata(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# ---- weights ---------------------------------------------------------------------------------
def instantiate_model(cfg):
    """The eval-mode model of ``cfg.model`` without backbone downloads (checkpoint weights follow, or none)."""
    return instantiate(disable_pretrained_downloads(copy.deepcopy(cfg.model))).eval()


def load_model(path, cfg=None):
    """The eval-mode model and its config from a Lightning ``.ckpt`` or an exporter ``.pth``; a file without a
    config (a bare state dict) takes ``cfg``."""
    stored = torch.load(path, map_location="cpu", weights_only=False)
    saved = stored.get("hyper_parameters", {}).get("cfg", stored.get("cfg")) if isinstance(stored, dict) else None
    if saved is not None:
        cfg = saved if isinstance(saved, DictConfig) else OmegaConf.create(saved)
    if cfg is None:
        raise ValueError(f"{path} stores no config; pass the checkpoint it was exported from as well")
    weights = stored.get("state_dict", stored)
    if any(key.startswith("model.") for key in weights):  # LitModel keys
        weights = {key.removeprefix("model."): value for key, value in weights.items() if key.startswith("model.")}
    model = instantiate_model(cfg)
    model.load_state_dict(weights, strict=True)
    return model, cfg


def save_pth(path, cfg, model):
    """Lightning-free weights: ``{"cfg": <resolved config>, "state_dict": <model weights>}`` for ``load_model``."""
    torch.save({"cfg": OmegaConf.to_container(cfg, resolve=True), "state_dict": model.state_dict()}, path)
    return Path(path)


# ---- runs ------------------------------------------------------------------------------------
@contextmanager
def mha_fastpath_disabled():
    """Trace and compare with the same attention kernels: the MHA fast path is not traceable."""
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def onnx_session(path, provider="CPUExecutionProvider", optimize=True):
    """One ONNX Runtime session; ``optimize=False`` runs the graph as stored (parity of the export itself)."""
    if provider not in ort.get_available_providers():
        raise ValueError(f"Requested {provider}; available providers: {ort.get_available_providers()}")
    options = ort.SessionOptions()
    level = ort.GraphOptimizationLevel
    options.graph_optimization_level = level.ORT_ENABLE_ALL if optimize else level.ORT_DISABLE_ALL
    options.intra_op_num_threads = torch.get_num_threads()
    session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
    session.disable_fallback()
    return session


def run_onnx(session, feeds, output_names):
    """``{output name: array}``; feeds the graph does not take are dropped (absent goal / noise inputs)."""
    graph_inputs = {node.name for node in session.get_inputs()}
    values = session.run(output_names, {name: value for name, value in feeds.items() if name in graph_inputs})
    return dict(zip(output_names, values))


def compare_outputs(names, reference, observed, rtol, atol):
    """Per output: the max abs / rel error of ``observed`` against ``reference`` and whether every element meets
    ``|observed - reference| <= atol + rtol * |reference|`` (NumPy's ``assert_allclose`` rule)."""
    report = {}
    for name, expected, actual in zip(names, reference, observed):
        expected, actual = _as_float64(expected), _as_float64(actual)
        if expected.shape != actual.shape:
            raise ValueError(f"{name}: shape {actual.shape} differs from the reference {expected.shape}")
        diff = np.abs(actual - expected)
        report[name] = {
            "max_abs": float(diff.max()) if diff.size else 0.0,
            "max_rel": float((diff / np.maximum(np.abs(expected), atol)).max()) if diff.size else 0.0,
            "pass": bool(np.all(diff <= atol + rtol * np.abs(expected))),
            "finite": bool(np.isfinite(actual).all()),
        }
    return report


def _as_float64(value):
    value = value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return value.astype(np.float64)


# ---- the export's closing steps --------------------------------------------------------------
def finalize_export(
    output,
    model_onnx,
    *,
    model,
    cfg,
    checkpoint,
    reference,
    feeds,
    output_names,
    precision,
    opset,
    seed,
    decision,
    extra=None,
):
    """Cast the traced graph to ``precision`` and save it, check ONNX Runtime parity at that precision's
    tolerance (enforced for checkpoint weights, reported for untrained ones), write ``.inputs.npz``, ``.pth`` and
    ``.metadata.json`` beside it and print the summary. Returns the metadata; ``decision(model, outputs)`` is
    the family's ``(label, endpoint, lines)`` of the newest decision, ``extra`` its metadata keys."""
    output = Path(output)
    if precision != "fp32":
        logger.info(f"Casting weights to {precision} (io stays fp32)...")
    model_onnx = convert_onnx(model_onnx, precision)
    onnx.save(model_onnx, str(output))
    stored, elements = onnx_precision(model_onnx)
    logger.info(f"Saved {output} ({size_mb(output)}, {stored} weights, fp32 io, opset {opset})")

    outputs = run_onnx(onnx_session(output, optimize=False), feeds, output_names)
    rtol, atol = tolerance(precision)
    report = compare_outputs(output_names, reference, [outputs[name] for name in output_names], rtol, atol)
    errors = {name: item["max_abs"] for name, item in report.items()}
    failed = [name for name, item in report.items() if not item["pass"]]
    summary = ", ".join(f"{name}={error:.3g}" for name, error in errors.items())
    if failed and checkpoint:
        raise AssertionError(f"PyTorch/ONNX parity exceeded rtol {rtol}, atol {atol} for {failed}: {summary}")
    if failed:
        logger.warning(f"Parity tolerance exceeded for {failed} ({summary}); untrained weights, not enforced")
    logger.info(f"Nonzero-input PyTorch/ONNX parity ({precision}, rtol {rtol}, atol {atol}): {summary}")

    np.savez(output.with_suffix(".inputs.npz"), **feeds)
    pth = save_pth(output.with_suffix(".pth"), cfg, model)
    label, _, lines = decision(model, outputs)
    meta = {
        "exp_name": cfg.get("exp_name"),
        "model": type(model).__name__,
        "weights": "checkpoint" if checkpoint else "untrained",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "pth": str(pth),
        "pth_sha256": sha256_file(pth),
        "precision": precision,
        "stored_precision": stored,  # fp16 conversion keeps the initializers of a few blocked ops in fp32
        "initializer_elements": elements,
        "io_precision": "fp32",
        "opset": opset,
        "input_names": list(feeds),
        "output_names": list(output_names),
        "input_shapes": {name: list(value.shape) for name, value in feeds.items()},
        "input_dtypes": {name: str(value.dtype) for name, value in feeds.items()},
        "output_shapes": {name: list(np.asarray(value).shape) for name, value in outputs.items()},
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parity_max_abs_error": errors,
        "parity_tolerance": {"rtol": rtol, "atol": atol},
        "decision": label,
        "seed": seed,
        "config": OmegaConf.to_container(cfg, resolve=True),
        **(extra or {}),
    }
    write_metadata(output.with_suffix(".metadata.json"), meta)
    print_export_summary(meta, lines)
    return meta


def print_export_summary(meta, lines):
    """The blocks to paste in a report: source weights, artifacts, io, parity, then the decision lines."""
    print("=" * 40 + " EXPORT " + "=" * 40)
    source = meta["weights"]
    if meta.get("checkpoint"):
        source += f" {meta['checkpoint']} (sha256 {meta['checkpoint_sha256'][:12]})"
    print(f"model   : {meta['model']} {meta['parameters_total'] / 1e6:.1f}M params, {source}")
    print(
        f"onnx    : {meta['onnx']} ({size_mb(meta['onnx'])}) {meta['stored_precision']} weights, fp32 io,"
        f" opset {meta['opset']}, sha256 {meta['onnx_sha256'][:12]}"
    )
    pth = meta["pth"]
    print(f"pth     : {pth} ({size_mb(pth)}) fp32 weights + config, sha256 {meta['pth_sha256'][:12]}")
    dtypes = meta["input_dtypes"]
    print("inputs  : " + ", ".join(f"{n}{tuple(s)} {dtypes[n]}" for n, s in meta["input_shapes"].items()))
    print("outputs : " + ", ".join(f"{name}{tuple(shape)}" for name, shape in meta["output_shapes"].items()))
    tol = meta["parity_tolerance"]
    print(
        "parity  : "
        + ", ".join(f"{name} {error:.3g}" for name, error in meta["parity_max_abs_error"].items())
        + f" max abs error vs PyTorch (rtol {tol['rtol']}, atol {tol['atol']})"
    )
    print("=" * 40 + " SANITY CHECK " + "=" * 40)
    print("\n".join(lines))
