"""Export a policy to a deployment ONNX graph (single frame + feature buffer) and verify parity.

    uv run visnavkit-export checkpoint=/path/last.ckpt output=policy.onnx precision=fp16
    uv run visnavkit-export checkpoint=null model=gnm  # untrained pipeline check

Graph inputs are presence-driven: ``vision``, ``feature_buffer``, then one ``goal`` input per goal
encoder, one input per key its modality encoders read (``ego``, ``intrinsics``/``extrinsics``,
...), and ``noise`` for generative decoders.
Outputs: ``plan``, ``feat_out``, plus ``speed`` when the recipe enables the auxiliary speed head.
``precision`` (``fp32`` | ``fp16``) is the stored weight dtype; io stays fp32. Beside the graph:
``<output>.pth`` (the fp32 weights and config, Lightning-free), ``.metadata.json`` (shapes, hashes,
parity) and ``.inputs.npz`` (the traced inputs), read by ``visnavkit-check-export`` and
``visnavkit-build-engine``.
"""

import copy
from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.utils.fusion import fuse_linear_bn_eval

from visnavkit.benchmark.export import sha256_file
from visnavkit.models.action.outputs import parse_plan_output as parse_tensor_plan_output
from visnavkit.models.lit_model import LitModel
from visnavkit.utils.artifacts import (
    ONNX_PRECISIONS,
    check_precision,
    compare_outputs,
    convert_onnx,
    mha_fastpath_disabled,
    onnx_precision,
    print_export_summary,
    save_pth,
    size_mb,
    tolerance,
    write_metadata,
)
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)
EXPORT_OPTIONS = ("checkpoint", "output", "onnx_opset_version", "precision", "export_heads")


def enforce_output_order(model_onnx: onnx.ModelProto, output_names: list[str]) -> onnx.ModelProto:
    output_map = {out.name: out for out in model_onnx.graph.output}
    if not all(name in output_map for name in output_names):
        return model_onnx
    del model_onnx.graph.output[:]
    for name in output_names:
        model_onnx.graph.output.append(output_map[name])
    return model_onnx


def fuse_linear_bn_pairs(module: nn.Module) -> None:
    """Fold Linear -> BatchNorm1d pairs."""
    children = list(module.named_children())
    for (name, child), (next_name, next_child) in zip(children, children[1:]):
        if isinstance(child, nn.Linear) and isinstance(next_child, nn.BatchNorm1d):
            setattr(module, name, fuse_linear_bn_eval(child, next_child))
            setattr(module, next_name, nn.Identity())
    for _, child in module.named_children():
        fuse_linear_bn_pairs(child)


def reparameterize_model(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    fuse_linear_bn_pairs(model)
    return model


def parse_plan_output(output, M, num_pts, pose_width):
    """Single-sample NumPy helper for deployment consumers (xy trajectories only)."""
    output = np.asarray(output).reshape(1, M * (num_pts * 2 * pose_width + 1))
    parsed = parse_tensor_plan_output(torch.from_numpy(output), num_modes=M, num_pts=num_pts, pose_size=pose_width)
    return dict(
        pred_logits=output.reshape(M, -1)[:, -1],
        pred_confs=parsed["confs"][0].numpy(),
        pred_plans=parsed["plans"][0, :, :, :2].numpy(),
        best_plan=parsed["best_plan"][0, :, :2].numpy(),
    )


class _ExportPolicy(nn.Module):
    """Positional ``predict`` wrapper so absent goal/noise inputs never appear in the graph."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, *inputs):
        kwargs = dict(zip(self.policy.export_input_names(), inputs))
        goal_names = self.policy.goal_input_names()
        goal = [kwargs[name] for name in goal_names] if len(goal_names) > 1 else kwargs.get("goal")
        modalities = {name: kwargs[name] for name in self.policy.modality_input_names if name in kwargs}
        return self.policy.predict(
            kwargs["vision"], kwargs["feature_buffer"], goal=goal, noise=kwargs.get("noise"), **modalities
        )


def check_parity(model_path, feeds, reference, output_names, *, precision="fp32", strict=True, rtol=None, atol=None):
    """Run ONNX Runtime on the traced inputs and compare with the PyTorch outputs at the precision's tolerance."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    graph_inputs = {node.name for node in sess.get_inputs()}
    output = sess.run(output_names, {name: value for name, value in feeds.items() if name in graph_inputs})
    rtol, atol = tolerance(precision, rtol, atol)
    report = compare_outputs(output_names, reference, output, rtol, atol)
    errors = {name: item["max_abs"] for name, item in report.items()}
    failed = [name for name, item in report.items() if not item["pass"]]
    summary = ", ".join(f"{name}={error:.3g}" for name, error in errors.items())
    if failed and strict:
        raise AssertionError(f"PyTorch/ONNX parity exceeded rtol {rtol}, atol {atol} for {failed}: {summary}")
    if failed:
        logger.warning(f"Parity tolerance exceeded for {failed} ({summary}); untrained weights, not enforced")
    logger.info(f"Nonzero-input PyTorch/ONNX numerical parity ({precision}, rtol {rtol}, atol {atol}): {summary}")
    return output, errors


def prepare_export_config(cfg):
    """Restore checkpoint architecture/preprocessing while retaining export options."""
    cfg = copy.deepcopy(cfg)
    if cfg.checkpoint:
        checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        saved = checkpoint.get("hyper_parameters", {}).get("cfg")
        if saved is not None:
            saved = OmegaConf.create(saved)
            options = {key: cfg[key] for key in EXPORT_OPTIONS}
            cfg = OmegaConf.merge(OmegaConf.to_container(saved, resolve=True), options)
    return cfg


def prepare_policy_graph(model, cfg, *, batch_size=1, export_heads=(), seed=0, device=None):
    """The traced module, its example inputs and io names, from an eval-mode policy: the newest-frame
    reduction, folded FastViT / Linear-BatchNorm branches, ViT position embeddings for the export resolution."""
    infer_model = reparameterize_model(model).eval()
    if infer_model.temporal_encoder.reduction == "none":
        # Training predicts per frame; deployment wants the newest frame's decision only.
        infer_model.temporal_encoder.reduction = "last"
    infer_model.export_heads = [name for name in export_heads if name in infer_model.vision_encoder.heads]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    infer_model = infer_model.to(device)
    img_w = int(cfg.common.crop_wh[0] // cfg.common.downscale_factor)
    img_h = int(cfg.common.crop_wh[1] // cfg.common.downscale_factor)
    infer_model.vision_encoder = infer_model.vision_encoder.prepare_for_export((img_h, img_w))
    torch.manual_seed(seed)
    inputs = infer_model.example_inputs(batch_size, (img_h, img_w), device)
    wrapper = _ExportPolicy(infer_model).eval()
    return wrapper, inputs, infer_model.export_input_names(), infer_model.export_output_names()


def export_policy(cfg, output, *, precision=None, checkpoint=..., batch_size=1, opset=None, export_heads=None):
    """Export the deployment graph, slim it, cast it to ``precision``, verify parity and write the sidecars.

    Returns the per-output maximum absolute parity errors. Parity is enforced for checkpoints
    and reported for untrained pipeline checks.
    """
    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        if checkpoint is not ...:
            cfg.checkpoint = checkpoint
        cfg.setdefault("checkpoint", None)
        cfg.setdefault("output", str(output))
        cfg.setdefault("onnx_opset_version", 14)
        cfg.setdefault("precision", "fp16")
        cfg.setdefault("export_heads", [])
        if precision is not None:
            cfg.precision = precision
        if opset is not None:
            cfg.onnx_opset_version = opset
        if export_heads is not None:
            cfg.export_heads = list(export_heads)
    cfg = prepare_export_config(cfg)
    precision = check_precision(str(cfg.precision), ONNX_PRECISIONS)
    opset = int(cfg.onnx_opset_version)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if cfg.checkpoint is None:  # untrained weights: pipeline check
        lmodel = LitModel(cfg, initialize_pretrained=False)
    else:
        lmodel = LitModel.load_from_checkpoint(cfg.checkpoint, cfg=cfg)
    lmodel.eval()
    wrapper, inputs, input_names, output_names = prepare_policy_graph(
        lmodel.model, cfg, batch_size=batch_size, export_heads=list(cfg.export_heads)
    )
    logger.info("Export inputs: " + ", ".join(f"{name}{tuple(t.shape)}" for name, t in zip(input_names, inputs)))

    with mha_fastpath_disabled():
        with torch.no_grad():
            reference = wrapper(*inputs)
        if len(reference) != len(output_names):
            raise ValueError(f"Expected {len(output_names)} outputs ({output_names}), got {len(reference)}")
        dynamic_axes = {name: {0: "batch_size"} for name in [*input_names, *output_names]}
        logger.info("Exporting from torch model...")
        torch.onnx.export(
            wrapper,
            inputs,
            str(output),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            verbose=False,
            dynamo=False,
        )

    model_onnx = onnx.load(str(output))
    try:
        import onnxslim

        logger.info("Slimming...")
        model_onnx = onnxslim.slim(model_onnx)
    except ImportError:
        logger.warning("onnxslim not installed (uv sync --extra export); skipping graph slimming")
    if precision != "fp32":
        logger.info(f"Casting weights to {precision} (io stays fp32)...")
    model_onnx = convert_onnx(model_onnx, precision)
    model_onnx = enforce_output_order(model_onnx, output_names)  # index-based runtimes rely on the order
    onnx.save(model_onnx, str(output))
    stored, elements = onnx_precision(model_onnx)
    logger.info(f"Saved {output} ({size_mb(output)}, {stored} weights, fp32 io, opset {opset})")

    feeds = {name: tensor.cpu().numpy() for name, tensor in zip(input_names, inputs)}
    outputs, errors = check_parity(
        output, feeds, reference, output_names, precision=precision, strict=cfg.checkpoint is not None
    )
    np.savez(output.with_suffix(".inputs.npz"), **feeds)
    pth = save_pth(output.with_suffix(".pth"), cfg, lmodel.model)
    rtol, atol = tolerance(precision)
    meta = {
        "exp_name": cfg.get("exp_name"),
        "model": type(lmodel.model).__name__,
        "weights": "checkpoint" if cfg.checkpoint else "untrained",
        "checkpoint": str(cfg.checkpoint) if cfg.checkpoint else None,
        "checkpoint_sha256": sha256_file(cfg.checkpoint) if cfg.checkpoint else None,
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "pth": str(pth),
        "pth_sha256": sha256_file(pth),
        "precision": precision,
        "stored_precision": stored,  # fp16 conversion keeps the initializers of a few blocked ops in fp32
        "initializer_elements": elements,
        "io_precision": "fp32",
        "opset": opset,
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": {name: list(value.shape) for name, value in feeds.items()},
        "input_dtypes": {name: str(value.dtype) for name, value in feeds.items()},
        "output_shapes": {name: list(np.asarray(value).shape) for name, value in zip(output_names, outputs)},
        "parameters_total": sum(p.numel() for p in lmodel.model.parameters()),
        "parity_max_abs_error": errors,
        "parity_tolerance": {"rtol": rtol, "atol": atol},
        "seed": 0,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    write_metadata(output.with_suffix(".metadata.json"), meta)
    print_export_summary(meta)
    decoder = lmodel.model.action_decoder
    parsed = parse_plan_output(
        outputs[0][:1], M=decoder.num_modes, num_pts=decoder.num_pts, pose_width=decoder.pose_size
    )
    print("=" * 40 + " SANITY CHECK " + "=" * 40)
    if "speed" in output_names:
        speed = outputs[output_names.index("speed")]
        print(f"speed: {float(np.asarray(speed).reshape(-1)[0]):.4f}")
    print(f"logits: {np.round(parsed['pred_logits'], 3)}")
    print(f"best_plan p0: {np.round(parsed['best_plan'][0], 2)}")
    print(f"best_plan pN: {np.round(parsed['best_plan'][-1], 2)}")
    return errors


@hydra.main(version_base=None, config_path="../configs", config_name="export")
def main(cfg: DictConfig):
    return export_policy(cfg, cfg.output)


if __name__ == "__main__":
    main()
