"""The ``NavigationPolicy`` deployment graph: one frame plus the feature buffer in, the newest decision out.

Graph inputs are presence-driven: ``vision``, ``feature_buffer``, then one ``goal`` input per goal encoder, one
input per key its modality encoders read (``ego``, ``intrinsics``/``extrinsics``, ...), and ``noise`` for
generative decoders. Outputs: ``plan``, ``feat_out``, plus ``speed`` when the recipe enables the auxiliary speed
head, then the requested ``export_heads``. The batch axis is dynamic.
"""

import copy
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
from torch.nn.utils.fusion import fuse_linear_bn_eval

from visnavkit.export.artifacts import (
    default_device,
    finalize_export,
    instantiate_model,
    load_model,
    mha_fastpath_disabled,
)
from visnavkit.export.precision import ONNX_PRECISIONS, check_precision
from visnavkit.models.action.outputs import parse_plan_output as parse_tensor_plan_output
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
    """A deep copy with the FastViT training branches folded and Linear-BatchNorm pairs fused."""
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


def prepare_graph(model, cfg, *, batch_size=1, seed=0, export_heads=(), device=None, **_):
    """``(wrapper, inputs, input_names, output_names)`` from an eval-mode policy: the newest-frame reduction,
    folded FastViT / Linear-BatchNorm branches, ViT position embeddings for the export resolution."""
    infer_model = reparameterize_model(model).eval()
    if infer_model.temporal_encoder.reduction == "none":
        # Training predicts per frame; deployment wants the newest frame's decision only.
        infer_model.temporal_encoder.reduction = "last"
    infer_model.export_heads = [name for name in export_heads if name in infer_model.vision_encoder.heads]
    device = torch.device(device) if device else default_device()
    infer_model = infer_model.to(device)
    img_w = int(cfg.common.crop_wh[0] // cfg.common.downscale_factor)
    img_h = int(cfg.common.crop_wh[1] // cfg.common.downscale_factor)
    infer_model.vision_encoder = infer_model.vision_encoder.prepare_for_export((img_h, img_w))
    torch.manual_seed(seed)
    inputs = infer_model.example_inputs(batch_size, (img_h, img_w), device)
    wrapper = _ExportPolicy(infer_model).eval()
    return wrapper, inputs, infer_model.export_input_names(), infer_model.export_output_names()


def decision(model, outputs):
    """The newest decision: a label, its endpoint (x, y) in metres and the SANITY CHECK lines."""
    decoder = model.action_decoder
    parsed = parse_plan_output(
        np.asarray(outputs["plan"])[:1], M=decoder.num_modes, num_pts=decoder.num_pts, pose_width=decoder.pose_size
    )
    best = int(np.argmax(parsed["pred_logits"]))
    lines = [f"speed: {float(np.asarray(outputs['speed']).reshape(-1)[0]):.4f}"] if "speed" in outputs else []
    lines += [
        f"logits: {np.round(parsed['pred_logits'], 3)}",
        f"best_plan p0: {np.round(parsed['best_plan'][0], 2)}",
        f"best_plan pN: {np.round(parsed['best_plan'][-1], 2)}",
    ]
    return f"mode {best} logit={parsed['pred_logits'][best]:.3f}", parsed["best_plan"][-1].astype(np.float64), lines


def load_policy(cfg):
    """The policy and its config: a checkpoint restores its own architecture / preprocessing config with the
    export options retained; ``checkpoint=null`` composes untrained weights (pipeline check)."""
    if not cfg.checkpoint:
        return instantiate_model(cfg), cfg
    model, saved = load_model(cfg.checkpoint, cfg)
    options = {key: cfg[key] for key in EXPORT_OPTIONS}
    return model, OmegaConf.merge(OmegaConf.to_container(saved, resolve=True), options)


def export_policy(
    cfg,
    output,
    *,
    precision=None,
    checkpoint=...,
    batch_size=1,
    opset=None,
    export_heads=None,
    seed=0,
    strict=None,
    device=None,
):
    """Trace, slim, cast to ``precision``, verify parity and write the sidecars; returns the metadata.

    Keyword arguments override the export options of ``cfg`` (``export.yaml`` on top of a train config);
    ``device`` is where the reference runs (default: CUDA when usable).
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
    model, cfg = load_policy(cfg)
    precision = check_precision(str(cfg.precision), ONNX_PRECISIONS)
    opset = int(cfg.onnx_opset_version)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper, inputs, input_names, output_names = prepare_graph(
        model, cfg, batch_size=batch_size, seed=seed, export_heads=list(cfg.export_heads), device=device
    )
    logger.info("Export inputs: " + ", ".join(f"{name}{tuple(t.shape)}" for name, t in zip(input_names, inputs)))

    with mha_fastpath_disabled():
        with torch.no_grad():
            reference = wrapper(*inputs)
        if len(reference) != len(output_names):
            raise ValueError(f"Expected {len(output_names)} outputs ({output_names}), got {len(reference)}")
        logger.info("Exporting from torch model...")
        torch.onnx.export(
            wrapper,
            inputs,
            str(output),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={name: {0: "batch_size"} for name in [*input_names, *output_names]},
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
    model_onnx = enforce_output_order(model_onnx, output_names)  # index-based runtimes rely on the order
    return finalize_export(
        output,
        model_onnx,
        model=model,
        cfg=cfg,
        checkpoint=cfg.checkpoint,
        reference=reference,
        feeds={name: tensor.cpu().numpy() for name, tensor in zip(input_names, inputs)},
        output_names=output_names,
        precision=precision,
        opset=opset,
        seed=seed,
        decision=decision,
        strict=strict,
    )
