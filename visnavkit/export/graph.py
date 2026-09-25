"""A model's deployment ONNX graph at a precision, with its ``.pth`` / ``.metadata.json`` / ``.inputs.npz`` sidecars.

The model owns its graph: ``export_graph(cfg, batch_size, **options)`` returns the traced wrapper, its example
inputs and the io names; ``decision(outputs)`` reads the newest decision back from NumPy outputs. Shapes are
fixed at export (``batch_size``): TensorRT builds them without a profile.
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


class _RMSNorm(nn.Module):
    """``nn.RMSNorm`` in primitive ops: ``aten::rms_norm`` has no ONNX lowering below opset 23."""

    def __init__(self, src: nn.RMSNorm):
        super().__init__()
        self.weight = src.weight
        self.eps = src.eps

    def forward(self, x):
        eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)
        return x if self.weight is None else x * self.weight


def unfuse_rms_norm(module):
    for name, child in module.named_children():
        unfuse_rms_norm(child) if not isinstance(child, nn.RMSNorm) else setattr(module, name, _RMSNorm(child))
    return module


def fuse_linear_bn_pairs(module: nn.Module) -> None:
    """Fold Linear -> BatchNorm1d pairs."""
    children = list(module.named_children())
    for (name, child), (next_name, next_child) in zip(children, children[1:]):
        if isinstance(child, nn.Linear) and isinstance(next_child, nn.BatchNorm1d):
            setattr(module, name, fuse_linear_bn_eval(child, next_child))
            setattr(module, next_name, nn.Identity())
    for _, child in module.named_children():
        fuse_linear_bn_pairs(child)


def reparameterize_model(model: nn.Module) -> nn.Module:
    """An eval-mode deep copy with the FastViT training branches folded, Linear-BatchNorm pairs fused and
    RMSNorm in primitive ops."""
    model = copy.deepcopy(model).eval()
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    fuse_linear_bn_pairs(model)
    return unfuse_rms_norm(model)


def enforce_output_order(model_onnx: onnx.ModelProto, output_names: list[str]) -> onnx.ModelProto:
    output_map = {out.name: out for out in model_onnx.graph.output}
    if not all(name in output_map for name in output_names):
        return model_onnx
    del model_onnx.graph.output[:]
    for name in output_names:
        model_onnx.graph.output.append(output_map[name])
    return model_onnx


def parse_plan_output(output, M, num_pts, pose_width):
    """Single-sample NumPy helper for deployment consumers of the policy's ``plan`` (xy trajectories only)."""
    output = np.asarray(output).reshape(1, M * (num_pts * 2 * pose_width + 1))
    parsed = parse_tensor_plan_output(torch.from_numpy(output), num_modes=M, num_pts=num_pts, pose_size=pose_width)
    return dict(
        pred_logits=output.reshape(M, -1)[:, -1],
        pred_confs=parsed["confs"][0].numpy(),
        pred_plans=parsed["plans"][0, :, :, :2].numpy(),
        best_plan=parsed["best_plan"][0, :, :2].numpy(),
    )


def prepare_graph(model, cfg, *, batch_size=1, seed=0, device=None, **options):
    """``(wrapper, inputs, input_names, output_names)``: the model's own ``export_graph`` on a folded copy placed
    on ``device`` (default: CUDA when usable), with seeded example inputs. ``model`` keeps its structure."""
    device = torch.device(device) if device else default_device()
    torch.manual_seed(seed)
    return reparameterize_model(model).to(device).export_graph(cfg, batch_size, **options)


def load(cfg):
    """The model and its config: a checkpoint restores its own architecture / preprocessing config with the
    export options retained; ``checkpoint=null`` composes untrained weights (pipeline check)."""
    if not cfg.checkpoint:
        return instantiate_model(cfg), cfg
    model, saved = load_model(cfg.checkpoint, cfg)
    options = {key: cfg[key] for key in EXPORT_OPTIONS}
    return model, OmegaConf.merge(OmegaConf.to_container(saved, resolve=True), options)


def export_onnx(
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
    model, cfg = load(cfg)
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
        strict=strict,
    )
