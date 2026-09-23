"""The FlowPilot-DST window graph: the 20-slot window in, the current frame's top-k plans out.

Shapes are fixed (batch 1 by default), so every slot must hold a frame and a route patch; pad a short history by
repeating the oldest frame. ``docs/flowpilot_dst_onnx.md`` documents the contract.
"""

from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from visnavkit.benchmark.export import target_times
from visnavkit.export.artifacts import finalize_export, instantiate_model, load_model, mha_fastpath_disabled
from visnavkit.export.policy import reparameterize_model
from visnavkit.export.precision import ONNX_PRECISIONS, check_precision
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

INPUTS = ("vision", "route_patch", "goal", "ego", "action_bounds")
OUTPUTS = ("modes", "probs", "speed")


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


class _ExportDST(nn.Module):
    """Positional wrapper around ``FlowPilotDST.deploy`` for the tracer."""

    def __init__(self, model, top_k):
        super().__init__()
        self.model, self.top_k = model, top_k

    def forward(self, vision, route_patch, goal, ego, action_bounds):
        return self.model.deploy(vision, goal, route_patch, ego, action_bounds, self.top_k)


def example_inputs(model, cfg, batch_size, seed=0):
    """One window of plausible inputs: frames in [0, 1], route class ids, [distance, cos, sin], [v, w], bounds."""
    torch.manual_seed(seed)
    t = int(cfg.common.seq_length)
    w, h = (int(v // cfg.common.downscale_factor) for v in cfg.common.crop_wh)
    route = model.route_encoder
    goal = torch.tensor([8.0, 1.0, 0.0]).expand(batch_size, t, 3).contiguous()
    return (
        torch.rand(batch_size, t, 3, h, w),
        torch.randint(0, route.num_classes, (batch_size, t, *route.hw)).float(),
        goal,
        torch.tensor([1.0, 0.0]).expand(batch_size, t, 2).contiguous(),
        torch.tensor([[-0.0, -0.09, -0.05, 0.0, -0.85], [0.14, 0.09, 0.05, 2.76, 0.85]]).expand(batch_size, 2, 5),
    )


def prepare_graph(model, cfg, *, batch_size=1, seed=0, top_k=6, **_):
    """``(wrapper, inputs, input_names, output_names)`` from an eval-mode model: folded FastViT branches,
    RMSNorm in primitive ops; the caller's model keeps its structure (deep copy)."""
    model = unfuse_rms_norm(reparameterize_model(model))
    wrapper = _ExportDST(model, top_k).eval()
    return wrapper, example_inputs(model, cfg, batch_size, seed), list(INPUTS), list(OUTPUTS)


def decision(model, outputs):
    """The top plan: a label, its endpoint (x, y) in metres and the SANITY CHECK lines."""
    modes, probs, speed = (np.asarray(outputs[name]) for name in OUTPUTS)
    lines = [
        f"probs: {np.round(probs[0], 3)}",
        f"best plan p0 [x y yaw v w]: {np.round(modes[0, 0, 0], 3)}",
        f"best plan pN [x y yaw v w]: {np.round(modes[0, 0, -1], 3)}",
        f"speed: {float(speed.reshape(-1)[0]):.4f}",
    ]
    return f"top plan p={probs[0, 0]:.3f}", modes[0, 0, -1, :2].astype(np.float64), lines


def export_dst(cfg, output, *, checkpoint=None, batch_size=1, top_k=6, opset=17, seed=0, precision="fp32"):
    """Trace the window graph, check ONNX Runtime parity and write the sidecars; returns the metadata."""
    check_precision(precision, ONNX_PRECISIONS)
    model, cfg = load_model(checkpoint, cfg) if checkpoint else (instantiate_model(cfg), cfg)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper, inputs, input_names, output_names = prepare_graph(
        model, cfg, batch_size=batch_size, seed=seed, top_k=top_k
    )
    logger.info("Export inputs: " + ", ".join(f"{n}{tuple(v.shape)}" for n, v in zip(input_names, inputs)))

    with mha_fastpath_disabled():
        with torch.no_grad():
            reference = wrapper(*inputs)
        torch.onnx.export(
            wrapper,
            inputs,
            str(output),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # fixed shapes: one window, one decision
        )
    onnx.checker.check_model(str(output))
    return finalize_export(
        output,
        onnx.load(str(output)),
        model=model,
        cfg=cfg,
        checkpoint=checkpoint,
        reference=reference,
        feeds={name: value.numpy() for name, value in zip(input_names, inputs)},
        output_names=output_names,
        precision=precision,
        opset=opset,
        seed=seed,
        decision=decision,
        extra={
            "pose_fields": ["x_m", "y_m", "yaw_rad", "v_mps", "w_radps"],
            "target_times_s": target_times(cfg).tolist(),
            "top_k": top_k,
            "denoising_steps": model.action_decoder.sample_steps,
            "num_anchors": int(model.action_decoder.anchors.shape[0]),
        },
    )
