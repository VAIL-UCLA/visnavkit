"""Export FlowPilot-DST to a window ONNX graph and verify ONNX Runtime parity.

    uv run visnavkit-export-dst checkpoint=/path/last.ckpt output=flowpilot_dst.onnx

One decision per call: the 20-slot window in, the current frame's top-k plans out. The graph has
fixed shapes (batch 1 by default), so every slot must hold a frame and a route patch; pad a short
history by repeating the oldest frame. ``docs/flowpilot_dst_onnx.md`` documents the contract.
"""

import copy
import json
from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from visnavkit.benchmark.export import load_native_model, sha256_file, target_times
from visnavkit.scripts.export import reparameterize_model
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

    def forward(self, vision, route_patch, goal, ego, action_bounds, noise=None):
        if noise is None:  # no noise input: the model's own deterministic path
            return self.model.deploy(vision, goal, route_patch, ego, action_bounds, self.top_k)
        return self.model.deploy(vision, goal, route_patch, ego, action_bounds, self.top_k, noise=noise)


def example_inputs(cfg, batch_size, seed=0):
    """One window of plausible inputs: frames in [0, 1], route class ids, [distance, cos, sin], [v, w], bounds."""
    torch.manual_seed(seed)
    t = int(cfg.common.seq_length)
    w, h = (int(v // cfg.common.downscale_factor) for v in cfg.common.crop_wh)
    route_hw = tuple(cfg.dataset.val_loader.route_hw)
    classes = int(cfg.model.route.num_classes)
    goal = torch.tensor([8.0, 1.0, 0.0]).expand(batch_size, t, 3).contiguous()
    return (
        torch.rand(batch_size, t, 3, h, w),
        torch.randint(0, classes, (batch_size, t, *route_hw)).float(),
        goal,
        torch.tensor([1.0, 0.0]).expand(batch_size, t, 2).contiguous(),
        torch.tensor([[-0.0, -0.09, -0.05, 0.0, -0.85], [0.14, 0.09, 0.05, 2.76, 0.85]]).expand(batch_size, 2, 5),
    )


def export_dst(cfg, output, *, checkpoint=None, batch_size=1, top_k=6, opset=17, seed=0, noise="zero", num_samples=1):
    """Trace the window graph, check ONNX Runtime parity and write ``<output>.metadata.json``. ``noise``: ``zero`` (the
    model's deterministic path) or ``randn`` (a ``noise`` input ``(B, num_samples, T, 5)`` the caller fills with N(0, I);
    models whose ``deploy`` takes ``noise``, e.g. FlowMatchingPolicy)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    if checkpoint:
        stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved := stored.get("hyper_parameters", {}).get("cfg")) is not None:
            cfg = OmegaConf.create(saved) if isinstance(saved, dict) else saved
    model = reparameterize_model(load_native_model(cfg, checkpoint))  # fold the FastViT training branches
    unfuse_rms_norm(model)
    wrapper = _ExportDST(model, top_k).eval()
    inputs, names = example_inputs(cfg, batch_size, seed), INPUTS
    if noise == "randn":
        num_pts = model.action_decoder.num_pts
        inputs, names = (*inputs, torch.randn(batch_size, num_samples, num_pts, 5)), (*INPUTS, "noise")
    elif noise != "zero":
        raise ValueError(f"noise must be zero or randn, got {noise!r}")
    logger.info("Export inputs: " + ", ".join(f"{n}{tuple(v.shape)}" for n, v in zip(names, inputs)))

    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            reference = wrapper(*inputs)
        torch.onnx.export(
            wrapper,
            inputs,
            str(output),
            input_names=list(names),
            output_names=list(OUTPUTS),
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # fixed shapes: one window, one decision
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    onnx.checker.check_model(str(output))

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    used = {i.name for i in session.get_inputs()}  # the exporter prunes an unused input, e.g. goal without a goal token
    feeds = {name: value.numpy() for name, value in zip(names, inputs) if name in used}
    observed = session.run(list(OUTPUTS), feeds)
    parity = {}
    for name, expected, actual in zip(OUTPUTS, reference, observed):
        expected = expected.numpy()
        parity[name] = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4, err_msg=name)
    logger.info("PyTorch/ONNX parity: " + ", ".join(f"{k}={v:.3g}" for k, v in parity.items()))

    meta = {
        "exp_name": cfg.exp_name,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "onnx_sha256": sha256_file(output),
        "precision": "float32",
        "opset": opset,
        "input_shapes": {name: list(value.shape) for name, value in feeds.items()},
        "output_shapes": {name: list(np.asarray(value).shape) for name, value in zip(OUTPUTS, observed)},
        "pose_fields": ["x_m", "y_m", "yaw_rad", "v_mps", "w_radps"],
        "target_times_s": target_times(cfg).tolist(),
        "top_k": top_k,
        "noise": noise,
        "denoising_steps": model.action_decoder.sample_steps,
        "num_anchors": int(getattr(model.action_decoder, "anchors", torch.empty(0)).shape[0]),  # 0: no anchors
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parity_max_abs_error": parity,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savez(output.with_suffix(".inputs.npz"), **feeds)
    logger.info(f"Saved {output} ({output.stat().st_size / 2**20:.1f}MB), metadata and sample inputs")
    return meta


@hydra.main(version_base=None, config_path="../configs", config_name="export_dst")
def main(cfg: DictConfig):
    return export_dst(
        cfg,
        cfg.output,
        checkpoint=cfg.checkpoint,
        batch_size=cfg.batch_size,
        top_k=cfg.top_k,
        opset=cfg.onnx_opset_version,
        noise=cfg.noise,
        num_samples=cfg.num_samples,
    )


if __name__ == "__main__":
    main()
