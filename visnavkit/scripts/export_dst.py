"""Export FlowPilot-DST to a window ONNX graph and verify ONNX Runtime parity.

    uv run visnavkit-export-dst checkpoint=/path/last.ckpt output=flowpilot_dst.onnx [streaming=true]

One decision per call: the 20-slot window in, the current frame's top-k plans out. The graph has
fixed shapes (batch 1 by default), so every slot must hold a frame and a route patch; pad a short
history by repeating the oldest frame. ``docs/flowpilot_dst_onnx.md`` documents the contract.
``streaming`` (FlowPilot-DST): one frame per call, the past slots' temporal inputs carried in a buffer.
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
from visnavkit.models.vision.pair_encoder import PairEncoder
from visnavkit.scripts.export import reparameterize_model
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

INPUTS = ("vision", "route_patch", "goal", "ego", "action_bounds")
OUTPUTS = ("modes", "probs", "speed")
STREAM_INPUTS = (*INPUTS, "feat_buffer", "buffer_mask")
STREAM_OUTPUTS = (*OUTPUTS, "feat_buffer_out", "buffer_mask_out")


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


class _StreamDST(nn.Module):
    """FlowPilot-DST one frame per call, equal to the window graph's current slot (``deploy``). In: ``vision``
    (B, P, 3, H, W) the current frame, with the previous one first for a pair encoder (P = 2; zeros when there is none,
    as a window's first slot in training); ``route_patch`` (B, 1, h, w), ``goal`` (B, 1, 3), ``ego`` (B, 1, 2) of the
    current frame; ``action_bounds``; ``feat_buffer`` (B, T - 1, F) the past slots' temporal inputs [global | route],
    oldest first, and ``buffer_mask`` (B, T - 1) 1 where a slot holds one. Out: the current frame's top-k plans,
    probabilities and speed, and the buffer and mask shifted by the current frame for the next call."""

    def __init__(self, model, top_k):
        super().__init__()
        self.model, self.top_k = model, top_k
        self.num_frames = 2 if isinstance(model.pair_encoder, PairEncoder) else 1

    def encode_frame(self, vision):
        """-> global (B, C), patches (B, C, gh, gw), speed (B, 1) of the current frame; the backbone runs once."""
        enc = self.model.pair_encoder
        if not isinstance(enc, PairEncoder):  # a single-frame encoder, e.g. DuneEncoder
            glob, patches, speed, _ = enc(vision, vision.new_ones(vision.shape[:2], dtype=torch.bool))
            return glob[:, -1], patches[:, -1], speed[:, -1]
        x = torch.cat([(vision[:, 1] - enc.mean) / enc.std, (vision[:, 0] - enc.mean) / enc.std], 1)  # PairEncoder's
        patches = enc.backbone(x)[-1]
        glob = patches.mean((2, 3))
        return glob, patches, enc.speed_head(glob)

    def forward(self, vision, route_patch, goal, ego, action_bounds, feat_buffer, buffer_mask):
        m, b = self.model, vision.shape[0]
        t = feat_buffer.shape[1] + 1
        glob, patches, speed = self.encode_frame(vision)
        route = m.route_encoder(route_patch, route_patch.new_ones(b, 1, dtype=torch.bool))[:, -1].to(glob.dtype)
        cur = m.temporal_inputs(glob[:, None], route[:, None])
        valid = torch.cat([buffer_mask > 0.5, buffer_mask.new_ones(b, 1, dtype=torch.bool)], 1)
        temporal = m.temporal_encoder(torch.cat([feat_buffer, cur], 1), valid)[:, -1]
        feats = {
            "global": glob,
            "patch": patches,
            "route": route,
            "goal": m.goal(goal[:, -1] * goal.new_tensor([0.01, 1.0, 1.0])),
            "temporal": temporal,
        }
        slot = torch.full((b,), t - 1, dtype=torch.long, device=vision.device)
        embodiment = torch.full((b,), m.num_embodiments, dtype=torch.long, device=vision.device)
        kv = m.tokens(feats, slot, embodiment)
        for layer in m.context:
            kv = layer(kv)
        modes, prob = m.action_decoder.top_modes(kv, action_bounds.float(), ego[:, -1, :2].float(), self.top_k)
        buffer = torch.cat([feat_buffer[:, 1:], cur.to(feat_buffer.dtype)], 1)
        return modes, prob, speed, buffer, torch.cat([buffer_mask[:, 1:], buffer_mask.new_ones(b, 1)], 1)


def stream_inputs(stream, window, reference):
    """Stream the example ``window`` through ``stream`` slot by slot from an empty buffer and check its last decision
    against the window graph's ``reference`` (modes, probs, speed) -> the last call's inputs."""
    vision, route, goal, ego, bounds = window
    b, t = vision.shape[:2]
    width = stream.model.temporal_encoder.proj[0].in_features
    buffer, mask = vision.new_zeros(b, t - 1, width), vision.new_zeros(b, t - 1)
    prev = torch.cat([torch.zeros_like(vision[:, :1]), vision[:, :-1]], 1)
    with torch.no_grad():
        for i in range(t):
            frames = torch.stack([prev[:, i], vision[:, i]], 1) if stream.num_frames == 2 else vision[:, i : i + 1]
            args = (frames, route[:, i : i + 1], goal[:, i : i + 1], ego[:, i : i + 1], bounds, buffer, mask)
            *out, buffer, mask = stream(*args)
    gap = {n: float((a - r).abs().max()) for n, a, r in zip(OUTPUTS, out, reference)}
    logger.info("Streaming vs window, last slot: " + ", ".join(f"{k}={v:.3g}" for k, v in gap.items()))
    for n, a, r in zip(OUTPUTS, out, reference):
        np.testing.assert_allclose(a.numpy(), r.numpy(), rtol=1e-3, atol=1e-4, err_msg=f"streaming {n}")
    return args


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


def export_dst(
    cfg,
    output,
    *,
    checkpoint=None,
    batch_size=1,
    top_k=6,
    opset=17,
    seed=0,
    noise="zero",
    num_samples=1,
    streaming=False,
):
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
    inputs, names, outputs = example_inputs(cfg, batch_size, seed), INPUTS, OUTPUTS
    if streaming:
        if noise != "zero" or not isinstance(getattr(model, "tokens", None), nn.Module):
            raise ValueError("streaming exports FlowPilot-DST with noise=zero")
        with torch.no_grad():
            reference = wrapper(*inputs)
        wrapper = _StreamDST(model, top_k).eval()
        inputs, names, outputs = stream_inputs(wrapper, inputs, reference), STREAM_INPUTS, STREAM_OUTPUTS
    elif noise == "randn":
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
            output_names=list(outputs),
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
    observed = session.run(list(outputs), feeds)
    parity = {}
    for name, expected, actual in zip(outputs, reference, observed):
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
        "output_shapes": {name: list(np.asarray(value).shape) for name, value in zip(outputs, observed)},
        "streaming": streaming,
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
        streaming=cfg.get("streaming", False),
    )


if __name__ == "__main__":
    main()
