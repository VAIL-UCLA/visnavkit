"""Export FlowPilot-DST to a window ONNX graph and verify ONNX Runtime parity.

    uv run visnavkit-export-dst checkpoint=/path/last.ckpt output=flowpilot_dst.onnx precision=fp32

One decision per call: the 20-slot window in, the current frame's top-k plans out. The graph has
fixed shapes (batch 1 by default), so every slot must hold a frame and a route patch; pad a short
history by repeating the oldest frame. ``docs/flowpilot_dst_onnx.md`` documents the contract.
``precision`` (``fp32`` | ``fp16``) is the stored weight dtype; io stays fp32. Beside the graph:
``<output>.pth`` (fp32 weights and config), ``.metadata.json`` and ``.inputs.npz`` (the traced
inputs), read by ``visnavkit-check-export`` and ``visnavkit-build-engine``.
"""

import copy
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


def prepare_dst_graph(model, cfg, *, batch_size=1, top_k=6, seed=0):
    """The traced module, its example inputs and io names, from an eval-mode model: folded FastViT branches,
    RMSNorm in primitive ops."""
    model = unfuse_rms_norm(reparameterize_model(model))  # deep copy: the caller's model keeps its structure
    wrapper = _ExportDST(model, top_k).eval()
    return wrapper, example_inputs(model, cfg, batch_size, seed), list(INPUTS), list(OUTPUTS)


def export_dst(cfg, output, *, checkpoint=None, batch_size=1, top_k=6, opset=17, seed=0, precision="fp32"):
    """Trace the window graph, check ONNX Runtime parity and write the ``.pth``, metadata and sample inputs."""
    check_precision(precision, ONNX_PRECISIONS)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    if checkpoint:
        stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved := stored.get("hyper_parameters", {}).get("cfg")) is not None:
            cfg = OmegaConf.create(saved) if isinstance(saved, dict) else saved
    native = load_native_model(cfg, checkpoint)
    wrapper, inputs, input_names, output_names = prepare_dst_graph(
        native, cfg, batch_size=batch_size, top_k=top_k, seed=seed
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
    if precision != "fp32":
        logger.info(f"Casting weights to {precision} (io stays fp32)...")
    model_onnx = convert_onnx(onnx.load(str(output)), precision)
    onnx.save(model_onnx, str(output))
    stored_precision, elements = onnx_precision(model_onnx)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    feeds = {name: value.numpy() for name, value in zip(input_names, inputs)}
    observed = session.run(output_names, feeds)
    rtol, atol = tolerance(precision)
    report = compare_outputs(output_names, reference, observed, rtol, atol)
    parity = {name: item["max_abs"] for name, item in report.items()}
    failed = [name for name, item in report.items() if not item["pass"]]
    if failed:
        raise AssertionError(f"PyTorch/ONNX parity exceeded rtol {rtol}, atol {atol} for {failed}: {parity}")
    logger.info(f"PyTorch/ONNX parity ({precision}): " + ", ".join(f"{k}={v:.3g}" for k, v in parity.items()))

    pth = save_pth(output.with_suffix(".pth"), cfg, native)
    np.savez(output.with_suffix(".inputs.npz"), **feeds)
    meta = {
        "exp_name": cfg.exp_name,
        "model": type(native).__name__,
        "weights": "checkpoint" if checkpoint else "untrained",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "pth": str(pth),
        "pth_sha256": sha256_file(pth),
        "precision": precision,
        "stored_precision": stored_precision,
        "initializer_elements": elements,
        "io_precision": "fp32",
        "opset": opset,
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": {name: list(value.shape) for name, value in feeds.items()},
        "input_dtypes": {name: str(value.dtype) for name, value in feeds.items()},
        "output_shapes": {name: list(np.asarray(value).shape) for name, value in zip(output_names, observed)},
        "pose_fields": ["x_m", "y_m", "yaw_rad", "v_mps", "w_radps"],
        "target_times_s": target_times(cfg).tolist(),
        "top_k": top_k,
        "denoising_steps": native.action_decoder.sample_steps,
        "num_anchors": int(native.action_decoder.anchors.shape[0]),
        "parameters_total": sum(p.numel() for p in native.parameters()),
        "parity_max_abs_error": parity,
        "parity_tolerance": {"rtol": rtol, "atol": atol},
        "seed": seed,
    }
    write_metadata(output.with_suffix(".metadata.json"), meta)
    print_export_summary(meta)
    modes, probs, speed = (np.asarray(value) for value in observed)
    print("=" * 40 + " SANITY CHECK " + "=" * 40)
    print(f"probs: {np.round(probs[0], 3)}")
    print(f"best plan p0 [x y yaw v w]: {np.round(modes[0, 0, 0], 3)}")
    print(f"best plan pN [x y yaw v w]: {np.round(modes[0, 0, -1], 3)}")
    print(f"speed: {float(speed.reshape(-1)[0]):.4f}")
    logger.info(f"Saved {output} ({size_mb(output)}), {pth.name}, metadata and sample inputs")
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
        precision=cfg.precision,
    )


if __name__ == "__main__":
    main()
