"""Export the full observation window with an explicit trajectory output contract."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from visnavkit.models.checkpoint import disable_pretrained_downloads, load_model, read, saved_config
from visnavkit.utils.common import anchor_times


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_times(cfg):
    common = cfg.common
    return anchor_times(
        float(cfg.plan_len_seconds),
        int(cfg.plan_len_points),
        common.get("offset_t_anchors", False),
        common.get("uniform_t_anchors", False),
    )


def architecture_config(model_cfg):
    """Resolved model config without initialization-only fields, for checkpoint/recipe comparison."""
    model_cfg = disable_pretrained_downloads(copy.deepcopy(model_cfg))
    return OmegaConf.to_container(model_cfg, resolve=True)


class SequencePolicy(nn.Module):
    """One decision per independent history window; no hidden feature cache.

    Goal-conditioned recipes run with their learned null goal token (goal-free inference, as in
    NoMaD exploration) and every modality encoder's null token; the metadata labels them.
    ``speed`` is emitted only by recipes with the auxiliary speed head.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder = model.action_decoder

    @property
    def output_names(self) -> list[str]:
        return ["trajectories", "scores"] + (["speed"] if self.model.vision_encoder.has_speed_head else [])

    @property
    def conditioning(self) -> str:
        parts = ["goal_free" if self.model.goal_tokens == 0 else "null_goal_token"]
        parts += [f"null_{name}_token" for name, e in self.model.modality_encoders.items() if e.num_tokens]
        return "+".join(parts)

    def forward(self, vision, initial_noise=None):
        batch, history = vision.shape[:2]
        encoded, context, _ = self.model.encode_window(vision, {})
        flat = self.decoder(context[:, -1], self.model.null_goal_tokens(batch), initial_noise).plans
        parsed = self.decoder.parse_output(flat)
        outputs = (parsed["plans"], parsed["confs"])
        if self.model.vision_encoder.has_speed_head:
            outputs += (encoded.speed.reshape(batch, history, -1)[:, -1],)
        return outputs


def export_native(cfg, output, *, checkpoint=None, seed=42, batch_size=1, model_id="base"):
    """FP32 full-window export plus metadata and nonzero-input numerical parity."""
    import onnx

    from visnavkit.benchmark.runtime import create_session

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    checkpoint_path = Path(checkpoint) if checkpoint else None
    if checkpoint_path:
        saved_cfg = saved_config(read(checkpoint_path))
        if saved_cfg is not None:
            if architecture_config(cfg.model) != architecture_config(saved_cfg.model):
                raise ValueError(
                    "Checkpoint model config differs from the requested recipe. Compose its original model/config to avoid mislabeled benchmarks."
                )
            cfg = saved_cfg
    torch.manual_seed(seed)
    model = load_model(cfg, checkpoint_path)
    parameters_total = sum(p.numel() for p in model.parameters())
    parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size = cfg.common
    h, w = int(size.crop_wh[1] // size.downscale_factor), int(size.crop_wh[0] // size.downscale_factor)
    model.vision_encoder = model.vision_encoder.prepare_for_export((h, w))
    wrapper = SequencePolicy(model).eval()
    frames = torch.rand(batch_size, int(size.seq_length), 3, h, w)
    head = model.action_decoder
    is_diffusion = head.uses_noise
    inputs = (frames,)
    names = ["vision"]
    if is_diffusion:
        inputs += (head.example_noise(batch_size),)
        names += ["initial_noise"]
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            expected = wrapper(*inputs)
            with FlopCounterMode(display=False) as counter:
                wrapper(*inputs)
            counted_flops = int(counter.get_total_flops())
            torch.onnx.export(
                wrapper,
                inputs,
                str(output),
                input_names=names,
                output_names=wrapper.output_names,
                opset_version=17,
                dynamo=False,
                # Fixed batch/context dimensions make the benchmark contract explicit.
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    onnx.checker.check_model(str(output))
    session = create_session(output)
    feeds = {name: tensor.numpy() for name, tensor in zip(names, inputs)}
    actual = session.run(None, feeds)
    parity = {}
    for name, reference, observed in zip(wrapper.output_names, expected, actual):
        target = reference.detach().numpy()
        np.testing.assert_allclose(observed, target, rtol=2e-3, atol=2e-4, err_msg=name)
        parity[name] = {"max_abs_error": float(np.max(np.abs(observed - target)))}
    has_checkpoint = checkpoint_path is not None
    meta = {
        "schema_version": 1,
        "model_id": model_id,
        "implementation_kind": "architecture_adaptation",
        "weights": "checkpoint" if has_checkpoint else "untrained_policy",
        "checkpoint_sha256": sha256_file(checkpoint_path) if has_checkpoint else None,
        "onnx_sha256": sha256_file(output),
        "inference_mode": "full_context",
        "output_names": wrapper.output_names,
        "conditioning": wrapper.conditioning,
        "precision": "float32",
        "input_shapes": {name: list(value.shape) for name, value in feeds.items()},
        "target_times_s": target_times(cfg).tolist(),
        "trajectory_units": ["meter", "meter"] + (["meter_per_second"] if head.pose_size == 3 else []),
        "selection": "unranked_samples" if is_diffusion else "highest_score",
        "num_candidates": head.num_modes,
        "denoising_steps": head.sample_steps if is_diffusion else 0,
        "action_space": head.action_space.kind,
        "parameters_total": parameters_total,
        "parameters_trainable": parameters_trainable,
        "parameter_scope": "source model, including auxiliary heads; not inferred from ONNX constants",
        "torch_counted_flops": counted_flops,
        "flop_scope": "full-context decision at exported batch size; PyTorch registered operators only, not a total",
        "parity": parity,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "seed": seed,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    np.savez(output.with_suffix(".inputs.npz"), **feeds)
    return meta
