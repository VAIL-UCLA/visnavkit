"""Check a configured policy on synthetic inputs: pipeline shapes, a training forward/backward,
and the three inference views agreeing — ``act`` (window -> actions), ``forward``'s newest
decision and ``predict`` through the deployment feature buffer.

python -m visnavkit.scripts.sanity_check
python -m visnavkit.scripts.sanity_check model/vision_encoder=resnet18 model/temporal_encoder=bidirectional
python -m visnavkit.scripts.sanity_check --onnx model/action_decoder=flow_dit model/goal_encoder=point
python -m visnavkit.scripts.sanity_check model/modality_encoder=ego_camera
"""

import argparse
import copy
from pathlib import Path

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from visnavkit.models.lit_model import build_targets, disable_pretrained_downloads


def _params(module) -> str:
    """Trainable / total parameters of one stage, so the diagram also sizes the model."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    scale = "M" if total >= 1e6 else "k"
    divisor = 1e6 if total >= 1e6 else 1e3
    return f"{trainable / divisor:.2f}/{total / divisor:.2f}{scale} params"


def _goal_shapes(goal):
    if goal is None:
        return []
    return [list(g.shape) if g is not None else None for g in (goal if isinstance(goal, list) else [goal])]


def _pipeline(model, vision, goal, modalities, output, cfg):
    batch, sequence, channels, height, width = vision.shape
    tokens = output.vision.tokens
    per_frame = tokens.reshape(batch, sequence, model.vision_tokens, model.feat_size)
    per_frame = model._per_frame_tokens(per_frame, output.modality_tokens or {}, dim=2)
    context = model.temporal_encoder(per_frame)
    plans = output.plan.plans
    decoder = model.action_decoder
    trajectories = decoder.parse_output(plans)["plans"]
    goal_shapes = _goal_shapes(goal)
    goal_names = [type(encoder).__name__ for encoder in model.goal_encoders]
    goal_line = (
        "(no goal tokens)"
        if output.goal_tokens is None
        else f"{goal_shapes} -> tokens {list(output.goal_tokens.shape)}"
    )
    modality_lines = [
        f"+-- modality '{name}': {type(model.modality_encoders[name]).__name__} "
        f"({_params(model.modality_encoders[name])}) "
        f"{[list(modalities[key].shape) for key in model.modality_encoders[name].input_names]}"
        f" -> tokens {list(tokens.shape)}"
        for name, tokens in (output.modality_tokens or {}).items()
    ] or ["+-- modalities: (none)"]
    lines = [
        f"{type(model).__name__} ({_params(model)})",
        "|",
        f"+-- vision {list(vision.shape)}  [batch, frames, RGB, height, width]",
        f"|   `-- flatten frames -> {[batch * sequence, channels, height, width]}",
        f"+-- vision_encoder: {type(model.vision_encoder).__name__} ({cfg.model.vision_encoder.backbone_name},"
        f" {model.vision_encoder.token_mode}, {_params(model.vision_encoder)})",
        f"|   `-- tokens -> {list(tokens.shape)}  [frames, tokens per frame, feat_size]",
        *modality_lines,
        f"+-- temporal_encoder: {type(model.temporal_encoder).__name__} ({_params(model.temporal_encoder)})",
        f"|   `-- reduction={model.temporal_encoder.reduction} -> {list(context.shape)}  [batch, decisions, tokens, feat_size]",
        f"+-- goal_encoder: {goal_names} {goal_line} ({_params(model.goal_encoders)})",
        f"`-- action_decoder: {type(decoder).__name__} ({decoder.action_space.kind}, {decoder.num_modes} modes,"
        f" {_params(decoder)})",
    ]
    if model.vision_encoder.has_speed_head:
        lines.insert(6, f"|   `-- speed (auxiliary) -> {list(output.vision.speed.shape)}")
    if hasattr(decoder, "denoiser"):
        lines.append(
            f"    +-- denoiser: {type(decoder.denoiser).__name__} x {decoder.sample_steps} steps ({type(decoder.scheduler).__name__})"
        )
    lines += [
        f"    +-- plans (flat) -> {list(plans.shape)}",
        f"    `-- trajectories -> {list(trajectories.shape)}  [decisions, modes, points, pose dimensions]",
    ]
    return "\n".join(lines)


def _newest_goal(model, goal):
    """Deployment passes one goal per encoder, already reduced to the newest frame."""
    values = goal if isinstance(goal, list) else [goal]
    newest = [
        None if value is None else (value[:, -1] if encoder.per_frame else value)
        for encoder, value in zip(model.goal_encoders, values)
    ]
    return newest if isinstance(goal, list) else newest[0]


@torch.no_grad()
def _check_feature_buffer(model, vision, goal, modalities, seq_step):
    exported = copy.deepcopy(model).eval()
    temporal = exported.temporal_encoder
    if temporal.reduction == "none":
        temporal.reduction = "last"
    batch, sequence = vision.shape[:2]
    noise = exported.action_decoder.example_noise(batch) if exported.action_decoder.uses_noise else None
    expected = exported(vision, goal=goal, noise=noise, **modalities)
    features = model._per_frame_tokens(
        expected.vision.tokens.reshape(batch, sequence, model.vision_tokens, model.feat_size),
        expected.modality_tokens or {},
        dim=2,
    ).flatten(2)
    history_size = (sequence - 1) * seq_step
    buffer = torch.randn(batch, history_size, model.token_dim)
    # Unselected slots represent intervening frames and must not affect the decision.
    buffer[:, ::seq_step] = features[:, :-1]
    outputs = exported.predict(
        vision[:, -1],
        buffer,
        goal=_newest_goal(exported, goal),
        noise=noise,
        **{name: value[:, -1] for name, value in modalities.items()},
    )
    plans, token = outputs[0], outputs[1]
    torch.testing.assert_close(plans, expected.plan.plans, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(token, features[:, -1])
    if model.vision_encoder.has_speed_head:
        torch.testing.assert_close(outputs[2], expected.vision.speed.reshape(batch, sequence, -1)[:, -1])
    return history_size, temporal.reduction, exported.action_decoder.parse_output(expected.plan.plans), noise


@torch.no_grad()
def _check_act(model, vision, goal, modalities, noise, expected):
    """``act`` on the training model (any reduction) returns the newest decision of the window."""
    batch = vision.shape[0]
    decoder = model.action_decoder
    trajectories, scores = model.act(vision, goal=goal, noise=noise, **modalities)
    assert tuple(trajectories.shape) == (batch, decoder.num_modes, decoder.num_pts, decoder.pose_size)
    assert tuple(scores.shape) == (batch, decoder.num_modes)
    assert torch.isfinite(trajectories).all() and torch.allclose(scores.sum(1), torch.ones(batch))
    torch.testing.assert_close(trajectories, expected["plans"], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(scores, expected["confs"], rtol=2e-4, atol=2e-5)
    return list(trajectories.shape), list(scores.shape)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=gnm model/action_decoder=mhp")
    parser.add_argument(
        "--onnx", action="store_true", help="Also export the deployment ONNX graph and verify ONNX Runtime parity"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sanity"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for small synthetic checks")
    args = parser.parse_intermixed_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(
            config_name="train",
            overrides=["common.seq_length=3", "common.crop_wh=[64,64]", "common.downscale_factor=1", *args.overrides],
        )
    disable_pretrained_downloads(cfg.model)
    model = instantiate(cfg.model).cpu().eval()
    batch, sequence = args.batch_size, int(cfg.common.seq_length)
    width, height = (int(value // cfg.common.downscale_factor) for value in cfg.common.crop_wh)
    if batch * sequence < 2:
        parser.error("Training BatchNorm requires batch-size * common.seq_length >= 2")
    vision, goal, modalities = model.example_batch(batch, sequence, (height, width))
    decoder = model.action_decoder
    reduction = model.temporal_encoder.reduction
    decisions = batch * sequence if reduction == "none" else batch

    with torch.no_grad():
        output = model(vision, goal=goal, **modalities)
        assert tuple(output.vision.tokens.shape) == (batch * sequence, model.vision_tokens, model.feat_size), (
            "Unexpected token shape"
        )
        assert torch.isfinite(output.vision.tokens).all(), "Nonfinite vision tokens"
        if model.vision_encoder.has_speed_head:
            assert tuple(output.vision.speed.shape) == (batch * sequence, 1), "Unexpected speed shape"
        plans = output.plan.plans
        assert tuple(plans.shape) == (decisions, decoder.flat_size), "Unexpected action output shape"
        assert torch.isfinite(plans).all(), "Nonfinite action outputs"
        diagram = _pipeline(model, vision, goal, modalities, output, cfg)
    print(diagram)
    print("\n[PASS] Forward shapes and finite outputs")

    model.train()
    targets = build_targets(
        {
            "frame_speeds": torch.rand(batch, sequence, 1),
            "future_poses": torch.rand(batch, sequence, decoder.num_pts, decoder.pose_size),
        },
        action_reduction=reduction,
    )
    model.zero_grad(set_to_none=True)
    losses, _ = model.get_losses(model(vision, goal=goal, **modalities), targets)
    assert all(torch.isfinite(value).all() for value in losses.values()), "Nonfinite training loss"
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients), "Missing or nonfinite gradients"
    assert any(gradient.abs().sum() > 0 for gradient in gradients), "All gradients are zero"
    print(f"[PASS] Training loss={losses['loss'].item():.6f}; backward gradients finite")
    model.zero_grad(set_to_none=True)
    model.eval()
    history_size, export_reduction, expected, noise = _check_feature_buffer(
        model, vision, goal, modalities, int(cfg.model.export_cfg.seq_step)
    )
    print(f"[PASS] Feature-buffer parity (history={history_size}, reduction={export_reduction})")
    trajectories, scores = _check_act(model, vision, goal, modalities, noise, expected)
    print(f"[PASS] act(): trajectories {trajectories} [batch, modes, points, pose], scores {scores}; matches predict")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = args.output_dir / "pipeline.txt"
    pipeline_path.write_text(diagram + "\n")
    print(f"Pipeline: {pipeline_path}")

    if args.onnx:
        from visnavkit.export.policy import export_policy

        model_path = args.output_dir / "model.onnx"
        meta = export_policy(cfg, model_path, precision="fp32", checkpoint=None, batch_size=batch)
        parity = meta["parity_max_abs_error"]
        errors = ", ".join(f"{name}={value:.3g}" for name, value in parity.items())
        print(f"[PASS] ONNX Runtime parity (maximum absolute errors: {errors})")
        print(f"ONNX: {model_path}")


if __name__ == "__main__":
    main()
