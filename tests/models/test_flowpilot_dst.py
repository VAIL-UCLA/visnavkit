"""FlowPilot-DST: masked pair encoding, per-frame kv tokens, the anchored flow head and its two-mode output."""

from types import SimpleNamespace

import numpy as np

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from torchvision.io import decode_png, read_file

from visnavkit.models.lit_model import ROUTE_COLORS, LitModel, build_targets, disable_pretrained_downloads, route_images

SMALL = [
    "model=flowpilot_dst",
    "model.pretrained=false",
    "model.backbone_name=fastvit_t8",
    "model.dim=32",
    "model.temporal.num_layers=1",
    "model.temporal.num_heads=2",
    "model.head.num_layers=1",
    "model.head.num_heads=2",
    "model.head.num_anchors=4",
    "model.head.wp_dim=8",
    "plan_len_points=8",
    "plan_len_seconds=0.4",
    "common.seq_length=4",
    "common.uniform_t_anchors=true",
]
FRAME_MASK = torch.tensor([[True, True, True, True], [True, False, False, True]])


def make_model(*overrides):
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="train", overrides=[*SMALL, *overrides])
    disable_pretrained_downloads(cfg.model)
    return instantiate(cfg.model)


def make_batch(b=2, t=4, hw=(32, 64)):
    """Straight 1 m/s motion; window 1 has no frames in its middle slots and no route."""
    times = torch.arange(1, 9) * 0.05
    future = torch.zeros(b, t, 8, 5)
    future[..., 0], future[..., 3] = times, 1.0
    bounds = torch.tensor([[-0.1, -0.1, -0.1, -0.1, -1.0], [0.3, 0.1, 0.1, 3.0, 1.0]])
    return dict(
        vision=torch.rand(b, t, 3, *hw),
        frame_mask=FRAME_MASK.clone(),
        route_patch=torch.randint(0, 3, (b, t, 80, 80)).float(),
        route_mask=torch.tensor([[True] * t, [False] * t]),
        goal=torch.tensor([10.0, 1.0, 0.0]).expand(b, t, 3).clone(),
        ego=torch.ones(b, t, 2),
        embodiment_id=torch.tensor([0, 3]),
        action_bounds=bounds.expand(b, 2, 5).clone(),
        future_poses=future,
        frame_speeds=torch.ones(b, t, 1),
        target_times_s=times.expand(b, 8).clone(),
    )


def inputs(batch):
    keys = ("frame_mask", "route_patch", "route_mask", "ego", "embodiment_id", "action_bounds")
    return batch["vision"], batch["goal"], {k: batch[k] for k in keys}


def test_training_supervises_the_frames_that_hold_a_frame_and_the_whole_pairs():
    model, batch = make_model().train(), make_batch()
    vision, goal, mods = inputs(batch)
    out = model(vision, goal, **mods)
    assert torch.equal(out.plan.idx, FRAME_MASK.reshape(-1).nonzero().squeeze(1))
    assert out.plan.tokens.shape == (6, 1 + 2 + 3 + 1, 32)  # global, 1 x 2 patches, route, goal, temporal, embodiment
    assert torch.all(out.speed[~FRAME_MASK.reshape(-1)] == 0)
    pair = out.pair_mask.reshape(2, 4)
    assert not pair[:, 0].any() and not pair[1, 3]  # no previous slot at all, or an empty one
    assert (out.plan.plans == 0).all()  # generative: no sampling while training

    losses, _ = model.get_losses(out, build_targets(batch))
    assert torch.isfinite(losses["loss"]) and {"action_reg", "action_cls", "vision_speed"} <= losses.keys()
    losses["loss"].backward()
    assert model.pair_encoder.backbone.stem_0.conv_kxk[0].conv.weight.grad is not None
    assert model.action_decoder.score[-1].weight.grad is not None
    assert not any(p.requires_grad for p in model.route_encoder.parameters())
    assert not model.route_encoder.training


def test_the_embodiment_token_can_be_left_out():
    model, batch = make_model("model.embodiment_token=false").train(), make_batch()
    out = model(*inputs(batch)[:2], **inputs(batch)[2])
    assert model.tokens.embodiment is None
    assert out.plan.tokens.shape == (6, 1 + 2 + 3, 32)  # global, 1 x 2 patches, route, goal, temporal


def test_deploy_matches_the_top_modes_of_an_eval_forward():
    """The export path decodes the same plans as the training-time panels read."""
    model, batch = make_model("model.embodiment_token=false").eval(), make_batch()
    batch["frame_mask"] = torch.ones_like(FRAME_MASK)
    batch["route_mask"] = torch.ones_like(FRAME_MASK)
    vision, goal, mods = inputs(batch)
    with torch.no_grad():
        out = model(vision, goal, **mods)
        windows, modes, probs, _ = model.current_modes(out.plan, seq_len=4, k=3)
        deployed, deployed_probs, speed = model.deploy(
            vision, goal, batch["route_patch"], batch["ego"], batch["action_bounds"], k=3
        )
    assert torch.equal(windows, torch.tensor([0, 1]))
    torch.testing.assert_close(deployed, modes)
    torch.testing.assert_close(deployed_probs, probs)
    torch.testing.assert_close(speed, out.speed.reshape(2, 4, 1)[:, -1])


def test_speed_head_is_supervised_on_whole_pairs_only():
    """An empty or dropped previous slot is black to the network and takes the slot's speed target with it."""
    model, batch = make_model().train(), make_batch()
    for part in (model.pair_encoder.backbone, model.pair_encoder.speed_head):
        part.eval()  # running BatchNorm statistics: the rows stay independent
    vision, goal, mods = inputs(batch)
    model.pair_encoder.p_drop_prev = 0.0
    out = model(vision, goal, **mods)
    pair = out.pair_mask.reshape(2, 4)
    assert pair.tolist() == [[False, True, True, True], [False] * 4]  # window 1: what a corpus below 20 fps gives
    speed = model.get_losses(out, build_targets(batch))[0]["vision_speed"]
    elsewhere = dict(batch, frame_speeds=torch.where(pair[..., None], batch["frame_speeds"], torch.tensor(50.0)))
    assert speed > 0 and model.get_losses(out, build_targets(elsewhere))[0]["vision_speed"] == speed

    other_past = vision.clone()
    other_past[0, 2] = torch.rand(3, 32, 64)  # slot 3's previous frame
    assert not torch.equal(model(other_past, goal, **mods).speed[3], out.speed[3])
    model.pair_encoder.p_drop_prev = 1.0  # every previous frame dropped: black, and no whole pair is left
    dropped = model(vision, goal, **mods)
    assert not dropped.pair_mask.any()
    torch.testing.assert_close(model(other_past, goal, **mods).speed[3], dropped.speed[3])

    sparse = {k: v[1:] for k, v in batch.items()}  # the window with gaps alone: plans to learn, no speed target
    losses, _ = model.get_losses(model(*inputs(sparse)[:2], **inputs(sparse)[2]), build_targets(sparse))
    assert losses["vision_speed"] == 0 and torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grad = model.pair_encoder.speed_head.head[0].weight.grad
    assert grad is not None and not grad.any()  # still in the graph (DDP wants every parameter), without a signal


def test_eval_decodes_two_modes_and_leaves_rows_without_frames_empty():
    model, batch = make_model().eval(), make_batch()
    vision, goal, mods = inputs(batch)
    with torch.no_grad():
        out = model(vision, goal, **mods)
    assert out.plan.plans.shape == (8, 2 * (2 * 8 * 5 + 1))
    parsed = model.action_decoder.parse_output(out.plan.plans)
    assert parsed["plans"].shape == (8, 2, 8, 5) and parsed["confs"].shape == (8, 2)
    assert torch.all(out.plan.plans[~out.plan.valid] == 0) and torch.isfinite(parsed["plans"]).all()
    assert torch.equal(out.plan.valid, FRAME_MASK.reshape(-1))

    # the noise-0 mode is deterministic; black frames in the empty slots never reach the network
    garbage = dict(batch, vision=batch["vision"].clone())
    garbage["vision"][1, 1:3] = torch.rand(2, 3, 32, 64)
    with torch.no_grad():
        again = model(*inputs(garbage)[:2], **inputs(garbage)[2])
    torch.testing.assert_close(
        parsed["plans"][:, 0], model.action_decoder.parse_output(again.plan.plans)["plans"][:, 0]
    )
    with torch.no_grad():
        no_goal = model(vision, None, **mods)
    assert torch.isfinite(no_goal.plan.plans).all()


def test_example_batch_runs_the_smoke_path():
    model = make_model().eval()
    vision, goal, mods = model.example_batch(2, 4, (32, 64))
    with torch.no_grad():
        out = model(vision, goal, **mods)
    assert out.plan.plans.shape == (8, model.action_decoder.flat_size)


def test_route_images_are_collected_over_batches_until_the_sample_count(tmp_path):
    batch = make_batch()
    batch["route_mask"] = torch.tensor([[False] * 4, [True] * 4])  # only window 1 has a route
    (image,) = route_images(batch)
    assert image.shape == (3, 32, 64 + 32) and image.dtype == torch.uint8
    frame = (batch["vision"][1, -1] * 255).round().to(torch.uint8)
    assert torch.equal(image[..., :64], frame)
    patch = ROUTE_COLORS[batch["route_patch"][1, -1].long()].permute(2, 0, 1)  # 80 x 80 -> 32 x 32, nearest
    assert torch.equal(image[..., 64:], patch[:, (torch.arange(32) * 2.5).long()][..., (torch.arange(32) * 2.5).long()])
    assert route_images({**batch, "route_mask": torch.zeros(2, 4, dtype=torch.bool)}) == []

    logged = []
    wandb_like = SimpleNamespace(log_image=lambda **kwargs: logged.append(kwargs))
    model = make_model().train()
    lit = SimpleNamespace(
        trainer=SimpleNamespace(log_dir=str(tmp_path)),
        cfg=SimpleNamespace(trainer=SimpleNamespace(logging={"log_images_num_samples": 3, "log_images_top_k": 3})),
        global_step=7,
        loggers=[wandb_like],
        model=model,
        image_samples={"val": dict(step=7, windows=0, images={"route": [], "plan": []})},
    )
    lit.flush_images = lambda stage: LitModel.flush_images(lit, stage)
    lit.plan_panels = lambda *args: LitModel.plan_panels(lit, *args)
    LitModel.on_fit_start(lit)
    batch["past_poses"] = torch.zeros(2, 4, 3)
    batch["camera"] = torch.tensor([30.0, 30.0, 32.0, 16.0, 0, 0, 0, 0, 0.5, 0]).expand(2, 10)
    LitModel.record_images(lit, batch, "val", inputs(batch))  # 2 of 3 windows: still collecting
    assert not logged and not (tmp_path / "images").exists()
    routed = {**batch, "route_mask": torch.ones(2, 4, dtype=torch.bool)}
    LitModel.record_images(lit, routed, "val", inputs(routed))  # 1 more: done
    entries = {entry["key"]: entry for entry in logged}
    assert set(entries) == {"val/route", "val/plan"} and all(e["step"] == 7 for e in logged)
    assert len(entries["val/route"]["images"]) == 2  # window 1 of the first batch + window 0 of the second
    assert torch.equal(torch.from_numpy(entries["val/route"]["images"][0]).permute(2, 0, 1), image)
    plans = entries["val/plan"]["images"]
    assert len(plans) == 3 and all(p.dtype == np.uint8 and p.ndim == 3 and p.shape[-1] == 3 for p in plans)
    assert model.training  # the eval forward restores the mode
    assert not (tmp_path / "images").exists()  # wandb takes them: nothing on disk
    assert lit.image_samples == {}
    LitModel.record_images(lit, batch, "val", inputs(batch))  # no collection running: nothing
    assert len(logged) == 2

    lit.loggers = []  # no image logger: PNGs on disk
    lit.image_samples = {"train": dict(step=9, windows=1, images={"route": [image], "plan": []})}
    LitModel.flush_images(lit, "train")
    assert torch.equal(decode_png(read_file(str(tmp_path / "images" / "train_step0000009" / "route_000.png"))), image)


def test_top_modes_rank_the_anchors_and_keep_the_best_decode():
    model, batch = make_model().eval(), make_batch()
    out = model(*inputs(batch)[:2], **inputs(batch)[2])
    windows, modes, probs, ego_vw = model.current_modes(out.plan, 4, k=3)
    assert windows.tolist() == [0, 1] and modes.shape == (2, 3, 8, 5) and probs.shape == (2, 3)
    assert torch.all(probs[:, :-1] >= probs[:, 1:]) and torch.all(probs.sum(1) <= 1 + 1e-5)
    rows = (out.plan.idx % 4 == 3).nonzero().squeeze(1)
    zeros = torch.zeros(2, *model.action_decoder.anchors.shape)
    best, _ = model.action_decoder.sample(out.plan.tokens[rows], out.plan.bounds[rows], out.plan.ego_vw[rows], zeros)
    torch.testing.assert_close(modes[:, 0], best)  # rank 1 = the noise-0 plan the metrics read


def test_dune_frame_encoder_is_frozen_and_pools_single_frames():
    """flowpilot_dune_dst: one frame per slot through the frozen DUNE, 16 x 28 cells pooled x4, no whole pair."""
    import os

    import pytest

    if not os.path.isdir(os.path.join(torch.hub.get_dir(), "naver_dune_main")):
        pytest.skip("naver/dune is not in the torch.hub cache")
    from visnavkit.models.dune_encoder import DuneEncoder

    encoder = DuneEncoder(downscale=4).train()
    frames, mask = torch.rand(1, 2, 3, 216, 384), torch.tensor([[True, False]])
    glob, patches, speed, pair = encoder(frames, mask)
    assert patches.shape == (1, 2, 768, 4, 7) and glob.shape == (1, 2, 768)
    assert not encoder.encoder.training and not any(p.requires_grad for p in encoder.encoder.parameters())
    assert (glob[0, 1] == 0).all() and not pair.any() and (speed == 0).all()
    glob.sum().backward()
    assert encoder.adapt.weight.grad is not None


def test_window_export_matches_onnx_runtime_and_the_check_replays_it(tmp_path):
    """The traced window graph: fp32 parity, the .pth / metadata / inputs sidecars, and the check on all of them."""
    from visnavkit.export.check import check_export
    from visnavkit.export.dst import export_dst

    torch.set_num_threads(1)
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="train", overrides=[*SMALL, "common.crop_wh=[64,32]", "common.downscale_factor=1"])
    disable_pretrained_downloads(cfg.model)
    path = tmp_path / "flowpilot_dst.onnx"
    meta = export_dst(cfg, path, top_k=3, precision="fp32")
    assert meta["output_shapes"] == {"modes": [1, 3, 8, 5], "probs": [1, 3], "speed": [1, 1]}
    assert meta["input_shapes"]["route_patch"] == [1, 4, 80, 80] and meta["top_k"] == 3
    report = check_export(pth=str(path.with_suffix(".pth")), onnx_path=str(path), iterations=0)
    assert report["ok"] and report["artifacts"]["onnx"]["outputs"]["modes"]["pass"], report
