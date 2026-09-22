"""FlowPilot-DST sanity check and per-stage latency for one decision (batch 1, the full window).

    uv run python -m visnavkit.scripts.profile_dst checkpoint=logs/visnavkit/<run>/checkpoints/last.ckpt
    uv run python -m visnavkit.scripts.profile_dst experiment=flowpilot_dst_clips1k device=cpu iters=5

Checks: a train forward / backward with finite losses and gradients, ``deploy`` shapes, finite and ranked
outputs, determinism, ``deploy`` == ``forward``'s current frame, and how far each Euler step count's top plan
lands from the recipe's. Times: each ``encode`` stage, one Euler step (``denoise`` over all K anchors), the
decode per step count and the whole ``deploy``. Inputs are synthetic (``export_dst.example_inputs``).
"""

import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from visnavkit.benchmark.export import load_native_model
from visnavkit.scripts.export import reparameterize_model
from visnavkit.scripts.export_dst import example_inputs
from visnavkit.utils.display import source


def check(name, ok):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return bool(ok)


def timed(fn, device, warmup, iters):
    """Mean ms per call."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return f"{trainable / 1e6:.2f}/{total / 1e6:.2f}M"


def print_modules(model):
    """Each stage (a direct child of the model, the head one level deeper): its class and trainable / total params."""
    head = model.action_decoder
    named = [(model, name, m) for name, m in model.named_children()]
    named += [(head, name, m) for name, m in head.named_children()]
    named += [(model, name, p) for name, p in model.named_parameters(recurse=False)]
    rows = []
    for owner, attr, m in named:
        name = f"action_decoder.{attr}" if owner is head else attr
        if isinstance(m, torch.nn.Parameter):
            rows.append(
                (
                    name,
                    "Parameter",
                    f"{m.numel() * m.requires_grad / 1e6:.2f}/{m.numel() / 1e6:.2f}M",
                    source(owner, attr, m),
                )
            )
        else:
            rows.append((name, type(m).__name__, params(m), source(owner, attr, m)))
    wn, wc = (max(len(r[i]) for r in rows) for i in (0, 1))
    print(f"\n{'module':<{wn}}  {'class':<{wc}}  {'params (train/total)':>20}  source")
    for name, cls, size, src in rows:
        print(f"{name:<{wn}}  {cls:<{wc}}  {size:>20}  {src}")
    print()


def train_check(model, t, hw, device):
    """One training forward / backward on random targets: finite losses, every trainable parameter gets a gradient."""
    model.train()
    vision, goal, mods = model.example_batch(2, t, hw, device)
    mods["route_mask"] = torch.ones_like(mods["route_mask"])
    out = model(vision, goal, **mods)
    num_pts = model.action_decoder.num_pts
    targets = {
        "action": {"future_poses": torch.randn(2 * t, num_pts, 5, device=device) * 0.1},
        "vision": {"frame_speeds": torch.rand(2, t, device=device)},
    }
    losses, _ = model.get_losses(out, targets)
    losses["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    ok = check(
        f"train losses finite: {', '.join(f'{k} {v.item():.3f}' for k, v in losses.items())}",
        all(torch.isfinite(v).all() for v in losses.values()),
    )
    ok &= check(
        f"every trainable parameter has a gradient ({len(missing)} missing{': ' + missing[0] if missing else ''})",
        not missing,
    )
    model.zero_grad(set_to_none=True)
    return ok


@torch.no_grad()
def deploy_check(model, inputs, k=6):
    vision, route, goal, ego, bounds = inputs
    b, t = vision.shape[:2]
    modes, prob, speed = model.deploy(vision, goal, route, ego, bounds, k)
    again = model.deploy(vision, goal, route, ego, bounds, k)
    full = torch.ones(b, t, dtype=torch.bool, device=vision.device)
    out = model(vision, goal, full, route, full, ego, None, bounds)
    _, fwd_modes, fwd_prob, _ = model.current_modes(out.plan, t, k)
    num_pts = model.action_decoder.num_pts
    ok = check(
        f"deploy shapes modes {tuple(modes.shape)} probs {tuple(prob.shape)} speed {tuple(speed.shape)}",
        modes.shape == (b, k, num_pts, 5) and prob.shape == (b, k) and speed.shape == (b, 1),
    )
    ok &= check("outputs finite", all(torch.isfinite(x).all() for x in (modes, prob, speed)))
    ok &= check(
        f"probs ranked, sum {prob.sum().item():.3f} <= 1",
        bool((prob[:, :-1] >= prob[:, 1:]).all()) and prob.sum(1).max().item() <= 1 + 1e-5,
    )
    ok &= check("deterministic (noise 0)", all(torch.equal(x, y) for x, y in zip((modes, prob, speed), again)))
    diff = (modes - fwd_modes).abs().max().item()
    ok &= check(f"deploy == forward's current frame (max |diff| {diff:.1e})", diff < 1e-4)
    x, y = modes[0, 0, -1, :2].tolist()
    print(f"  top plan: p {prob[0, 0].item():.3f}, v0 {modes[0, 0, 0, 3].item():.2f} m/s, end ({x:.2f}, {y:.2f}) m")
    return ok


@torch.no_grad()
def profile(model, inputs, device, warmup, iters, euler_steps):
    vision, route, goal, ego, bounds = inputs
    b, t = vision.shape[:2]
    full, one = (torch.ones(b, n, dtype=torch.bool, device=device) for n in (t, 1))
    head = model.action_decoder
    glob, _, _, _ = model.pair_encoder(vision, full)
    route_feat = model.route_encoder(route, full)
    fused = model.temporal_inputs(glob, route_feat)
    kv, ego_vw, rows_bounds, *_ = model.encode(vision, goal, full, route, full, ego, None, bounds)
    kv, ego_vw, rows_bounds = kv[-1:], ego_vw[-1:], rows_bounds[-1:]  # the current frame
    zeros = torch.zeros(1, *head.anchors.shape, device=device)
    tt = torch.zeros(1, device=device)
    cond, kv_n = head.time_embed(tt).to(kv.dtype), head.kv_norm(kv)
    ego_emb = head.ego(head.ego_cond(ego_vw, rows_bounds)).to(kv.dtype)

    def run(fn):
        return timed(fn, device, warmup, iters)

    rows = [
        ("frame encoder, 1 frame", params(model.pair_encoder), run(lambda: model.pair_encoder(vision[:, -1:], one))),
        (f"frame encoder, {t} frames", "", run(lambda: model.pair_encoder(vision, full))),
        (f"route encoder, {t} patches", params(model.route_encoder), run(lambda: model.route_encoder(route, full))),
        (
            f"temporal fusion, {t} slots",
            params(model.temporal_encoder),
            run(lambda: model.temporal_encoder(fused, full)),
        ),
        (
            f"encode (all of the above + tokens), {t} frames",
            "",
            run(lambda: model.encode(vision, goal, full, route, full, ego, None, bounds)),
        ),
        (
            f"flow: 1 Euler step, {head.anchors.shape[0]} anchors x {head.num_pts} pts",
            params(head),
            run(lambda: head.denoise(zeros, tt, cond, kv_n, ego_emb)),
        ),
    ]
    rows += [
        (f"flow: decode, {s} Euler steps", "", run(lambda s=s: head.decode(kv, rows_bounds, ego_vw, zeros, s)))
        for s in euler_steps
    ]
    rows.append(
        (
            f"deploy (encode + top_modes, {head.sample_steps} steps)",
            params(model),
            run(lambda: model.deploy(vision, goal, route, ego, bounds)),
        )
    )
    width = max(len(r[0]) for r in rows)
    print(f"\n{'stage':<{width}}  {'params (train/total)':>20}  {'ms':>9}")
    for name, size, ms in rows:
        print(f"{name:<{width}}  {size:>20}  {ms:>9.2f}")

    ref = head.sample(kv, rows_bounds, ego_vw, zeros, head.sample_steps)[0]
    print(f"\ntop plan vs {head.sample_steps} steps (noise 0): ADE / FDE m")
    for s in euler_steps:
        plan = head.sample(kv, rows_bounds, ego_vw, zeros, s)[0]
        err = (plan[..., :2] - ref[..., :2]).norm(dim=-1)
        print(f"  {s:>2} steps: {err.mean().item():.3f} / {err[:, -1].mean().item():.3f}")


@hydra.main(version_base=None, config_path="../configs", config_name="profile_dst")
def main(cfg: DictConfig):
    device = cfg.device
    run_cfg = cfg
    if cfg.checkpoint:
        saved = (
            torch.load(cfg.checkpoint, map_location="cpu", weights_only=False).get("hyper_parameters", {}).get("cfg")
        )
        if saved is not None:
            run_cfg = OmegaConf.create(saved) if isinstance(saved, dict) else saved
    model = load_native_model(run_cfg, cfg.checkpoint).to(device)
    inputs = [x.to(device) for x in example_inputs(run_cfg, 1)]
    t, hw = inputs[0].shape[1], tuple(inputs[0].shape[-2:])
    print(
        f"FlowPilot-DST {cfg.checkpoint or 'untrained'} on {device}: window {t} x {hw}, "
        f"{params(model)} params, dim {run_cfg.model.dim}"
    )
    print_modules(model)
    ok = train_check(model, t, hw, device)
    model = reparameterize_model(model.eval())  # the deployed graph: FastViT branches folded
    ok &= deploy_check(model, inputs)
    profile(model, inputs, device, cfg.warmup, cfg.iters, list(cfg.euler_steps))
    print(f"\nsanity: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
