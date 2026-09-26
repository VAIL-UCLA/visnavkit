"""FlowPilot-DST head, no anchors: flow matching from N(0, I) with one query token per plan step (``flowpilot_step_dst``).

The state is AnchorFlowHead's: the normalised per-step [dx, dy, dyaw, v, w] under the row's ``action_bounds``; t = 0
noise, 1 data, Beta(1.5, 1) times. Query per step = Linear(x_t step) + learned step position + the ego [v, w] embedding
(each channel hidden w.p. ``ego_mask_p`` in training); ``num_layers`` DiT blocks [adaLN-Zero(flow time) self-attention
over the steps, cross-attention to the frame's kv, FF] -> the velocity per step. Loss MSE(velocity, x1 - eps).
Inference: ``sample_steps`` Euler steps; mode 0 from noise 0 (deterministic), the others from fixed seeded draws, equal
probabilities (no scorer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.action.anchor_flow import AnchorFlowHead
from visnavkit.models.action.denoisers.dit import DiTBlock, modulate
from visnavkit.models.layers.embeddings import SinusoidalTimeEmbedding


class StepFlowHead(nn.Module):
    pose_size, num_modes = 5, 2
    flat_size, parse_output = AnchorFlowHead.flat_size, AnchorFlowHead.parse_output
    norm, metric, ego_cond, sample_time = (
        AnchorFlowHead.norm,
        AnchorFlowHead.metric,
        AnchorFlowHead.ego_cond,
        AnchorFlowHead.sample_time,
    )

    def __init__(
        self, dim, num_pts=80, num_layers=4, num_heads=8, dropout=0.1, ego_mask_p=0.9, sample_steps=4, max_modes=16
    ):
        super().__init__()
        self.num_pts, self.ego_mask_p, self.sample_steps = num_pts, ego_mask_p, sample_steps
        self.inp = nn.Linear(self.pose_size, dim)
        self.pos = nn.Parameter(torch.randn(num_pts, dim) * 0.02)
        self.time_embed = SinusoidalTimeEmbedding(dim)
        self.ego = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.kv_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(DiTBlock(dim, num_heads, dim, dropout=dropout) for _ in range(num_layers))
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.ada_out[-1].weight), nn.init.zeros_(self.ada_out[-1].bias)
        self.out = nn.Linear(dim, self.pose_size)
        noise = torch.randn(max_modes - 1, num_pts, self.pose_size, generator=torch.Generator().manual_seed(0))
        self.register_buffer("mode_noise", noise, persistent=False)  # modes 1.. at inference

    def velocity(self, x, t, kv, ego):
        """``(N, T, 5)``, ``(N,)``, kv ``(N, L, D)``, ego ``(N, D)`` -> ``(N, T, 5)``."""
        c = self.time_embed(t).to(kv.dtype)
        h = self.inp(x.to(kv.dtype)) + self.pos + ego[:, None]
        for block in self.blocks:
            h = block(h, c, kv)
        shift, scale = self.ada_out(c)[:, None].chunk(2, -1)
        return self.out(modulate(self.norm_out(h), shift, scale))

    def loss(self, kv, actions, bounds, ego_vw):
        kv = self.kv_norm(kv)
        x1 = self.norm(actions, bounds)
        noise = torch.randn_like(x1)
        t = self.sample_time(len(kv), kv.device)
        x_t = (1 - t)[:, None, None] * noise + t[:, None, None] * x1
        ego = self.ego(self.ego_cond(ego_vw, bounds)).to(kv.dtype)
        v = F.mse_loss(self.velocity(x_t, t, kv, ego).float(), x1 - noise, reduction="none").sum(-1).mean()
        zero = v.new_zeros(())
        return dict(total=v, reg=v, cls=zero, velocity=v, state=zero)

    @torch.no_grad()
    def decode(self, kv, bounds, ego_vw, noise):
        """Euler from ``noise`` ``(N, T, 5)`` -> metric ``(N, T, 5)`` [x, y, yaw, v, w]."""
        n, kv = len(kv), self.kv_norm(kv)
        ego = self.ego(self.ego_cond(ego_vw, bounds)).to(kv.dtype)
        x, dt = noise, 1.0 / self.sample_steps
        for i in range(self.sample_steps):
            x = x + dt * self.velocity(x, torch.full((n,), i * dt, device=kv.device), kv, ego).float()
        return self.metric(x, bounds)

    @torch.no_grad()
    def top_modes(self, kv, bounds, ego_vw, k=6):
        """``k`` metric modes ``(N, k, T, 5)``, mode 0 from noise 0, and equal probabilities ``(N, k)``."""
        n = len(kv)
        noise = torch.cat([torch.zeros_like(self.mode_noise[:1]), self.mode_noise[: k - 1]]).repeat(n, 1, 1)
        rep = [v.repeat_interleave(k, 0) for v in (kv, bounds, ego_vw)]
        modes = self.decode(*rep, noise).view(n, k, self.num_pts, self.pose_size)
        return modes, torch.full((n, k), 1.0 / k, device=kv.device)

    def plans(self, kv, bounds, ego_vw):
        """``num_modes`` modes -> flat plans ``(N, flat_size)``, mu ``(N, M, T, 5)``, logits ``(N, M)``."""
        mu, _ = self.top_modes(kv, bounds, ego_vw, self.num_modes)
        logits = mu.new_zeros(len(kv), self.num_modes)
        flat = torch.cat([mu.flatten(2), torch.zeros_like(mu.flatten(2)), logits[..., None]], -1).flatten(1)
        return flat, mu, logits
