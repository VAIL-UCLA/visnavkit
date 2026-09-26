"""StepFlowHead: flow matching from N(0, I) with one query token per plan step (``flow_matching_policy``).

The state is AnchorFlowHead's: the normalised per-step [dx, dy, dyaw, v, w] under the row's ``action_bounds``; t = 0
noise, 1 data, Beta(1.5, 1) times. Query per step = Linear(x_t step) + learned step position + the ego [v, w] embedding
(each channel hidden w.p. ``ego_mask_p`` in training); ``num_layers`` DiT blocks [adaLN-Zero(flow time) self-attention
over the steps, cross-attention to the frame's kv, FF] -> the velocity per step. Loss MSE(velocity, x1 - eps).
Inference: ``sample_steps`` Euler steps from noise 0 (one deterministic plan) or from N(0, I) draws (``num_samples``
plans, equal probabilities: no scorer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.action.anchor_flow import AnchorFlowHead
from visnavkit.models.action.outputs import parse_plan_output
from visnavkit.models.action.denoisers.dit import DiTBlock, modulate
from visnavkit.models.layers.embeddings import SinusoidalTimeEmbedding


class StepFlowHead(nn.Module):
    pose_size = 5
    norm, metric, ego_cond, sample_time = (
        AnchorFlowHead.norm,
        AnchorFlowHead.metric,
        AnchorFlowHead.ego_cond,
        AnchorFlowHead.sample_time,
    )

    def __init__(
        self, dim, num_pts=80, num_layers=4, num_heads=8, dropout=0.1, ego_mask_p=0.9, sample_steps=4, num_samples=1
    ):
        super().__init__()
        self.num_pts, self.ego_mask_p, self.sample_steps, self.num_samples = (
            num_pts,
            ego_mask_p,
            sample_steps,
            num_samples,
        )
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

    def velocity(self, x, t, kv, ego):
        """``(N, T, 5)``, ``(N,)``, kv ``(N, L, D)``, ego ``(N, D)`` -> ``(N, T, 5)``."""
        c = self.time_embed(t).to(kv.dtype)
        h = self.inp(x.to(kv.dtype)) + self.pos + ego[:, None]
        for block in self.blocks:
            h = block(h, c, kv)
        shift, scale = self.ada_out(c)[:, None].chunk(2, -1)
        return self.out(modulate(self.norm_out(h), shift, scale))

    def loss(self, kv, actions, bounds, ego_vw):
        """MSE(velocity, x1 - eps) on the normalised state, summed over its 5 channels."""
        kv = self.kv_norm(kv)
        x1 = self.norm(actions, bounds)
        noise = torch.randn_like(x1)
        t = self.sample_time(len(kv), kv.device)
        x_t = (1 - t)[:, None, None] * noise + t[:, None, None] * x1
        ego = self.ego(self.ego_cond(ego_vw, bounds)).to(kv.dtype)
        return F.mse_loss(self.velocity(x_t, t, kv, ego).float(), x1 - noise, reduction="none").sum(-1).mean()

    @torch.no_grad()
    def sample(self, kv, bounds, ego_vw, noise=None):
        """Euler from ``noise`` ``(N, S, T, 5)`` (None: zeros, S = 1) -> metric ``(N, S, T, 5)`` [x, y, yaw, v, w]."""
        n = len(kv)
        if noise is None:
            noise = kv.new_zeros(n, 1, self.num_pts, self.pose_size, dtype=torch.float32)
        s = noise.shape[1]
        kv, bounds, ego_vw = (v.repeat_interleave(s, 0) for v in (kv, bounds, ego_vw))
        kv = self.kv_norm(kv)
        ego = self.ego(self.ego_cond(ego_vw, bounds)).to(kv.dtype)
        x, dt = noise.flatten(0, 1).float(), 1.0 / self.sample_steps
        for i in range(self.sample_steps):
            x = x + dt * self.velocity(x, torch.full((n * s,), i * dt, device=kv.device), kv, ego).float()
        return self.metric(x, bounds).view(n, s, self.num_pts, self.pose_size)

    def randn(self, n, device):
        """``num_samples`` N(0, I) draws per row ``(n, S, T, 5)``."""
        return torch.randn(n, self.num_samples, self.num_pts, self.pose_size, device=device)

    def pack(self, modes):
        """Metric ``(N, M, T, 5)`` -> the flat plan layout ``(N, M (2 T 5 + 1))`` (zero log-scales and logits)."""
        logits = modes.new_zeros(*modes.shape[:2], 1)
        return torch.cat([modes.flatten(2), torch.zeros_like(modes.flatten(2)), logits], -1).flatten(1)

    @torch.no_grad()
    def parse_output(self, plans):
        """The flat layout with M inferred from its width (zero noise: 1, randn: ``num_samples``)."""
        m = plans.shape[1] // (2 * self.num_pts * self.pose_size + 1)
        return parse_plan_output(plans, num_modes=m, num_pts=self.num_pts, pose_size=self.pose_size)
