"""FlowPilot-DST head: anchored flow matching (the reference AnchorFlowPlanner); see ``models/flowpilot_dst.py``, item 5."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.action.outputs import parse_plan_output
from visnavkit.models.layers.embeddings import SinusoidalTimeEmbedding, timestep_embedding
from visnavkit.models.layers.mlp import build_mlp
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class CrossAttention(nn.Module):
    """Mode tokens attend to the frame's kv tokens; per-head RMSNorm on q and k."""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.num_heads, self.dropout = num_heads, dropout
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.q_norm, self.k_norm = nn.RMSNorm(dim // num_heads), nn.RMSNorm(dim // num_heads)

    def forward(self, x, ctx):
        n, m, d = x.shape
        q = self.q_norm(self.q(x).view(n, m, self.num_heads, -1)).transpose(1, 2)
        k, v = self.kv(ctx).view(n, ctx.shape[1], 2, self.num_heads, -1).unbind(2)
        k, v = self.k_norm(k).transpose(1, 2), v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(n, m, d))


class DiTBlock(nn.Module):
    """adaLN(cond)-modulated cross-attention to the kv tokens + FF; no self-attention between the modes."""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = CrossAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, ctx, cond):
        shift, scale = self.ada(cond)[:, None].chunk(2, -1)
        x = x + self.attn(self.norm1(x) * (1 + scale) + shift, ctx)
        return x + self.ff(self.norm2(x))


class AnchorFlowHead(nn.Module):
    """Anchored flow matching over the normalised per-step state; see the module docstring, item 5.

    State: ``2 u - 1`` with ``u = (step - lo) / (hi - lo)`` per row's ``action_bounds``; ``metric`` inverts it
    (cumsum of dx, dy, dyaw; v, w levels). ``anchors`` are the k-means centres of ``u``'s dx, dy.
    """

    pose_size, num_modes = 5, 2

    def __init__(
        self,
        dim,
        num_pts=80,
        anchors_path=None,
        num_anchors=64,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        wp_dim=64,
        ego_mask_p=0.5,
        sample_steps=4,
        start_step=0.0,
        v_weight=1.0,
        state_weight=1.0,
        ce_weight=1.0,
    ):
        super().__init__()
        if anchors_path is None:
            logger.warning("AnchorFlowHead: no anchors_path, the anchors are random (tests only)")
            anchors = torch.rand(num_anchors, num_pts, 2)
        else:
            loaded = np.load(anchors_path)
            anchors = torch.as_tensor(
                np.asarray(loaded["anchors"] if hasattr(loaded, "files") else loaded), dtype=torch.float32
            )
            if anchors.ndim != 3 or anchors.shape[1:] != (num_pts, 2):
                raise ValueError(
                    f"{anchors_path} must hold (K, {num_pts}, 2) normalised dx, dy anchors, got {tuple(anchors.shape)}"
                )
        self.register_buffer("anchors", 2 * anchors - 1)  # (K, T, 2) in the state space
        self.num_pts, self.wp_dim = num_pts, wp_dim
        self.mode_emb = nn.Parameter(torch.randn(len(anchors), dim) * 0.02)
        self.w1, self.w2, self.w3 = nn.Linear(num_pts * wp_dim, dim), nn.Linear(2 * dim, dim), nn.Linear(dim, dim)
        self.time_embed = SinusoidalTimeEmbedding(dim)
        self.ego = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.ego[2].weight), nn.init.zeros_(self.ego[2].bias)  # a silent ego branch at init
        self.kv_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(DiTBlock(dim, num_heads, dropout) for _ in range(num_layers))
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.velocity = build_mlp(dim, dim, num_pts * 2, layers=1)
        self.state = build_mlp(dim, dim, num_pts * 3, layers=1)
        self.score = build_mlp(dim, dim, 1, layers=1)
        self.ego_mask_p, self.sample_steps, self.start_step = ego_mask_p, sample_steps, start_step
        self.weights = (v_weight, state_weight, ce_weight)

    @property
    def flat_size(self):
        return self.num_modes * (2 * self.num_pts * self.pose_size + 1)

    # ---- normalisation ----------------------------------------------------------------------
    def norm(self, actions, bounds):
        """Metric ``(N, T, 5)`` [x, y, yaw, v, w], ``(N, 2, 5)`` -> state ``(N, T, 5)`` ~ [-1, 1]."""
        d = torch.diff(actions[..., :3], dim=1, prepend=torch.zeros_like(actions[:, :1, :3]))
        s = torch.cat([d[..., :2], wrap(d[..., 2:3]), actions[..., 3:]], -1)
        return 2 * (s - bounds[:, None, 0]) / (bounds[:, None, 1] - bounds[:, None, 0]) - 1

    def metric(self, state, bounds):
        """State ``(N, T, 5)`` -> metric ``(N, T, 5)`` [x, y, yaw (unwrapped), v, w]."""
        s = (state + 1) / 2 * (bounds[:, None, 1] - bounds[:, None, 0]) + bounds[:, None, 0]
        return torch.cat([s[..., :3].cumsum(1), s[..., 3:]], -1)

    def ego_cond(self, ego_vw, bounds, avail=None):
        """``(N, 2)`` [v, w] -> ``(N, 4)`` [v_n a_v, w_n a_w, a_v, a_w]; a channel hidden w.p. ego_mask_p in training."""
        lo, hi = bounds[:, 0, 3:], bounds[:, 1, 3:]
        e = 2 * (ego_vw - lo) / (hi - lo) - 1
        if avail is None:
            avail = torch.rand_like(e) >= self.ego_mask_p if self.training else torch.ones_like(e, dtype=torch.bool)
        avail = avail.to(e.dtype)
        return torch.cat([e * avail, avail], -1)

    # ---- denoiser -----------------------------------------------------------------------------
    def mode_tokens(self, x, t):
        """Noisy ``(N, K, T, 2)`` + flow time ``(N,)`` -> ``(N, K, D)``: sine waypoint features, t, the anchor's embedding."""
        quarter = self.wp_dim // 4
        freq = torch.exp(-math.log(10000.0) * torch.arange(quarter, device=x.device, dtype=torch.float32) / quarter)
        ang = x[..., None] * (2 * math.pi) * freq  # (N, K, T, 2, quarter)
        wp = torch.cat([ang[..., 1, :].sin(), ang[..., 1, :].cos(), ang[..., 0, :].sin(), ang[..., 0, :].cos()], -1)
        a = self.w1(wp.flatten(2).to(self.w1.weight.dtype))
        tau = timestep_embedding(t, a.shape[-1]).to(a.dtype)[:, None].expand_as(a)
        return self.w3(F.silu(self.w2(torch.cat([a, tau], -1)))) + self.mode_emb

    def denoise(self, x, t, cond, kv, ego):
        """adaLN on ``cond`` (flow time), the ego embedding ``(N, D)`` added to every mode token -> velocity
        ``(N, K, T, 2)``, state ``(N, K, T, 3)`` [dyaw, v, w], score ``(N, K)``."""
        h = self.mode_tokens(x, t) + ego[:, None]
        for block in self.blocks:
            h = block(h, kv, cond)
        shift, scale = self.ada_out(cond)[:, None].chunk(2, -1)
        h = self.norm_out(h) * (1 + scale) + shift
        n, k = h.shape[:2]
        return (
            self.velocity(h).view(n, k, self.num_pts, 2),
            self.state(h).view(n, k, self.num_pts, 3),
            self.score(h).squeeze(-1),
        )

    def sample_time(self, n, device):
        """Beta(1.5, 1) tilted toward the noisy end (t = 0 is noise, 1 the anchor), as the reference."""
        t = torch.distributions.Beta(1.5, 1.0).sample((n,)).to(device)
        return (0.999 - t) / 0.999

    # ---- loss / sampling ----------------------------------------------------------------------
    def loss(self, kv, actions, bounds, ego_vw):
        n, kv = len(kv), self.kv_norm(kv)
        x1 = self.norm(actions, bounds)
        anchors = self.anchors[None].expand(n, -1, -1, -1)
        noise = torch.randn_like(anchors)
        t = self.sample_time(n, kv.device)
        x_t = (1 - t)[:, None, None, None] * noise + t[:, None, None, None] * anchors
        ego = self.ego(self.ego_cond(ego_vw, bounds)).to(kv.dtype)
        velocity, state, score = self.denoise(x_t, t, self.time_embed(t).to(kv.dtype), kv, ego)
        lo, hi = bounds[:, None, None, 0, :2], bounds[:, None, None, 1, :2]
        paths = ((anchors + 1) / 2 * (hi - lo) + lo).cumsum(2)  # (N, K, T, 2) metric
        winner = (paths - actions[:, None, :, :2]).norm(dim=-1).mean(-1).argmin(1)  # the anchor nearest the GT
        rows = torch.arange(n, device=kv.device)
        v = (
            F.mse_loss(velocity[rows, winner].float(), x1[..., :2] - noise[rows, winner], reduction="none")
            .sum(-1)
            .mean()
        )
        s = F.mse_loss(state[rows, winner].float(), x1[..., 2:], reduction="none").sum(-1).mean()
        ce = F.cross_entropy(score.float(), winner)
        wv, ws, wc = self.weights
        return dict(total=wv * v + ws * s + wc * ce, reg=v + s, cls=ce, velocity=v, state=s)

    @torch.no_grad()
    def decode(self, kv, bounds, ego_vw, noise=None, num_steps=None):
        """Every anchor's state ``(N, K, T, 5)`` and score ``(N, K)``; ``noise`` None = N(0, I)."""
        n, kv = len(kv), self.kv_norm(kv)
        anchors = self.anchors[None].expand(n, -1, -1, -1)
        noise = torch.randn_like(anchors) if noise is None else noise
        s0, steps = self.start_step, num_steps or self.sample_steps
        x, dt = noise * (1 - s0) + anchors * s0, (1 - s0) / steps
        ego = self.ego(self.ego_cond(ego_vw, bounds)).to(kv.dtype)
        for i in range(steps):  # Euler on the predicted velocity
            t = torch.full((n,), s0 + i * dt, device=kv.device)
            velocity, state, score = self.denoise(x, t, self.time_embed(t).to(kv.dtype), kv, ego)
            x = x + dt * velocity.float()
        return torch.cat([x.clamp(-3.0, 3.0), state.float()], -1), score.float()

    @torch.no_grad()
    def sample(self, kv, bounds, ego_vw, noise=None, num_steps=None):
        """-> the top-score mode's metric plan ``(N, T, 5)`` and its score ``(N,)``; ``noise`` None = N(0, I)."""
        state, score = self.decode(kv, bounds, ego_vw, noise, num_steps)
        rows, best = torch.arange(len(kv), device=kv.device), score.argmax(1)
        return self.metric(state[rows, best], bounds), score[rows, best]

    @torch.no_grad()
    def top_modes(self, kv, bounds, ego_vw, k=6):
        """The ``k`` highest-score modes decoded from noise 0: metric ``(N, k, T, 5)``, softmax probabilities ``(N, k)``."""
        zeros = torch.zeros(len(kv), *self.anchors.shape, device=kv.device)
        state, score = self.decode(kv, bounds, ego_vw, noise=zeros)
        prob, top = score.softmax(1).topk(min(k, score.shape[1]), dim=1)
        state = torch.gather(state, 1, top[..., None, None].expand(-1, -1, *state.shape[2:]))
        n, k = top.shape
        metric = self.metric(state.flatten(0, 1), bounds.repeat_interleave(k, 0)).view(n, k, *state.shape[2:])
        return metric, prob

    def plans(self, kv, bounds, ego_vw):
        """Two modes, decoded from noise 0 and from one draw -> flat plans ``(N, flat_size)``, mu ``(N, 2, T, 5)``, logits ``(N, 2)``."""
        zeros = torch.zeros(len(kv), *self.anchors.shape, device=kv.device)
        (m0, s0), (m1, s1) = self.sample(kv, bounds, ego_vw, noise=zeros), self.sample(kv, bounds, ego_vw)
        mu, logits = torch.stack([m0, m1], 1), torch.stack([s0, s1], 1)
        flat = torch.cat([mu.flatten(2), torch.zeros_like(mu.flatten(2)), logits[..., None]], -1).flatten(1)
        return flat, mu, logits

    @torch.no_grad()
    def parse_output(self, plans):
        return parse_plan_output(plans, num_modes=self.num_modes, num_pts=self.num_pts, pose_size=self.pose_size)
