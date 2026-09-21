"""FlowPilot-DST (decoupled spatial-temporal): frame pairs, route and goal per 20 Hz slot -> per-frame kv tokens -> anchored flow DiT.

Trains on ``dataset=pose`` windows with frames: ``vision`` (B, T, 3, H, W) in [0, 1] + ``frame_mask`` (B, T)
(a slot without a source frame is zeros and False), ``route_patch`` (B, T, h, w) class ids + ``route_mask``,
``goal`` (B, T, 3) [distance m, cos, sin] per slot, ``ego`` (B, T, >= 2) starting with [v, w],
``embodiment_id`` (B,) and ``action_bounds`` (B, 2, 5), [lo; hi] of the per-step [dx, dy, dyaw, v, w].

1. ``PairEncoder``: [frame_t | frame_t-1] (6 channels; the previous slot's frame, zeros when that slot holds
   none or it is dropped w.p. ``p_drop_prev`` in training) through the timm backbone, only the slots with a
   frame as one flat batch -> global (GAP) and patch features; ``SpeedHead`` on the global feature,
   supervised where the pair is whole (``pair_mask``). A corpus below the slot rate never fills two slots
   in a row: its previous frame is always black and it never supervises the speed head. ``frame_encoder`` swaps
   the whole stage for a module with the same interface (``flowpilot_dune_dst``: the frozen single-frame DuneEncoder).
2. ``RouteEncoder``: the frozen route VAE encoder on the slots with a route, zeros elsewhere.
3. ``TemporalFusion``: [global | route] -> D + slot embedding -> causal self-attention over the slots
   (random past drop ``mask_p`` in training; a slot without a frame is no one's key), zero where no frame.
4. ``FeatureTokens``, per frame with a frame: [global, gh x gw patches (+ 2-D sin-cos), route, goal, temporal],
   each Linear -> D + type + slot embedding, + one embodiment token (``embodiment_token``, off for one corpus). The goal token is an MLP of
   [distance / 100, cos, sin]; w.p. ``goal_mask_p`` per window in training, and whenever no goal is given,
   it is the learned empty-goal token.
5. ``AnchorFlowHead`` (the reference AnchorFlowPlanner): K anchors of normalised per-step dx, dy; queries
   x_t^k = (1 - t) eps + t anchor_k -> mode tokens (+ the anchor's embedding + the ego [v, w] embedding);
   ``num_layers`` x [adaLN(t) cross-attention to the kv tokens -> FF], no self-attention between modes;
   per mode a velocity field (T, 2), the direct state (T, 3) [dyaw, v, w] and a score. Loss on the anchor nearest the GT (metric ADE): MSE(velocity,
   x1_dxdy - eps) + MSE(state, x1_dyaw_v_w) + CE(score, winner). Inference: ``sample_steps`` Euler steps
   from noise 0 and from one N(0, I) draw, the top-score mode of each = 2 modes of metric [x, y, yaw, v, w]
   in the flat plan layout, so the open-loop metrics read them unchanged.
"""

import math
from dataclasses import dataclass

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.action.outputs import parse_plan_output
from visnavkit.models.layers.embeddings import SinusoidalTimeEmbedding, timestep_embedding
from visnavkit.models.layers.mlp import build_mlp
from visnavkit.models.outputs import BaseOutput, PlanOutput
from visnavkit.models.temporal.causal import generate_causal_mask
from visnavkit.models.vision.speed_head import SpeedHead
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def masked_rows(mask):
    """``(B, T)`` bool -> flat indices of the True entries; a static ``arange`` while tracing a full mask,
    so the export graph keeps fixed shapes."""
    flat = mask.reshape(-1)
    if torch.jit.is_tracing() and bool(flat.all()):
        return torch.arange(flat.numel(), device=mask.device)
    return flat.nonzero().squeeze(1)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def sincos_2d(h, w, dim, device, dtype):
    """Fixed 2-D sin-cos positional embedding ``(h w, dim)``, ``dim / 4`` frequencies per axis."""
    quarter = dim // 4
    f32 = dict(device=device, dtype=torch.float32)  # float32 throughout: the tracer promotes int arange to float64
    freq = torch.exp(-math.log(10000.0) * torch.arange(quarter, **f32) / quarter)
    yy = (torch.arange(h, **f32)[:, None, None] * freq).expand(h, w, quarter)
    xx = (torch.arange(w, **f32)[None, :, None] * freq).expand(h, w, quarter)
    return torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], -1).reshape(h * w, dim).to(dtype)


class PairEncoder(nn.Module):
    """[frame_t | frame_t-1] through a 6-channel timm backbone on the slots that hold a frame."""

    def __init__(self, backbone_name="fastvit_t12", pretrained=True, p_drop_prev=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, in_chans=6, num_classes=0, features_only=True, out_indices=(-1,)
        )
        self.dim = self.backbone.feature_info.channels()[-1]
        self.stride = self.backbone.feature_info.reduction()[-1]
        if pretrained:
            self.check_pretrained(backbone_name)
        self.speed_head = SpeedHead(self.dim)
        self.p_drop_prev = p_drop_prev
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def check_pretrained(self, backbone_name):
        """Log the load against timm's RGB checkpoint: its first convs tiled to 6 channels x 1/2, every other tensor identical."""
        rgb = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=(-1,)).state_dict()
        own, tiled, same = self.backbone.state_dict(), [], []
        for key, value in rgb.items():
            if own[key].shape == value.shape:
                same += [key] if torch.equal(own[key], value) else []
            elif value.ndim == 4 and value.shape[1] == 3:
                c = own[key].shape[1]
                tiled += [key] if torch.allclose(own[key], value.repeat(1, -(-c // 3), 1, 1)[:, :c] * (3 / c)) else []
        report = (
            f"PairEncoder: {backbone_name} ImageNet weights ({self.backbone.pretrained_cfg.get('hf_hub_id')}): "
            f"{len(same)} / {len(rgb)} tensors identical to the RGB checkpoint, {len(tiled)} first convs {tiled} "
            f"tiled to {own[tiled[0]].shape[1] if tiled else '?'} channels; features {self.dim}-d at stride {self.stride}"
        )
        if len(same) + len(tiled) != len(rgb):
            logger.warning(f"{report}; {sorted(set(rgb) - set(same) - set(tiled))} differ: NOT the pretrained weights")
        else:
            logger.info(report)

    def forward(self, frames, frame_mask):
        """``(B, T, 3, H, W)`` in [0, 1], ``(B, T)`` -> global (B, T, C), patches (B, T, C, gh, gw), speed (B, T, 1), pair_mask (B, T)."""
        b, t = frame_mask.shape
        prev = torch.cat([torch.zeros_like(frames[:, :1]), frames[:, :-1]], 1)
        prev_mask = torch.cat([frame_mask.new_zeros(b, 1), frame_mask[:, :-1]], 1)
        if self.training and self.p_drop_prev > 0:
            prev_mask = prev_mask & (torch.rand(b, t, device=frames.device) >= self.p_drop_prev)
        s = self.stride  # ceil with positive operands only: ONNX integer Div truncates toward zero
        gh, gw = (frames.shape[-2] + s - 1) // s, (frames.shape[-1] + s - 1) // s
        glob = frames.new_zeros(b * t, self.dim)
        patches = frames.new_zeros(b * t, self.dim, gh, gw)
        speed = frames.new_zeros(b * t, 1)
        idx = masked_rows(frame_mask)
        if len(idx):
            cur = frames.flatten(0, 1)[idx]
            old = (prev * prev_mask[..., None, None, None].to(prev.dtype)).flatten(0, 1)[idx]
            x = torch.cat([(cur - self.mean) / self.std, (old - self.mean) / self.std], 1)
            p = self.backbone(x)[-1]
            if p.shape[-2:] != (gh, gw):
                raise ValueError(
                    f"{type(self.backbone).__name__} gives a {tuple(p.shape[-2:])} grid, expected {(gh, gw)}"
                )
            patches[idx], glob[idx] = p.to(patches.dtype), p.mean((2, 3)).to(glob.dtype)
            speed[idx] = self.speed_head(glob[idx]).to(speed.dtype)
        return glob.view(b, t, -1), patches.view(b, t, self.dim, gh, gw), speed.view(b, t, 1), frame_mask & prev_mask


class _ConvResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Identity(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class RouteEncoder(nn.Module):
    """The frozen route VAE encoder (posterior mean) on the slots with a route, zeros elsewhere.

    Layer for layer the checkpoint's ``RouteEncoder`` (a 3 x 3 stem, three stride-2 stages, one residual
    block each, an MLP to ``dim``); patches are class ids 0 .. num_classes - 1 and enter as id / (C - 1).
    """

    def __init__(self, weights=None, num_classes=3, channels=(16, 32, 64), latent=256, dim=256, hw=(80, 80)):
        super().__init__()
        layers, width = [nn.Conv2d(1, channels[0], 3, padding=1, bias=False)], channels[0]
        layers += [nn.BatchNorm2d(width), nn.ReLU(inplace=True), nn.Sequential(_ConvResidualBlock(width))]
        for out in (*channels[1:], latent):
            layers += [
                nn.Conv2d(width, out, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
            ]
            layers.append(nn.Sequential(_ConvResidualBlock(out)))
            width = out
        self.encoder = nn.Sequential(*layers)
        cells = latent * (hw[0] >> len(channels)) * (hw[1] >> len(channels))
        self.encoder_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(cells, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.fc_mu = nn.Linear(dim, dim)
        self.num_classes, self.dim, self.hw = num_classes, dim, tuple(hw)
        if weights is None:
            logger.warning("RouteEncoder: no weights, the frozen route encoder is random (tests only)")
        else:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            prefix = "model.encoder."
            own = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix) and "fc_logvar" not in k}
            self.load_state_dict(own)  # strict: every parameter and BatchNorm statistic comes from the checkpoint
            logger.info(
                f"RouteEncoder: {len(own)} / {len(self.state_dict())} tensors (all, strict) from {weights} "
                f"(epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')}); {len(state) - len(own)} unused "
                f"(fc_logvar, decoder); frozen, posterior mean of {self.num_classes}-class {self.hw[0]} x {self.hw[1]} patches"
            )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):  # frozen: BatchNorm keeps its running statistics
        return super().train(False)

    def forward(self, route_patch, route_mask):
        """``(B, T, h, w)`` class ids, ``(B, T)`` -> ``(B, T, dim)``, zeros where the slot has no route."""
        b, t = route_mask.shape
        out = torch.zeros(b * t, self.dim, device=route_mask.device)
        idx = masked_rows(route_mask)
        if len(idx):
            x = route_patch.flatten(0, 1)[idx][:, None].float() / (self.num_classes - 1)
            with torch.no_grad():
                out[idx] = self.fc_mu(self.encoder_fc(self.encoder(x))).float()
        return out.view(b, t, self.dim)


class TemporalFusion(nn.Module):
    """Causal self-attention over the slots of [global | route]; a slot without a frame is no one's key."""

    reduction = "none"  # LitModel: one target per frame

    def __init__(self, in_dim, dim, seq_len, num_layers=2, num_heads=8, dropout=0.1, mask_p=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim))
        self.pos = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, num_heads, 4 * dim, dropout, "gelu", batch_first=True, norm_first=True)
            for _ in range(num_layers)
        )
        self.num_heads, self.mask_p = num_heads, mask_p

    def forward(self, feats, valid):
        """``(B, T, in_dim)``, ``(B, T)`` -> ``(B, T, dim)``."""
        b, t, _ = feats.shape
        x = self.proj(feats) + self.pos.weight[:t]
        mask = generate_causal_mask(t, self.mask_p if self.training else 0.0, x.device)
        eye = torch.eye(t, device=x.device) > 0  # float EyeLike + Greater: ONNX Runtime has no bool EyeLike
        drop = ~valid[:, None, :] & ~eye
        mask = mask[None].expand(b, t, t).masked_fill(drop, float("-inf")).repeat_interleave(self.num_heads, 0)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return x * valid[..., None]


class FeatureTokens(nn.Module):
    """Per frame: every feature Linear -> D + type + slot embedding (patches one token per cell), + the embodiment token
    (none when ``num_embodiments`` is None)."""

    def __init__(self, dims, dim, seq_len, num_embodiments):
        super().__init__()
        self.proj = nn.ModuleDict({name: nn.Linear(width, dim) for name, width in dims.items()})
        self.type_emb = nn.Parameter(torch.randn(len(dims), dim) * 0.02)
        self.slot = nn.Parameter(torch.randn(seq_len, dim) * 0.02)
        self.embodiment = (
            None if num_embodiments is None else nn.Embedding(num_embodiments + 1, dim)
        )  # last row = unknown

    def forward(self, feats, slot, embodiment_id):
        """``{name: (N, F) | (N, C, gh, gw)}``, slot ``(N,)``, ``(N,)`` -> ``(N, L, D)``."""
        tokens = []
        for i, name in enumerate(self.proj):
            f = feats[name]
            if f.dim() == 4:
                token = self.proj[name](f.flatten(2).transpose(1, 2))
                token = token + sincos_2d(*f.shape[-2:], token.shape[-1], token.device, token.dtype)
            else:
                token = self.proj[name](f)[:, None]
            tokens.append(token + self.type_emb[i] + self.slot[slot][:, None])
        if self.embodiment is not None:
            tokens.append(self.embodiment(embodiment_id)[:, None])
        return torch.cat(tokens, 1)


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


@dataclass
class DSTPlan(PlanOutput):
    """``plans`` for every frame of the batch (zeros where no frame); the rest describe the decoded rows."""

    valid: torch.Tensor | None = None  # (B T,) frames with a frame: the only rows decoded and supervised
    idx: torch.Tensor | None = None  # their flat indices
    ego_vw: torch.Tensor | None = None  # (N, 2)
    bounds: torch.Tensor | None = None  # (N, 2, 5)


@dataclass
class DSTOutput(BaseOutput):
    plan: DSTPlan
    speed: torch.Tensor  # (B T, 1)
    pair_mask: torch.Tensor  # (B T,) frames whose pair is whole: the speed head's supervision


class FlowPilotDST(nn.Module):
    """The policy; ``forward`` takes the batch keys by name (``modality_input_names`` beyond ``vision`` and ``goal``)."""

    modality_input_names = ["frame_mask", "route_patch", "route_mask", "ego", "embodiment_id", "action_bounds"]

    def __init__(
        self,
        backbone_name="fastvit_t12",
        pretrained=True,
        p_drop_prev=0.3,
        dim=512,
        seq_len=20,
        num_embodiments=14,
        goal_mask_p=0.5,
        speed_weight=1.0,
        embodiment_token=True,
        temporal=None,
        route=None,
        head=None,
        frame_encoder=None,
    ):
        super().__init__()
        self.pair_encoder = (  # frame_encoder: a module with PairEncoder's interface, e.g. DuneEncoder
            PairEncoder(backbone_name, pretrained, p_drop_prev) if frame_encoder is None else frame_encoder
        )
        self.route_encoder = RouteEncoder(**dict(route or {}))
        c, r = self.pair_encoder.dim, self.route_encoder.dim
        self.temporal_encoder = TemporalFusion(c + r, dim, seq_len, **dict(temporal or {}))
        self.goal = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.no_goal = nn.Parameter(torch.randn(dim) * 0.02)
        self.tokens = FeatureTokens(
            {"global": c, "patch": c, "route": r, "goal": dim, "temporal": dim},
            dim,
            seq_len,
            num_embodiments if embodiment_token else None,
        )
        self.action_decoder = AnchorFlowHead(dim, **dict(head or {}))
        self.num_embodiments, self.goal_mask_p, self.speed_weight = num_embodiments, goal_mask_p, speed_weight

    def encode(
        self,
        vision,
        goal=None,
        frame_mask=None,
        route_patch=None,
        route_mask=None,
        ego=None,
        embodiment_id=None,
        action_bounds=None,
    ):
        """The stages 1-4 of the module docstring -> the kv tokens ``(N, L, D)`` of the frames that hold a frame,
        their ego ``(N, 2)``, bounds ``(N, 2, 5)``, flat indices ``idx`` ``(N,)``, speed ``(B, T, 1)``, pair mask."""
        b, t = vision.shape[:2]
        device = vision.device
        if ego is None or action_bounds is None:
            raise ValueError("FlowPilotDST needs `ego` (B, T, >= 2) [v, w, ...] and `action_bounds` (B, 2, 5)")
        if frame_mask is None:
            frame_mask = torch.ones(b, t, dtype=torch.bool, device=device)
        if route_mask is None:
            route_patch, route_mask = vision.new_zeros(b, t, 1, 1), torch.zeros(b, t, dtype=torch.bool, device=device)
        if embodiment_id is None:
            embodiment_id = torch.full((b,), self.num_embodiments, dtype=torch.long, device=device)

        glob, patches, speed, pair_mask = self.pair_encoder(vision, frame_mask)
        route = self.route_encoder(route_patch, route_mask).to(glob.dtype)
        temporal = self.temporal_encoder(torch.cat([glob, route], -1), frame_mask)

        idx = masked_rows(frame_mask)

        def rows(value):
            return value.flatten(0, 1)[idx]

        goal_token = self.no_goal.expand(len(idx), -1)
        if goal is not None:
            encoded = self.goal(rows(goal) * goal.new_tensor([0.01, 1.0, 1.0]))
            hidden = (
                torch.rand(b, device=device) < self.goal_mask_p
                if self.training
                else torch.zeros(b, dtype=torch.bool, device=device)
            )
            goal_token = torch.where(hidden.repeat_interleave(t)[idx][:, None], goal_token.to(encoded.dtype), encoded)
        feats = {
            "global": rows(glob),
            "patch": rows(patches),
            "route": rows(route),
            "goal": goal_token,
            "temporal": rows(temporal),
        }
        kv = self.tokens(feats, torch.arange(t, device=device).repeat(b)[idx], embodiment_id.repeat_interleave(t)[idx])
        ego_vw, bounds = rows(ego)[:, :2].float(), action_bounds.repeat_interleave(t, 0)[idx].float()
        return kv, ego_vw, bounds, idx, speed, pair_mask, frame_mask

    def forward(
        self,
        vision,
        goal=None,
        frame_mask=None,
        route_patch=None,
        route_mask=None,
        ego=None,
        embodiment_id=None,
        action_bounds=None,
    ) -> DSTOutput:
        b, t = vision.shape[:2]
        kv, ego_vw, bounds, idx, speed, pair_mask, frame_mask = self.encode(
            vision, goal, frame_mask, route_patch, route_mask, ego, embodiment_id, action_bounds
        )
        plans = torch.zeros(b * t, self.action_decoder.flat_size, device=vision.device)
        logits = None
        if not self.training and len(idx):
            plans[idx], _, logits = self.action_decoder.plans(kv, bounds, ego_vw)
        plan = DSTPlan(
            plans=plans, logits=logits, tokens=kv, valid=frame_mask.reshape(-1), idx=idx, ego_vw=ego_vw, bounds=bounds
        )
        return DSTOutput(plan=plan, speed=speed.reshape(b * t, 1), pair_mask=pair_mask.reshape(-1))

    @torch.no_grad()
    def current_modes(self, plan: DSTPlan, seq_len, k=6):
        """Each window's current frame (its last slot) from an eval forward's ``plan``: the windows ``(M,)`` whose
        current slot holds a frame, their top-``k`` metric modes ``(M, k, T, 5)``, probabilities ``(M, k)``, ego [v, w]."""
        rows = (plan.idx % seq_len == seq_len - 1).nonzero().squeeze(1)
        modes, prob = self.action_decoder.top_modes(plan.tokens[rows], plan.bounds[rows], plan.ego_vw[rows], k)
        return plan.idx[rows] // seq_len, modes, prob, plan.ego_vw[rows]

    @torch.no_grad()
    def deploy(self, vision, goal, route_patch, ego, action_bounds, k=6):
        """The export path: one decision per window, from its current (last) slot, every slot holding a frame and a
        route. -> metric modes ``(B, k, T, 5)`` [x, y, yaw, v, w], probabilities ``(B, k)`` and the speed ``(B, 1)``.
        No embodiment input: a recipe that keeps the embodiment token gets its unknown-embodiment row."""
        b, t = vision.shape[:2]
        full = torch.ones(b, t, dtype=torch.bool, device=vision.device)
        kv, ego_vw, bounds, _, speed, _, _ = self.encode(
            vision, goal, full, route_patch, full, ego, None, action_bounds
        )
        rows = torch.arange(b, device=vision.device) * t + (t - 1)  # each window's current frame
        modes, prob = self.action_decoder.top_modes(kv[rows], bounds[rows], ego_vw[rows], k)
        return modes, prob, speed[:, -1]

    def example_batch(self, batch_size, frames, image_hw, device=None):
        """Synthetic ``(vision, goal, {modality key: value})`` inputs for shape checks; every slot holds a frame."""
        b, t = batch_size, frames
        vision = torch.rand(b, t, 3, *image_hw, device=device)
        goal = torch.rand(b, t, 3, device=device) * torch.tensor([20.0, 1.0, 1.0], device=device)
        bounds = torch.tensor([[-0.1, -0.1, -0.1, -0.1, -1.0], [0.3, 0.1, 0.1, 3.0, 1.0]], device=device)
        modalities = dict(
            frame_mask=torch.ones(b, t, dtype=torch.bool, device=device),
            route_patch=torch.zeros(b, t, *self.route_encoder.hw, device=device),
            route_mask=torch.zeros(b, t, dtype=torch.bool, device=device),
            ego=torch.rand(b, t, 2, device=device),
            embodiment_id=torch.zeros(b, dtype=torch.long, device=device),
            action_bounds=bounds.expand(b, 2, 5),
        )
        return vision, goal, modalities

    def get_losses(self, preds: DSTOutput, targets):
        plan = preds.plan
        actions = targets["action"]["future_poses"][plan.idx].float()  # (N, T, 5) per frame with a frame
        if actions.shape[-1] != 5:
            raise ValueError("FlowPilotDST needs pose_size 5 targets [x, y, yaw, v, w]")
        head = self.action_decoder.loss(plan.tokens, actions, plan.bounds, plan.ego_vw)
        m = preds.pair_mask
        speed_gt = targets["vision"]["frame_speeds"].reshape(-1, 1).float()
        speed = F.smooth_l1_loss(preds.speed[m].float(), speed_gt[m]) if m.any() else preds.speed.sum() * 0.0
        loss_dict = dict(
            loss=head["total"] + self.speed_weight * speed,
            action_reg=head["reg"].detach(),
            action_cls=head["cls"].detach(),
            action_velocity=head["velocity"].detach(),
            action_state=head["state"].detach(),
            vision_speed=speed.detach(),
        )
        return loss_dict, {}
