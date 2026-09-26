"""FlowMatchingPolicy: a basic flow-matching policy over 20 Hz pose windows (``model=flow_matching_policy``).

Per slot with a frame: an RGB timm backbone (``FrameEncoder``) -> its global average-pooled feature, and the frozen
route VAE latent; each Linear -> LayerNorm to D, concatenated -> ``TemporalFusion`` (causal self-attention over the
slots, random past-slot drop ``temporal.mask_p`` in training). The temporal feature (Linear + slot embedding) is the one
kv token of ``StepFlowHead`` (flow matching from N(0, I), one query per plan step + step position + ego [v, w]
embedding). Every slot that holds a frame is decoded and supervised; ``deploy`` decides for the last slot. No goal
input, no speed head. Eval: ``plan.plans`` is the zero-noise plan (1 per slot) and ``plan.variants["randn"]`` the
``num_samples`` N(0, I) samples; LitModel logs metrics for both.

Batch keys: ``vision`` (B, T, 3, H, W) in [0, 1] + ``frame_mask`` (B, T), ``route_patch`` (B, T, h, w) class ids +
``route_mask``, ``ego`` (B, T, >= 2) starting with [v, w], ``action_bounds`` (B, 2, 5); targets ``future_poses``
(B T, P, 5) [x, y, yaw, v, w] per slot in its own ego frame.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from visnavkit.models.layers.masking import masked_rows
from visnavkit.models.modality.route_vae import RouteEncoder
from visnavkit.models.outputs import BaseOutput, PlanOutput
from visnavkit.models.temporal.fusion import TemporalFusion


@dataclass
class FlowPlan(PlanOutput):
    """``plans`` for every slot of the batch (zeros where no frame); the rest describe the decoded rows."""

    valid: torch.Tensor | None = None  # (B T,) slots with a frame: the only rows decoded and supervised
    idx: torch.Tensor | None = None  # their flat indices
    ego_vw: torch.Tensor | None = None  # (N, 2)
    bounds: torch.Tensor | None = None  # (N, 2, 5)
    variants: dict | None = None  # eval: {name: flat plans (B T, ...)} scored like ``plans``, e.g. randn samples


@dataclass
class FlowOutput(BaseOutput):
    plan: FlowPlan


class FlowMatchingPolicy(nn.Module):
    """The policy; ``forward`` takes the batch keys by name (``modality_input_names`` beyond ``vision`` and ``goal``)."""

    modality_input_names = ["frame_mask", "route_patch", "route_mask", "ego", "action_bounds"]

    def __init__(self, frame_encoder, action_decoder, dim=256, seq_len=20, temporal=None, route=None):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.route_encoder = RouteEncoder(**dict(route or {}))
        self.global_in = nn.Sequential(nn.Linear(frame_encoder.dim, dim), nn.LayerNorm(dim))
        self.route_in = nn.Sequential(nn.Linear(self.route_encoder.dim, dim), nn.LayerNorm(dim))
        self.temporal_encoder = TemporalFusion(2 * dim, dim, seq_len, **dict(temporal or {}))
        self.kv = nn.Linear(dim, dim)
        self.slot = nn.Parameter(torch.randn(seq_len, dim) * 0.02)
        self.action_decoder = action_decoder(dim)  # a partial taking dim, e.g. StepFlowHead

    def encode(self, vision, frame_mask=None, route_patch=None, route_mask=None, ego=None, action_bounds=None):
        """-> kv ``(N, 1, D)`` of the N slots holding a frame, their ego [v, w] ``(N, 2)``, bounds ``(N, 2, 5)``, flat
        indices ``(N,)`` and the frame mask."""
        b, t = vision.shape[:2]
        device = vision.device
        if ego is None or action_bounds is None:
            raise ValueError("FlowMatchingPolicy needs `ego` (B, T, >= 2) [v, w, ...] and `action_bounds` (B, 2, 5)")
        if frame_mask is None:
            frame_mask = torch.ones(b, t, dtype=torch.bool, device=device)
        if route_mask is None:
            route_patch, route_mask = vision.new_zeros(b, t, 1, 1), torch.zeros(b, t, dtype=torch.bool, device=device)
        glob = self.frame_encoder(vision, frame_mask)[0]
        route = self.route_encoder(route_patch, route_mask).to(glob.dtype)
        temporal = self.temporal_encoder(torch.cat([self.global_in(glob), self.route_in(route)], -1), frame_mask)
        idx = masked_rows(frame_mask)
        slot = torch.arange(t, device=device).repeat(b)[idx]
        kv = (self.kv(temporal.flatten(0, 1)[idx]) + self.slot[slot])[:, None]
        ego_vw = ego.flatten(0, 1)[idx][:, :2].float()
        bounds = action_bounds.repeat_interleave(t, 0)[idx].float()
        return kv, ego_vw, bounds, idx, frame_mask

    def forward(
        self, vision, goal=None, frame_mask=None, route_patch=None, route_mask=None, ego=None, action_bounds=None
    ) -> FlowOutput:
        b, t = vision.shape[:2]
        kv, ego_vw, bounds, idx, frame_mask = self.encode(
            vision, frame_mask, route_patch, route_mask, ego, action_bounds
        )
        head, plans, variants = self.action_decoder, None, None
        if not self.training:  # zero noise -> 1 plan; randn -> num_samples plans
            zero = head.pack(head.sample(kv, bounds, ego_vw))
            randn = head.pack(head.sample(kv, bounds, ego_vw, head.randn(len(kv), kv.device)))
            plans, randn_plans = (vision.new_zeros(b * t, p.shape[1]) for p in (zero, randn))
            plans[idx], randn_plans[idx] = zero.to(plans.dtype), randn.to(plans.dtype)
            variants = {"randn": randn_plans}
        return FlowOutput(
            plan=FlowPlan(
                plans=plans,
                variants=variants,
                tokens=kv,
                valid=frame_mask.reshape(-1),
                idx=idx,
                ego_vw=ego_vw,
                bounds=bounds,
            )
        )

    @torch.no_grad()
    def current_modes(self, plan: FlowPlan, seq_len, k=1):
        """Each window's current (last) slot from an eval forward's ``plan``: window ids ``(M,)``, the zero-noise plan
        ``(M, 1, T, 5)`` (``k`` unused), probability ``(M, 1)``, ego [v, w]."""
        rows = (plan.idx % seq_len == seq_len - 1).nonzero().squeeze(1)
        modes = self.action_decoder.sample(plan.tokens[rows], plan.bounds[rows], plan.ego_vw[rows])
        return plan.idx[rows] // seq_len, modes, modes.new_ones(len(rows), 1), plan.ego_vw[rows]

    @torch.no_grad()
    def deploy(self, vision, goal, route_patch, ego, action_bounds, k=1, noise=None):
        """The export path (``visnavkit-export-dst``): one decision per window from its last slot, every slot holding a
        frame and a route. ``noise`` None: zero noise, 1 plan; ``(B, S, T, 5)``: S plans from it. -> metric modes
        ``(B, S, T, 5)``, probabilities ``(B, S)`` (1 / S), speed ``(B, 1)`` (zeros: no speed head). ``k`` and
        ``goal`` are unused (the exporter prunes goal)."""
        b, t = vision.shape[:2]
        full = torch.ones(b, t, dtype=torch.bool, device=vision.device)
        kv, ego_vw, bounds, _, _ = self.encode(vision, full, route_patch, full, ego, action_bounds)
        rows = torch.arange(b, device=vision.device) * t + (t - 1)
        modes = self.action_decoder.sample(kv[rows], bounds[rows], ego_vw[rows], noise)
        return modes, torch.full(modes.shape[:2], 1.0 / modes.shape[1], device=vision.device), vision.new_zeros(b, 1)

    def example_batch(self, batch_size, frames, image_hw, device=None):
        """Synthetic ``(vision, goal, {modality key: value})`` inputs for shape checks; every slot holds a frame."""
        b, t = batch_size, frames
        bounds = torch.tensor([[-0.1, -0.1, -0.1, -0.1, -1.0], [0.3, 0.1, 0.1, 3.0, 1.0]], device=device)
        modalities = dict(
            frame_mask=torch.ones(b, t, dtype=torch.bool, device=device),
            route_patch=torch.zeros(b, t, *self.route_encoder.hw, device=device),
            route_mask=torch.zeros(b, t, dtype=torch.bool, device=device),
            ego=torch.rand(b, t, 2, device=device),
            action_bounds=bounds.expand(b, 2, 5),
        )
        return torch.rand(b, t, 3, *image_hw, device=device), None, modalities

    def get_losses(self, preds: FlowOutput, targets):
        plan = preds.plan
        actions = targets["action"]["future_poses"][plan.idx].float()  # (N, T, 5) per slot with a frame
        if actions.shape[-1] != 5:
            raise ValueError("FlowMatchingPolicy needs pose_size 5 targets [x, y, yaw, v, w]")
        loss = self.action_decoder.loss(plan.tokens, actions, plan.bounds, plan.ego_vw)
        return dict(loss=loss, action_reg=loss.detach()), {}
