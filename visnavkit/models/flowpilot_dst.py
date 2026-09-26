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
3. ``TemporalFusion``: [global | route] (``input_norm``: each Linear -> LayerNorm to D first) -> D + slot embedding -> causal self-attention over the slots
   (random past drop ``mask_p`` in training; a slot without a frame is no one's key), zero where no frame.
4. ``FeatureTokens``, per frame with a frame: [global, gh x gw patches (+ 2-D sin-cos), route, goal, temporal],
   each Linear -> D + type + slot embedding, + one embodiment token (``embodiment_token``, off for one corpus). The goal token is an MLP of
   [distance / 100, cos, sin]; w.p. ``goal_mask_p`` per window in training, and whenever no goal is given,
   it is the learned empty-goal token.
   ``context.num_layers`` self-attention layers then mix one frame's tokens (0: none, older checkpoints).
   Aux (``temporal_reg_weight`` > 0): an MLP regresses one plan from each frame's temporal feature, MSE on the
   normalised per-step state as the head's ``norm``; training only, not in ``deploy``.
5. ``AnchorFlowHead`` (the reference AnchorFlowPlanner): K anchors of normalised per-step dx, dy; queries
   x_t^k = (1 - t) eps + t anchor_k -> mode tokens (+ the anchor's embedding + the ego [v, w] embedding);
   ``num_layers`` x [adaLN(t) cross-attention to the kv tokens -> FF], no self-attention between modes;
   per mode a velocity field (T, 2), the direct state (T, 3) [dyaw, v, w] and a score. Loss on the anchor nearest the GT (metric ADE): MSE(velocity,
   x1_dxdy - eps) + MSE(state, x1_dyaw_v_w) + CE(score, winner). Inference: ``sample_steps`` Euler steps
   from noise 0 and from one N(0, I) draw, the top-score mode of each = 2 modes of metric [x, y, yaw, v, w]
   in the flat plan layout, so the open-loop metrics read them unchanged.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.action.anchor_flow import AnchorFlowHead
from visnavkit.models.layers.embeddings import sincos_2d
from visnavkit.models.layers.masking import masked_rows
from visnavkit.models.layers.mlp import build_mlp
from visnavkit.models.modality.route_vae import RouteEncoder
from visnavkit.models.outputs import BaseOutput, PlanOutput
from visnavkit.models.temporal.fusion import TemporalFusion
from visnavkit.models.vision.pair_encoder import PairEncoder


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
    temporal_plan: torch.Tensor | None = None  # (N, T, 5) the temporal feature's single-mode plan (temporal_reg_weight)


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
        input_norm=False,
        context=None,
        temporal_reg_weight=0.0,
    ):
        super().__init__()
        self.pair_encoder = (  # frame_encoder: a module with PairEncoder's interface, e.g. DuneEncoder
            PairEncoder(backbone_name, pretrained, p_drop_prev) if frame_encoder is None else frame_encoder
        )
        self.route_encoder = RouteEncoder(**dict(route or {}))
        c, r = self.pair_encoder.dim, self.route_encoder.dim
        self.input_norm = input_norm
        if input_norm:  # global and route each Linear -> LayerNorm to dim before the concat: neither sets the scale
            self.global_in = nn.Sequential(nn.Linear(c, dim), nn.LayerNorm(dim))
            self.route_in = nn.Sequential(nn.Linear(r, dim), nn.LayerNorm(dim))
        self.temporal_encoder = TemporalFusion(2 * dim if input_norm else c + r, dim, seq_len, **dict(temporal or {}))
        self.goal = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.no_goal = nn.Parameter(torch.randn(dim) * 0.02)
        self.tokens = FeatureTokens(
            {"global": c, "patch": c, "route": r, "goal": dim, "temporal": dim},
            dim,
            seq_len,
            num_embodiments if embodiment_token else None,
        )
        ctx = {"num_layers": 0, "num_heads": 8, "dropout": 0.1, **dict(context or {})}
        self.context = nn.ModuleList(  # self-attention over one frame's kv tokens before the DiT reads them
            nn.TransformerEncoderLayer(
                dim, ctx["num_heads"], 4 * dim, ctx["dropout"], "gelu", batch_first=True, norm_first=True
            )
            for _ in range(ctx["num_layers"])
        )
        self.action_decoder = AnchorFlowHead(dim, **dict(head or {}))
        self.num_embodiments, self.goal_mask_p, self.speed_weight = num_embodiments, goal_mask_p, speed_weight
        self.temporal_reg_weight = temporal_reg_weight  # aux: one plan regressed from the temporal feature, 0 = off
        num_pts = self.action_decoder.num_pts
        self.temporal_reg = build_mlp(dim, dim, num_pts * 5, layers=1) if temporal_reg_weight else None

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
        their ego ``(N, 2)``, bounds ``(N, 2, 5)``, flat indices ``idx`` ``(N,)``, speed ``(B, T, 1)``, pair mask, frame
        mask and their temporal features ``(N, D)``."""
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
        temporal = self.temporal_encoder(self.temporal_inputs(glob, route), frame_mask)

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
        for layer in self.context:
            kv = layer(kv)
        ego_vw, bounds = rows(ego)[:, :2].float(), action_bounds.repeat_interleave(t, 0)[idx].float()
        return kv, ego_vw, bounds, idx, speed, pair_mask, frame_mask, feats["temporal"]

    def temporal_inputs(self, glob, route):
        """``(B, T, C)``, ``(B, T, R)`` -> the temporal stage's input [global | route]; with ``input_norm`` each is
        projected and normalised first."""
        if self.input_norm:
            glob, route = self.global_in(glob), self.route_in(route)
        return torch.cat([glob, route], -1)

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
        kv, ego_vw, bounds, idx, speed, pair_mask, frame_mask, temporal = self.encode(
            vision, goal, frame_mask, route_patch, route_mask, ego, embodiment_id, action_bounds
        )
        plans = torch.zeros(b * t, self.action_decoder.flat_size, device=vision.device)
        logits = None
        if not self.training and len(idx):
            plans[idx], _, logits = self.action_decoder.plans(kv, bounds, ego_vw)
        plan = DSTPlan(
            plans=plans, logits=logits, tokens=kv, valid=frame_mask.reshape(-1), idx=idx, ego_vw=ego_vw, bounds=bounds
        )
        temporal_plan = None if self.temporal_reg is None else self.temporal_reg(temporal).view(len(idx), -1, 5)
        return DSTOutput(
            plan=plan, speed=speed.reshape(b * t, 1), pair_mask=pair_mask.reshape(-1), temporal_plan=temporal_plan
        )

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
        kv, ego_vw, bounds, _, speed, _, _, _ = self.encode(
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
        if preds.temporal_plan is not None:  # aux: the normalised per-step state [dx, dy, dyaw, v, w], as the head's
            target = self.action_decoder.norm(actions, plan.bounds)
            reg = F.mse_loss(preds.temporal_plan.float(), target, reduction="none").sum(-1).mean()
            loss_dict["loss"] = loss_dict["loss"] + self.temporal_reg_weight * reg
            loss_dict["temporal_reg"] = reg.detach()
        return loss_dict, {}
