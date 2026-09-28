"""Generative trajectory decoder: denoiser x scheduler x optional anchors.

The matrix of recipes (diffusion/flow x MLP/DiT/U-Net x free/anchored) is configuration only:
``GenerativeDecoder(denoiser=DiTDenoiser, scheduler=FlowMatchingScheduler, anchors=AnchorSet)``.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.action.anchors import AnchorSet
from visnavkit.models.action.base import BaseActionDecoder
from visnavkit.models.action.schedulers.base import BaseScheduler
from visnavkit.models.layers.mlp import build_mlp
from visnavkit.models.outputs import PlanOutput
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["GenerativeDecoder"]


class GenerativeDecoder(BaseActionDecoder):
    """Sample ``num_modes`` trajectories from noise (or one refined trajectory per anchor).

    Without anchors the modes are independent draws with uniform confidence. With anchors the
    denoiser generates the residual to each anchor, so every anchor yields one mode and a
    classifier over anchors provides the confidences (``num_modes = K``).

    Args:
        denoiser: ``partial`` taking ``action_dim``, ``num_pts``, ``cond_dim``; the built module maps
            ``(x_t, t, cond, tokens) -> prediction``.
        scheduler: noise schedule and sampler (DDIM or flow matching).
        sample_steps: denoising steps at inference; export unrolls them.
        noise: ``forward(..., noise=)`` accepts explicit initial noise ``(N, M, T, A)`` for
            deterministic parity checks and ONNX graphs.
    """

    uses_noise = True

    def __init__(
        self,
        action_space,
        denoiser: Callable[..., nn.Module],
        scheduler: BaseScheduler,
        normalizer=None,
        feat_size=256,
        num_modes=5,
        sample_steps=8,
        anchors: AnchorSet | None = None,
        anchor_hidden=256,
        loss_cls_alpha=1.0,
        dropout=0.0,
        **kwargs,
    ):
        if anchors is not None:
            anchors = anchors.build(action_space.t_anchors.tolist(), action_space.pose_size)
            num_modes = anchors.num_anchors
        super().__init__(action_space, normalizer, feat_size=feat_size, num_modes=num_modes, dropout=dropout, **kwargs)
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        self.denoiser = denoiser(action_dim=self.action_dim, num_pts=self.num_pts, cond_dim=feat_size)
        self.scheduler = scheduler
        self.sample_steps = sample_steps
        self.anchors = anchors
        self.loss_cls_alpha = loss_cls_alpha
        if getattr(scheduler, "clip_sample", None) and self.normalizer.identity:
            logger.warning(
                f"{type(scheduler).__name__}.clip_sample={scheduler.clip_sample} clamps predicted actions in "
                "normalizer units, which are raw metres while normalizer.mode=none: plans longer than that are "
                "truncated. Fit an ActionNormalizer, or set the scheduler's clip_sample to null."
            )
        if anchors is not None:
            self.anchor_embed = nn.Linear(self.action_space.flat_dim, feat_size)
            self.classifier = build_mlp(feat_size, anchor_hidden, num_modes, layers=1, dropout=dropout)

    # ---- helpers --------------------------------------------------------------------------
    def anchor_actions(self) -> torch.Tensor:
        return self.normalizer.normalize(self.action_space.targets_from_poses(self.anchors.poses))

    def example_noise(self, batch_size, device=None):
        return torch.randn(batch_size, self.num_modes, self.num_pts, self.action_dim, device=device)

    def _sample(self, cond, tokens, noise):
        """Walk ``noise`` (t=1) back to a clean trajectory (t=0); export unrolls this loop."""
        x = noise
        times = self.scheduler.step_times(self.sample_steps)
        for t, t_next in zip(times[:-1], times[1:]):
            t_now = x.new_full((x.shape[0],), t)
            out = self.denoiser(x, t_now, cond, tokens)
            x = self.scheduler.step(out, x, t_now, x.new_full((x.shape[0],), t_next))
        return x

    # ---- forward ---------------------------------------------------------------------------
    def decode(self, cond, tokens, noise=None):
        n = cond.shape[0]
        if self.training:
            # Loss needs only the conditioning; sampling is skipped and plans stay zero.
            return PlanOutput(plans=cond.new_zeros(n, self.flat_size), cond=cond, tokens=tokens)
        if noise is None:
            noise = self.example_noise(n, cond.device).to(cond.dtype)
        if tuple(noise.shape) != (n, self.num_modes, self.num_pts, self.action_dim):
            raise ValueError(
                f"noise must be (N, {self.num_modes}, {self.num_pts}, {self.action_dim}), got {tuple(noise.shape)}"
            )
        m = self.num_modes
        cond_rep = cond.repeat_interleave(m, dim=0)
        tokens_rep = tokens.repeat_interleave(m, dim=0)
        logits = None
        if self.anchors is not None:
            anchor_actions = self.anchor_actions().to(cond.dtype)  # (K, T, A)
            cond_rep = cond_rep + self.anchor_embed(anchor_actions.flatten(1)).repeat(n, 1)
            logits = self.classifier(cond)
        samples = self._sample(cond_rep, tokens_rep, noise.flatten(0, 1)).reshape(n, m, self.num_pts, self.action_dim)
        if self.anchors is not None:
            samples = samples + anchor_actions[None]
        mu = samples
        return PlanOutput(plans=self.pack(mu, None, logits), mu=mu, logits=logits, cond=cond, tokens=tokens)

    # ---- loss ---------------------------------------------------------------------------
    def loss(self, preds, gt, targets):
        cond, tokens = preds.cond, preds.tokens
        x0 = gt
        cls = torch.zeros((), device=gt.device, dtype=gt.dtype)
        if self.anchors is not None:
            label = self.anchors.nearest(targets["future_poses"])
            anchor_actions = self.anchor_actions().to(gt.dtype)[label]
            x0 = gt - anchor_actions
            cond = cond + self.anchor_embed(anchor_actions.flatten(1))
            cls = F.cross_entropy(self.classifier(preds.cond), label)
        t = self.scheduler.sample_t(x0.shape[0], device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.scheduler.add_noise(x0, noise, t)
        prediction = self.denoiser(x_t, t, cond, tokens)
        per_sample = F.mse_loss(
            self.scheduler.predicted(prediction, x_t, t), self.scheduler.target(x0, noise, t), reduction="none"
        ).mean(dim=(1, 2))
        reg = per_sample.mean()
        return dict(total=reg + self.loss_cls_alpha * cls, reg=reg, cls=cls), dict(
            imitation_loss_per_sample=per_sample.detach()
        )
