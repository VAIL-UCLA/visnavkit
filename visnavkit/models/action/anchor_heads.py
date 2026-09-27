"""FlowMatchingPolicy's anchor heads: FlowPilot's anchor-mode-token DiT (``AnchorFlowHead``: one token per k-means anchor,
adaLN on time, cross-attention to the kv, per mode a dx, dy output, the state [dyaw, v, w] and a score) with the flow
changed. Both are deterministic and decode the top-score mode: 1 plan per decision.

- ``S2EHead`` (``flow_matching_policy_s2e``): anchors in, trajectories out, one pass, no flow (FlowPilot minus flow
  matching). Per mode dx, dy = anchor + residual.
- ``FlowBridgeHead`` (``flow_matching_policy_bridge``): flow matching from the anchor to the trajectory,
  x_t = (1 - t) anchor + t x1, velocity target x1 - anchor, ``sample_steps`` Euler steps from the anchors. As BridgeDrive
  (arXiv:2509.23589, github.com/ShuLiu-ETHZ/BridgeDrive), the anchor is the bridge's source and the scores come from the
  anchors alone (here the t = 0 pass, the first Euler step), never from a GT-mixed x_t; BridgeDrive's bridge is a DDBM
  (VP schedule, x0 prediction, stochastic), this one the straight noise-free flow.

Loss on the anchor nearest the GT (metric ADE): the dx, dy term + MSE(state) + CE(score).
"""

import torch
import torch.nn.functional as F

from visnavkit.models.action.anchor_flow import AnchorFlowHead
from visnavkit.models.action.step_flow import StepFlowHead


class _AnchorHead(AnchorFlowHead):
    uses_noise = False
    pack, parse_output = StepFlowHead.pack, StepFlowHead.parse_output

    def nearest(self, actions, bounds):
        """The anchor nearest the metric GT ``(N, T, 5)`` by mean waypoint distance -> ``(N,)``."""
        lo, hi = bounds[:, None, None, 0, :2], bounds[:, None, None, 1, :2]
        paths = ((self.anchors[None] + 1) / 2 * (hi - lo) + lo).cumsum(2)
        return (paths - actions[:, None, :, :2]).norm(dim=-1).mean(-1).argmin(1)

    def pass_(self, x, t, kv, ego):
        """One DiT pass over the mode tokens of ``x`` ``(N, K, T, 2)`` at time ``t`` ``(N,)`` -> dx, dy out, state, score."""
        return self.denoise(x, t, self.time_embed(t).to(kv.dtype), kv, ego)

    def setup(self, kv, bounds, ego_vw):
        n = len(kv)
        anchors = self.anchors[None].expand(n, -1, -1, -1)
        return n, self.kv_norm(kv), anchors, self.ego(self.ego_cond(ego_vw, bounds)).to(kv.dtype)

    def losses(self, xy, state, score, x1, winner, xy_target):
        rows = torch.arange(len(winner), device=winner.device)
        v = F.mse_loss(xy[rows, winner].float(), xy_target, reduction="none").sum(-1).mean()
        s = F.mse_loss(state[rows, winner].float(), x1[..., 2:], reduction="none").sum(-1).mean()
        ce = F.cross_entropy(score.float(), winner)
        wv, ws, wc = self.weights
        return dict(total=wv * v + ws * s + wc * ce, reg=v + s, cls=ce, xy=v, state=s)

    def best(self, xy, state, score, bounds):
        """The top-score mode -> metric ``(N, 1, T, 5)``."""
        rows = torch.arange(len(score), device=score.device)
        best = score.argmax(1)
        return self.metric(torch.cat([xy[rows, best], state[rows, best].float()], -1), bounds)[:, None]


class S2EHead(_AnchorHead):
    def forward_modes(self, kv, bounds, ego_vw):
        n, kv, anchors, ego = self.setup(kv, bounds, ego_vw)
        t = torch.ones(n, device=kv.device)  # no flow: a constant adaLN time
        residual, state, score = self.pass_(anchors, t, kv, ego)
        return anchors + residual.float(), state, score

    def loss(self, kv, actions, bounds, ego_vw):
        x1 = self.norm(actions, bounds)
        xy, state, score = self.forward_modes(kv, bounds, ego_vw)
        return self.losses(xy, state, score, x1, self.nearest(actions, bounds), x1[..., :2])

    @torch.no_grad()
    def sample(self, kv, bounds, ego_vw, noise=None):
        return self.best(*self.forward_modes(kv, bounds, ego_vw), bounds)


class FlowBridgeHead(_AnchorHead):
    def loss(self, kv, actions, bounds, ego_vw):
        n, kv, anchors, ego = self.setup(kv, bounds, ego_vw)
        x1 = self.norm(actions, bounds)
        winner = self.nearest(actions, bounds)
        rows = torch.arange(n, device=kv.device)
        _, _, score = self.pass_(anchors, torch.zeros(n, device=kv.device), kv, ego)  # the anchors alone: the scores
        t = self.sample_time(n, kv.device)
        tt = t[:, None, None, None]
        x_t = (1 - tt) * anchors + tt * x1[:, None, :, :2]  # each anchor bridged to the GT
        velocity, state, _ = self.pass_(x_t, t, kv, ego)
        return self.losses(velocity, state, score, x1, winner, x1[..., :2] - anchors[rows, winner])

    @torch.no_grad()
    def sample(self, kv, bounds, ego_vw, noise=None):
        n, kv, x, ego = self.setup(kv, bounds, ego_vw)
        dt, score = 1.0 / self.sample_steps, None
        for i in range(self.sample_steps):  # Euler from the anchors; the scores from the first (t = 0) pass
            velocity, state, s = self.pass_(x, torch.full((n,), i * dt, device=kv.device), kv, ego)
            score = s if score is None else score
            x = x + dt * velocity.float()
        return self.best(x, state, score, bounds)
