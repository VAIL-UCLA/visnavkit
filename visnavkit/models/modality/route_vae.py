"""FlowPilot-DST route stage: the frozen route VAE encoder on per-slot route patches."""

import torch
import torch.nn as nn

from visnavkit.models.layers.masking import masked_rows
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


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
