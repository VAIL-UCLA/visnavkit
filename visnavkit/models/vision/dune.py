"""DuneEncoder: a frozen DUNE ViT-B/14 as FlowPilot-DST's frame encoder (``flowpilot_dune_dst``).

One frame per slot (no pair): [0, 1] -> ImageNet mean / std, zero-padded right / bottom to a multiple of 14
(216 x 384 -> 224 x 392 -> 16 x 28 cells), the post-norm patch tokens (no CLS, registers dropped) -> global = their
mean over the full grid, patches = ``downscale`` avg-pooled (4 -> 4 x 7). ``adapt``: one trainable Linear(768, 768),
identity at init, on both (it commutes with the mean / pool). The ViT is frozen: eval mode and no_grad always.
``weights``: a Lightning checkpoint of the reference UnifiedModel (its ``model.encoder.vision.encoder.*``) or that
encoder's state dict saved alone; null keeps the torch.hub DUNE weights. Speed: a single frame carries no motion, so there
is no speed head and no whole pair (``pair_mask`` all False: the speed loss is 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DONOR_PREFIX = "model.encoder.vision.encoder."


class DuneEncoder(nn.Module):
    def __init__(self, weights=None, downscale=4, hub_name="dune_vitbase_14_448_encoder"):
        super().__init__()
        self.encoder = torch.hub.load("naver/dune", hub_name, skip_validation=True, trust_repo=True)
        if weights is not None:
            self.encoder.load_state_dict(self.donor_state(weights), strict=True)
        self.encoder.requires_grad_(False).eval()
        self.dim, self.patch_size, self.downscale = self.encoder.embed_dim, self.encoder.patch_size, downscale
        self.stride = self.patch_size * downscale
        self.adapt = nn.Linear(self.dim, self.dim)
        nn.init.eye_(self.adapt.weight), nn.init.zeros_(self.adapt.bias)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        logger.info(f"DuneEncoder: {hub_name} frozen, weights {weights or 'torch.hub'}, patches pooled x{downscale}")

    @staticmethod
    def donor_state(path):
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if "state_dict" not in state:
            return state
        return {k[len(DONOR_PREFIX) :]: v for k, v in state["state_dict"].items() if k.startswith(DONOR_PREFIX)}

    def train(self, mode=True):  # the frozen ViT stays in eval mode
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, frames, frame_mask):
        """``(B, T, 3, H, W)`` in [0, 1], ``(B, T)`` -> global (B, T, C), patches (B, T, C, gh, gw), speed (B, T, 1)
        zeros, pair_mask (B, T) all False."""
        b, t, _, h, w = frames.shape
        ps, d = self.patch_size, self.downscale
        gh, gw = (h + ps - 1) // ps, (w + ps - 1) // ps
        glob = frames.new_zeros(b * t, self.dim)
        patches = frames.new_zeros(b * t, self.dim, gh // d, gw // d)
        idx = frame_mask.reshape(-1).nonzero().squeeze(1)
        if len(idx):
            x = (frames.flatten(0, 1)[idx] - self.mean) / self.std
            with torch.no_grad():
                tokens = self.encoder(F.pad(x, (0, gw * ps - w, 0, gh * ps - h)))["x_norm_patchtokens"]
            grid = tokens.float().transpose(1, 2).reshape(len(idx), self.dim, gh, gw)
            pooled = F.avg_pool2d(grid, d) if d > 1 else grid
            glob[idx] = self.adapt(grid.mean((2, 3))).to(glob.dtype)
            patches[idx] = self.adapt(pooled.movedim(1, -1)).movedim(-1, 1).to(patches.dtype)
        speed = frames.new_zeros(b, t, 1)
        return glob.view(b, t, -1), patches.view(b, t, self.dim, gh // d, gw // d), speed, torch.zeros_like(frame_mask)
