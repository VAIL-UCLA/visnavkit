"""FrameEncoder: one RGB frame per slot through a timm backbone, ``PairEncoder``'s interface without the pair or the
speed head (``flow_matching_policy``). A single frame carries no motion: speed zeros, ``pair_mask`` all False.
``weights``: a Lightning checkpoint whose ``model.frame_encoder.backbone.*`` initialise the backbone (e.g. an earlier
flow_matching_policy run). ``freeze``: as ``DuneEncoder``, the backbone fixed in eval mode + a trainable identity-init
Linear(C, C) adapter on the global and patch features."""

import timm
import torch
import torch.nn as nn

from visnavkit.models.layers.masking import masked_rows
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)
DONOR_PREFIX = "model.frame_encoder.backbone."

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class FrameEncoder(nn.Module):
    def __init__(self, backbone_name="fastvit_sa12", pretrained=True, weights=None, freeze=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, features_only=True, out_indices=(-1,)
        )
        self.dim = self.backbone.feature_info.channels()[-1]
        self.stride = self.backbone.feature_info.reduction()[-1]
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        if weights is not None:
            state = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
            state = {k[len(DONOR_PREFIX) :]: v for k, v in state.items() if k.startswith(DONOR_PREFIX)}
            self.backbone.load_state_dict(state, strict=True)
            logger.info(f"FrameEncoder: {len(state)} backbone tensors from {weights}")
        self.freeze, self.adapt = freeze, None
        if freeze:
            self.backbone.requires_grad_(False).eval()
            self.adapt = nn.Linear(self.dim, self.dim)
            nn.init.eye_(self.adapt.weight), nn.init.zeros_(self.adapt.bias)

    def train(self, mode=True):  # a frozen backbone stays in eval mode (BatchNorm statistics fixed)
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, frames, frame_mask):
        """``(B, T, 3, H, W)`` in [0, 1], ``(B, T)`` -> global (B, T, C), patches (B, T, C, gh, gw), speed (B, T, 1), pair_mask (B, T)."""
        b, t = frame_mask.shape
        s = self.stride  # ceil with positive operands only: ONNX integer Div truncates toward zero
        gh, gw = (frames.shape[-2] + s - 1) // s, (frames.shape[-1] + s - 1) // s
        glob = frames.new_zeros(b * t, self.dim)
        patches = frames.new_zeros(b * t, self.dim, gh, gw)
        idx = masked_rows(frame_mask)
        if len(idx):
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
                p = self.backbone((frames.flatten(0, 1)[idx] - self.mean) / self.std)[-1]
            if self.adapt is not None:  # explicit perms: movedim traces to a negative Transpose perm (ONNX Runtime)
                p = self.adapt(p.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            patches[idx], glob[idx] = p.to(patches.dtype), p.mean((2, 3)).to(glob.dtype)
        return (
            glob.view(b, t, -1),
            patches.view(b, t, self.dim, gh, gw),
            frames.new_zeros(b, t, 1),
            torch.zeros_like(frame_mask),
        )
