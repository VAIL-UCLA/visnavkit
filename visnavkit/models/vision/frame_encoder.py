"""FrameEncoder: one RGB frame per slot through a timm backbone, ``PairEncoder``'s interface without the pair or the
speed head (``flowpilot_step_dst``). A single frame carries no motion: speed zeros, ``pair_mask`` all False."""

import timm
import torch
import torch.nn as nn

from visnavkit.models.layers.masking import masked_rows

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class FrameEncoder(nn.Module):
    def __init__(self, backbone_name="fastvit_sa12", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, features_only=True, out_indices=(-1,)
        )
        self.dim = self.backbone.feature_info.channels()[-1]
        self.stride = self.backbone.feature_info.reduction()[-1]
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, frames, frame_mask):
        """``(B, T, 3, H, W)`` in [0, 1], ``(B, T)`` -> global (B, T, C), patches (B, T, C, gh, gw), speed (B, T, 1), pair_mask (B, T)."""
        b, t = frame_mask.shape
        s = self.stride  # ceil with positive operands only: ONNX integer Div truncates toward zero
        gh, gw = (frames.shape[-2] + s - 1) // s, (frames.shape[-1] + s - 1) // s
        glob = frames.new_zeros(b * t, self.dim)
        patches = frames.new_zeros(b * t, self.dim, gh, gw)
        idx = masked_rows(frame_mask)
        if len(idx):
            p = self.backbone((frames.flatten(0, 1)[idx] - self.mean) / self.std)[-1]
            patches[idx], glob[idx] = p.to(patches.dtype), p.mean((2, 3)).to(glob.dtype)
        return (
            glob.view(b, t, -1),
            patches.view(b, t, self.dim, gh, gw),
            frames.new_zeros(b, t, 1),
            torch.zeros_like(frame_mask),
        )
