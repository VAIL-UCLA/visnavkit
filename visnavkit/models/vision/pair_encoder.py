"""FlowPilot-DST frame encoder: [frame_t | frame_t-1] through a 6-channel timm backbone + ``SpeedHead``."""

import timm
import torch
import torch.nn as nn

from visnavkit.models.layers.masking import masked_rows
from visnavkit.models.vision.speed_head import SpeedHead
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


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
