"""FlowPilot-DST temporal stage: causal self-attention over the slots with frames."""

import torch
import torch.nn as nn

from .causal import generate_causal_mask


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
