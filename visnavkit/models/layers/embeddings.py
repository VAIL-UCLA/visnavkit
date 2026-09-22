import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of continuous timesteps ``t in [0, 1]`` (scaled by 1000) -> ``(N, dim)``."""
    if dim < 2:
        raise ValueError("dim must be at least 2")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * 1000.0 * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, emb.new_zeros(emb.shape[0], 1)], dim=-1)
    return emb.to(t.dtype if t.is_floating_point() else torch.float32)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal features followed by a two-layer MLP, as in diffusion policy / DiT."""

    def __init__(self, dim: int, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or dim
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.dim))


def sincos_2d(h, w, dim, device, dtype):
    """Fixed 2-D sin-cos positional embedding ``(h w, dim)``, ``dim / 4`` frequencies per axis."""
    quarter = dim // 4
    f32 = dict(device=device, dtype=torch.float32)  # float32 throughout: the tracer promotes int arange to float64
    freq = torch.exp(-math.log(10000.0) * torch.arange(quarter, **f32) / quarter)
    yy = (torch.arange(h, **f32)[:, None, None] * freq).expand(h, w, quarter)
    xx = (torch.arange(w, **f32)[None, :, None] * freq).expand(h, w, quarter)
    return torch.cat([yy.sin(), yy.cos(), xx.sin(), xx.cos()], -1).reshape(h * w, dim).to(dtype)
