"""DDPM forward process with deterministic DDIM (eta=0) sampling."""

import math

import torch

from .base import BaseScheduler

__all__ = ["DDIMScheduler"]


def make_betas(train_timesteps: int, beta_schedule: str, beta_start: float, beta_end: float) -> torch.Tensor:
    """``squaredcos_cap_v2`` (cosine, diffusion policy / NoMaD default) is step-count independent;
    ``linear`` is the DDPM schedule and only reaches pure noise with ~1000 steps."""
    if beta_schedule == "linear":
        return torch.linspace(beta_start, beta_end, train_timesteps)
    if beta_schedule == "squaredcos_cap_v2":
        steps = torch.arange(train_timesteps + 1, dtype=torch.float64) / train_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999).float()
    raise ValueError("beta_schedule must be 'squaredcos_cap_v2' or 'linear'")


class DDIMScheduler(BaseScheduler):
    """DDPM forward process with deterministic DDIM sampling; eps / x / v prediction and loss (``BaseScheduler``).

    Args:
        prediction, loss: eps | x | v; eps / eps is DDPM, v is Salimans & Ho's ``sqrt(ab) eps - sqrt(1 - ab) x0``.
        min_scale: floor of the sqrt(ab) and sqrt(1 - ab) divisions in the loss (sampling divides exactly: both are
            positive on the sampling times, and ``clip_sample`` bounds x0).
        clip_sample: symmetric clamp on the predicted clean sample, in the decoder's
            **normalized action units** (diffusers clips at 1.0 for its ``[-1, 1]`` data). It
            keeps the ``1 / sqrt(alpha_bar)`` term bounded near ``t = 1``; without a normalizer
            the units are raw metres, so the clamp has to exceed the longest plan or it
            truncates it. ``GenerativeDecoder`` warns about that combination.
    """

    def __init__(
        self,
        train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: float | None = 4.0,
        prediction: str = "eps",
        loss: str = "eps",
        min_scale: float = 0.05,
    ):
        super().__init__(prediction, loss, min_scale)
        if train_timesteps < 2:
            raise ValueError("train_timesteps must be at least 2")
        self.train_timesteps = train_timesteps
        self.clip_sample = clip_sample
        betas = make_betas(train_timesteps, beta_schedule, beta_start, beta_end)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        index = (t.float() * (self.train_timesteps - 1)).round().long().clamp(0, self.train_timesteps - 1)
        alpha = self.alphas_cumprod[index]
        return torch.where(t <= 0, torch.ones_like(alpha), alpha)

    def sample_t(self, n, device=None):
        return torch.randint(0, self.train_timesteps, (n,), device=device).float() / (self.train_timesteps - 1)

    def add_noise(self, x0, noise, t):
        alpha = self._broadcast(self._alpha(t), x0)
        return alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise

    def coeffs(self, t):
        alpha = self._alpha(t)
        return alpha.sqrt(), (1 - alpha).sqrt()

    def v_coeffs(self, a, s):
        return -s, a

    def step(self, model_output, x_t, t, t_next):
        alpha_next = self._broadcast(self._alpha(t_next), x_t)
        x0, eps = self.split(model_output, x_t, t)
        if self.clip_sample is not None:
            x0 = x0.clamp(-self.clip_sample, self.clip_sample)
        return alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * eps
