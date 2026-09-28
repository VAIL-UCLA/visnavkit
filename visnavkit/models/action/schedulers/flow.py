"""Rectified flow matching: linear interpolation, Euler sampling; eps / x / v prediction and loss (``BaseScheduler``)."""

import torch

from .base import BaseScheduler

__all__ = ["FlowMatchingScheduler"]

TIME_SAMPLING = ("uniform", "logit_normal", "beta")


class FlowMatchingScheduler(BaseScheduler):
    """``x_t = (1 - t) x0 + t noise``, v = ``noise - x0``, Euler steps back to 0.

    Args:
        prediction, loss: eps | x | v (``BaseScheduler``); v / v is rectified flow, x / v JiT (Li & He 2025).
        min_scale: floor of the 1 - t and t divisions (loss and sampling).
        time_sampling: training-time distribution over ``t``. ``uniform``; ``logit_normal``
            (Stable Diffusion 3, concentrated near t=0.5); ``beta`` (openpi pi0, ``Beta(1.5, 1)``
            weighted toward the noisy end where the trajectory is still undecided).
        logit_normal_std: spread of the ``logit_normal`` sampler.
        beta_alpha, beta_beta: shape of the ``beta`` sampler.
        shift: sampling-time resolution shift ``t -> s t / (1 + (s - 1) t)`` (diffusers
            ``FlowMatchEulerDiscreteScheduler``); ``s > 1`` spends more of the step budget at
            high noise. Training times are unshifted, as in SD3/Flux.
    """

    def __init__(
        self,
        time_sampling: str = "uniform",
        logit_normal_std: float = 1.0,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        shift: float = 1.0,
        prediction: str = "v",
        loss: str = "v",
        min_scale: float = 0.05,
    ):
        super().__init__(prediction, loss, min_scale)
        if time_sampling not in TIME_SAMPLING:
            raise ValueError(f"time_sampling must be one of {TIME_SAMPLING}, got {time_sampling!r}")
        if min(beta_alpha, beta_beta) <= 0 or shift <= 0:
            raise ValueError("beta_alpha, beta_beta and shift must be positive")
        self.time_sampling = time_sampling
        self.logit_normal_std = logit_normal_std
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.shift = shift

    def sample_t(self, n, device=None):
        if self.time_sampling == "uniform":
            return torch.rand(n, device=device)
        if self.time_sampling == "logit_normal":
            return torch.sigmoid(torch.randn(n, device=device) * self.logit_normal_std)
        concentration = torch.tensor([self.beta_alpha, self.beta_beta], device=device)
        return torch.distributions.Beta(concentration[0], concentration[1]).sample((n,))

    def step_times(self, num_steps):
        times = super().step_times(num_steps)
        if self.shift == 1.0:
            return times
        return [self.shift * t / (1 + (self.shift - 1) * t) for t in times]

    def add_noise(self, x0, noise, t):
        t = self._broadcast(t, x0)
        return (1 - t) * x0 + t * noise

    def coeffs(self, t):
        return 1 - t, t

    def v_coeffs(self, a, s):
        return -1.0, 1.0

    def step(self, model_output, x_t, t, t_next):
        if self.prediction == "v":
            velocity = model_output
        else:  # t = 1 (pure noise) leaves x0 of an eps output undetermined: the floor bounds it
            x0, eps = self.split(model_output, x_t, t, self.min_scale)
            velocity = eps - x0
        return x_t + self._broadcast(t_next - t, x_t) * velocity
