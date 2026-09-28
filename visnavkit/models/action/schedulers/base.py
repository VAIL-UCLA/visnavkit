"""Noise schedulers on continuous time ``t in [0, 1]`` (``t=1`` pure noise, ``t=0`` clean)."""

import torch
import torch.nn as nn

__all__ = ["BaseScheduler", "PARAMETERIZATIONS"]

PARAMETERIZATIONS = ("eps", "x", "v")


class BaseScheduler(nn.Module):
    """Training corrupts ``x0`` at a sampled ``t``; sampling walks ``step_times`` from 1 to 0.

    ``x_t = a(t) x0 + s(t) eps``. The denoiser outputs ``prediction`` (eps | x | v) and the loss is the MSE in the
    ``loss`` space (eps | x | v): 3 x 3 combinations, converted through (x0, eps). v is linear in (x0, eps) with
    determinant 1 against x_t (flow: eps - x0; diffusion: a eps - s x0, Salimans & Ho 2022). The two divisions
    (eps -> x0 by a, x -> eps by s) are floored at ``min_scale`` in the loss, as JiT (Li & He 2025) clips 1 - t at
    0.05; prediction == loss regresses the output directly.
    """

    def __init__(self, prediction: str, loss: str, min_scale: float = 0.05):
        super().__init__()
        if prediction not in PARAMETERIZATIONS or loss not in PARAMETERIZATIONS:
            raise ValueError(f"prediction and loss must be in {PARAMETERIZATIONS}, got {prediction!r}, {loss!r}")
        self.prediction, self.loss, self.min_scale = prediction, loss, min_scale

    def coeffs(self, t: torch.Tensor):
        """``(a, s)`` of ``x_t = a x0 + s eps``, each ``(N,)``."""
        raise NotImplementedError

    def v_coeffs(self, a, s):
        """``(c_x, c_eps)`` of ``v = c_x x0 + c_eps eps``, with ``a c_eps - s c_x = 1``."""
        raise NotImplementedError

    def split(self, out, x_t, t, floor=None):
        """The denoiser output -> the (x0, eps) it implies at ``x_t``."""
        a, s = (self._broadcast(c, x_t) for c in self.coeffs(t))
        if self.prediction == "eps":
            return (x_t - s * out) / (a.clamp(min=floor) if floor else a), out
        if self.prediction == "x":
            return out, (x_t - a * out) / (s.clamp(min=floor) if floor else s)
        c_x, c_eps = self.v_coeffs(a, s)
        return c_eps * x_t - s * out, a * out - c_x * x_t

    def space(self, x0, eps, t, name):
        """(x0, eps) -> the ``name`` (eps | x | v) quantity."""
        if name == "x":
            return x0
        if name == "eps":
            return eps
        c_x, c_eps = self.v_coeffs(*(self._broadcast(c, x0) for c in self.coeffs(t)))
        return c_x * x0 + c_eps * eps

    def predicted(self, out, x_t, t):
        """The denoiser output in the loss space."""
        if self.prediction == self.loss:
            return out
        return self.space(*self.split(out, x_t, t, self.min_scale), t, self.loss)

    def sample_t(self, n: int, device=None) -> torch.Tensor:
        raise NotImplementedError

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The loss-space target: MSE(``predicted(out, x_t, t)``, ``target(x0, noise, t)``)."""
        return self.space(x0, noise, t, self.loss)

    def step_times(self, num_steps: int) -> list[float]:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        return torch.linspace(1.0, 0.0, num_steps + 1).tolist()

    def step(
        self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _broadcast(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return value.reshape(-1, *([1] * (like.ndim - 1))).to(like.dtype)
