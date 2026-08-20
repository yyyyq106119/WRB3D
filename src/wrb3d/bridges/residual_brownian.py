"""One residual Brownian-bridge implementation shared by low and high bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn


@dataclass
class SamplingResult:
    residual: Tensor
    timesteps: list[int]
    trajectory: list[Tensor] | None = None


def _as_batch_timestep(t: int | Tensor, batch: int, device: torch.device) -> Tensor:
    if isinstance(t, Tensor):
        out = t.to(device=device, dtype=torch.long).flatten()
    else:
        out = torch.tensor([int(t)], device=device, dtype=torch.long)
    if out.numel() == 1 and batch > 1:
        out = out.expand(batch)
    if out.numel() != batch:
        raise ValueError(f"timestep batch mismatch: expected {batch}, got {out.numel()}")
    return out


class ResidualBrownianBridge(nn.Module):
    """Brownian bridge from clean residual at ``t=0`` to zero at ``t=T``.

    ``q_sample`` implements

    ``r_t = (1-m_t) r_0 + sqrt(2 m_t (1-m_t) covariance) epsilon``.

    Covariance may be scalar, channel/band-wise, case-wise, or spatial.  The
    same class is used for every number of channels; low/high branches never
    own separate schedules or posterior formulas.
    """

    def __init__(self, num_timesteps: int = 1000, schedule: str = "linear") -> None:
        super().__init__()
        if int(num_timesteps) < 2:
            raise ValueError("num_timesteps must be >= 2")
        if schedule != "linear":
            raise ValueError("the formal baseline currently supports only a linear schedule")
        self.num_timesteps = int(num_timesteps)
        self.schedule_name = schedule
        self.register_buffer("m_t", torch.linspace(0.0, 1.0, self.num_timesteps + 1))

    def bridge_time(self, t: int | Tensor, like: Tensor) -> Tensor:
        index = _as_batch_timestep(t, like.shape[0], like.device)
        if torch.any(index < 0) or torch.any(index > self.num_timesteps):
            raise IndexError(f"timestep must be within [0,{self.num_timesteps}]")
        values = self.m_t.to(device=like.device, dtype=like.dtype).gather(0, index)
        return values.reshape(index.shape[0], *([1] * (like.ndim - 1)))

    @staticmethod
    def _broadcast_covariance(covariance: float | Tensor, like: Tensor) -> Tensor:
        # Variance and square-root calculations must remain FP32 under AMP.
        # A positive covariance can otherwise underflow to exactly zero before
        # sqrt, even when its standard deviation is representable in FP16.
        work_dtype = (
            torch.float32 if like.dtype in {torch.float16, torch.bfloat16} else like.dtype
        )
        cov = torch.as_tensor(covariance, device=like.device, dtype=work_dtype)
        if cov.ndim == 0:
            cov = cov.reshape(*([1] * like.ndim))
        elif cov.ndim == 1:
            if cov.numel() == 1:
                cov = cov.reshape(*([1] * like.ndim))
            elif cov.numel() == like.shape[1]:
                cov = cov.reshape(1, like.shape[1], *([1] * (like.ndim - 2)))
            elif like.shape[1] % cov.numel() == 0:
                cov = cov.repeat_interleave(like.shape[1] // cov.numel())
                cov = cov.reshape(1, like.shape[1], *([1] * (like.ndim - 2)))
            else:
                raise ValueError("1-D covariance must be scalar, per-channel, or divide channel count")
        elif cov.ndim == 2:
            if cov.shape[0] not in {1, like.shape[0]}:
                raise ValueError("case-wise covariance batch mismatch")
            if cov.shape[1] != like.shape[1]:
                if like.shape[1] % cov.shape[1] != 0:
                    raise ValueError("case-wise covariance channel mismatch")
                cov = cov.repeat_interleave(like.shape[1] // cov.shape[1], dim=1)
            cov = cov.reshape(cov.shape[0], cov.shape[1], *([1] * (like.ndim - 2)))
        elif cov.ndim == like.ndim - 1 and tuple(cov.shape) == tuple(like.shape[1:]):
            cov = cov.unsqueeze(0)
        elif cov.ndim != like.ndim:
            raise ValueError(f"unsupported covariance shape {tuple(cov.shape)} for {tuple(like.shape)}")
        try:
            cov = torch.broadcast_to(cov, like.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"covariance shape {tuple(cov.shape)} cannot broadcast to {tuple(like.shape)}"
            ) from exc
        if not torch.isfinite(cov).all() or torch.any(cov < 0):
            raise ValueError("covariance must be finite and non-negative")
        return cov

    def marginal_parameters(
        self,
        clean_residual: Tensor,
        t: int | Tensor,
        covariance: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        work = (
            clean_residual.float()
            if clean_residual.dtype in {torch.float16, torch.bfloat16}
            else clean_residual
        )
        m = self.bridge_time(t, work)
        cov = self._broadcast_covariance(covariance, work)
        mean = (1.0 - m) * work
        variance = 2.0 * m * (1.0 - m) * cov
        return mean, variance

    def q_sample(
        self,
        clean_residual: Tensor,
        t: int | Tensor,
        covariance: float | Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        mean, variance = self.marginal_parameters(clean_residual, t, covariance)
        epsilon = torch.randn_like(mean) if noise is None else noise.to(mean)
        if epsilon.shape != clean_residual.shape:
            raise ValueError("noise and clean_residual shapes must match")
        return mean + torch.sqrt(variance.clamp_min(0.0)) * epsilon, epsilon

    @staticmethod
    def predict_endpoint(
        model_output: Tensor,
        prediction_target: str = "residual_x0",
        mri_coefficient: Tensor | None = None,
    ) -> Tensor:
        if prediction_target == "residual_x0":
            return model_output
        if prediction_target == "endpoint_x0_legacy":
            if mri_coefficient is None:
                raise ValueError("legacy endpoint output requires the MRI coefficient anchor")
            return model_output - mri_coefficient
        raise ValueError(f"unsupported prediction_target={prediction_target!r}")

    def posterior_parameters(
        self,
        state: Tensor,
        clean_residual_hat: Tensor,
        t_current: int | Tensor,
        t_next: int | Tensor,
        covariance: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        if state.shape != clean_residual_hat.shape:
            raise ValueError("state and endpoint prediction shapes must match")
        current = _as_batch_timestep(t_current, state.shape[0], state.device)
        nxt = _as_batch_timestep(t_next, state.shape[0], state.device)
        if torch.any(current <= nxt):
            raise ValueError("reverse sampling requires t_current > t_next")
        work_state = state.float() if state.dtype in {torch.float16, torch.bfloat16} else state
        work_endpoint = (
            clean_residual_hat.float()
            if clean_residual_hat.dtype in {torch.float16, torch.bfloat16}
            else clean_residual_hat
        )
        u = self.bridge_time(current, work_state)
        v = self.bridge_time(nxt, work_state)
        if torch.any(v >= u):
            raise ValueError("reverse sampling requires m_next < m_current")
        mean_at_u = (1.0 - u) * work_endpoint
        mean = (1.0 - v) * work_endpoint + (v / u.clamp_min(1e-12)) * (
            work_state - mean_at_u
        )
        cov = self._broadcast_covariance(covariance, work_state)
        variance = 2.0 * v * (u - v) / u.clamp_min(1e-12) * cov
        terminal = v <= 0.0
        mean = torch.where(terminal, clean_residual_hat, mean)
        variance = torch.where(terminal, torch.zeros_like(variance), variance.clamp_min(0.0))
        return mean, variance

    def posterior_mean(
        self,
        state: Tensor,
        clean_residual_hat: Tensor,
        t_current: int | Tensor,
        t_next: int | Tensor,
        covariance: float | Tensor,
    ) -> Tensor:
        return self.posterior_parameters(
            state, clean_residual_hat, t_current, t_next, covariance
        )[0]

    def sample_step(
        self,
        state: Tensor,
        clean_residual_hat: Tensor,
        t_current: int | Tensor,
        t_next: int | Tensor,
        covariance: float | Tensor,
        *,
        stochastic: bool = False,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        mean, variance = self.posterior_parameters(
            state, clean_residual_hat, t_current, t_next, covariance
        )
        if not stochastic:
            return mean
        epsilon = (
            torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
            if noise is None
            else noise.to(mean)
        )
        if epsilon.shape != state.shape:
            raise ValueError("reverse noise and state shapes must match")
        return mean + torch.sqrt(variance) * epsilon

    def sampling_timesteps(self, num_steps: int, device: torch.device | str) -> Tensor:
        count = int(num_steps)
        if count < 1:
            raise ValueError("num_steps must be positive")
        count = min(count, self.num_timesteps)
        steps = torch.linspace(self.num_timesteps, 0, count + 1, device=device).round().long()
        steps = torch.unique_consecutive(steps)
        if steps[0].item() != self.num_timesteps or steps[-1].item() != 0:
            raise RuntimeError("sampling grid lost a bridge endpoint")
        return steps

    @torch.no_grad()
    def sample_loop(
        self,
        predict_endpoint: Callable[..., Tensor],
        initial_state: Tensor,
        covariance: float | Tensor,
        *,
        num_steps: int = 15,
        stochastic: bool = False,
        generator: torch.Generator | None = None,
        model_kwargs: dict[str, Any] | None = None,
        return_trajectory: bool = False,
    ) -> SamplingResult:
        steps = self.sampling_timesteps(num_steps, initial_state.device)
        state = initial_state
        trajectory = [state.clone()] if return_trajectory else None
        kwargs = model_kwargs or {}
        for current, nxt in zip(steps[:-1], steps[1:]):
            t = current.repeat(state.shape[0])
            endpoint = predict_endpoint(state, t, **kwargs)
            state = self.sample_step(
                state,
                endpoint,
                current,
                nxt,
                covariance,
                stochastic=stochastic,
                generator=generator,
            )
            if trajectory is not None:
                trajectory.append(state.clone())
        return SamplingResult(state, [int(v) for v in steps.cpu().tolist()], trajectory)

    def sample_training_timesteps(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        mode: str = "endpoint_mixture",
        endpoint_probability: float = 0.15,
    ) -> Tensor:
        if mode not in {"uniform_internal", "beta_internal", "endpoint_mixture"}:
            raise ValueError(f"unsupported t sampling mode={mode!r}")
        if self.num_timesteps <= 2:
            interior = torch.ones(batch_size, device=device, dtype=torch.long)
        elif mode == "beta_internal":
            values = torch.distributions.Beta(2.0, 2.0).sample((batch_size,)).to(device)
            interior = 1 + torch.floor(values * (self.num_timesteps - 1)).long()
            interior.clamp_(1, self.num_timesteps - 1)
        else:
            interior = torch.randint(1, self.num_timesteps, (batch_size,), device=device)
        if mode != "endpoint_mixture":
            return interior
        if not 0.0 <= float(endpoint_probability) <= 1.0:
            raise ValueError("endpoint_probability must be within [0,1]")
        mask = torch.rand(batch_size, device=device) < float(endpoint_probability)
        return torch.where(mask, torch.full_like(interior, self.num_timesteps), interior)
