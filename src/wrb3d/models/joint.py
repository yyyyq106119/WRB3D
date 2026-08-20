"""Unified MRI-to-PET low/high wavelet residual Brownian bridge model."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from ..bridges import ResidualBrownianBridge
from ..losses import (
    AuxiliaryWeights,
    Projection,
    ResidualBridgeLoss,
    aligned_high_losses,
    build_hotspot_mask,
    gain_regularization,
    hotspot_losses,
)
from ..metrics import high_band_metrics, residual_metrics
from ..wavelets import DWT3D, IDWT3D, add_residuals, compute_residuals
from .high import HighResidualBridgeNet
from .low import LowResidualBridgeNet
from .corrector import CaseAdaptiveWaveletCorrector


class WaveletResidualBridgeModel(nn.Module):
    """Formal model whose stochastic states are cross-modal wavelet residuals."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        condition_dim: int = 128,
        num_timesteps: int = 1000,
        prediction_target: str = "residual_x0",
        low_to_high_condition: str = "feature_gating",
        projection_mode: str = "none",
        projection_beta: float = 10.0,
        loss_kwargs: dict[str, Any] | None = None,
        corrector_kwargs: dict[str, Any] | None = None,
        auxiliary_loss_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if prediction_target not in {"residual_x0", "endpoint_x0_legacy"}:
            raise ValueError("unsupported prediction target")
        self.input_channels = int(input_channels)
        self.prediction_target = prediction_target
        self.low_to_high_condition = low_to_high_condition
        self.dwt = DWT3D()
        self.idwt = IDWT3D()
        self.bridge = ResidualBrownianBridge(num_timesteps=num_timesteps)
        self.low_model = LowResidualBridgeNet(
            input_channels=input_channels,
            channels=channels,
            condition_dim=condition_dim,
            num_timesteps=num_timesteps,
        )
        self.high_model = HighResidualBridgeNet(
            input_channels=input_channels,
            channels=channels,
            low_feature_channels=self.low_model.feature_channels,
            condition_dim=condition_dim,
            num_timesteps=num_timesteps,
            condition_mode=low_to_high_condition,
            include_mri_low=prediction_target == "residual_x0",
        )
        corrector_config = dict(corrector_kwargs or {})
        self.corrector_enabled = bool(corrector_config.pop("enabled", False))
        if self.corrector_enabled:
            self.case_adaptive_wavelet_corrector = CaseAdaptiveWaveletCorrector(
                self.low_model.feature_channels[-1],
                self.high_model.unet.feature_channels[-1],
                modality_channels=input_channels,
                hidden_dim=int(corrector_config.pop("hidden_dim", 128)),
                gamma=float(corrector_config.pop("gamma", 0.15)),
                identity_init=bool(corrector_config.pop("identity_init", True)),
            )
            if corrector_config:
                raise ValueError(f"unknown corrector settings: {sorted(corrector_config)}")
        resolved_loss_kwargs = dict(loss_kwargs or {})
        self.loss_image_domain = str(resolved_loss_kwargs.pop("image_domain", "raw"))
        if self.loss_image_domain not in {"raw", "projected"}:
            raise ValueError("loss image_domain must be raw or projected")
        self.loss_function = ResidualBridgeLoss(**resolved_loss_kwargs)
        self.projection = Projection(projection_mode, projection_beta)
        auxiliary = dict(auxiliary_loss_kwargs or {})
        hotspot = dict(auxiliary.pop("hotspot", {}))
        aligned = dict(auxiliary.pop("aligned_amplitude", {}))
        gain = dict(auxiliary.pop("gain_regularization", {}))
        if auxiliary:
            raise ValueError(f"unknown auxiliary loss groups: {sorted(auxiliary)}")
        self.hotspot_enabled = bool(hotspot.get("enabled", False))
        self.hotspot_foreground_threshold = float(hotspot.get("foreground_threshold", 1e-6))
        self.hotspot_quantile = float(hotspot.get("quantile", 0.90))
        self.hotspot_temperature_scale = float(hotspot.get("temperature_scale", 0.05))
        self.aligned_enabled = bool(aligned.get("enabled", False))
        self.aligned_smooth_l1_beta = float(aligned.get("smooth_l1_beta", 0.1))
        band_epsilon = torch.as_tensor(
            aligned.get("band_epsilon", [1e-8] * 7), dtype=torch.float32
        ).flatten()
        if band_epsilon.numel() != 7 or torch.any(band_epsilon <= 0):
            raise ValueError("aligned-amplitude band_epsilon must contain seven positives")
        self.register_buffer("aligned_band_epsilon", band_epsilon)
        self.gain_regularization_weight = float(gain.get("weight", 0.0))
        if not self.corrector_enabled and self.gain_regularization_weight != 0.0:
            raise ValueError("S2 cannot have gain regularization without a corrector")

    @property
    def architecture_key(self) -> str:
        channel_text = "-".join(str(v) for v in self.low_model.feature_channels)
        return (
            f"wrb3d:v2:C{self.input_channels}:ch{channel_text}:"
            f"target={self.prediction_target}:condition={self.low_to_high_condition}:"
            f"case_corrector={int(self.corrector_enabled)}"
        )

    def parameter_counts(self) -> dict[str, int]:
        low = sum(p.numel() for p in self.low_model.parameters())
        high = sum(p.numel() for p in self.high_model.parameters())
        return {"low": low, "high": high, "total": low + high}

    def _predict_residual_endpoints(
        self,
        state_low: Tensor,
        state_high: Tensor,
        mri_low: Tensor,
        mri_high: Tensor,
        t: Tensor,
        covariance_low: float | Tensor,
        covariance_high: float | Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor | list[Tensor]]]:
        legacy = self.prediction_target == "endpoint_x0_legacy"
        low_network_state = state_low + mri_low if legacy else state_low
        low_output, low_features = self.low_model(
            low_network_state,
            mri_low,
            mri_high,
            t,
            covariance_low,
            return_features=True,
        )
        predicted_low_residual = self.bridge.predict_endpoint(
            low_output, self.prediction_target, mri_low
        )
        high_network_state = state_high + mri_high if legacy else state_high
        low_state_for_high = state_low + mri_low if legacy else state_low
        low_prediction_for_high = low_output if legacy else predicted_low_residual
        high_output = self.high_model(
            high_network_state,
            mri_high,
            mri_low,
            low_state_for_high,
            low_prediction_for_high,
            low_features,
            t,
            covariance_high,
            return_features=True,
        )
        high_output_tensor, high_features = high_output
        predicted_high_raw = self.bridge.predict_endpoint(
            high_output_tensor, self.prediction_target, mri_high
        )
        gains: Tensor | None = None
        if self.corrector_enabled:
            predicted_high_residual, gains = self.case_adaptive_wavelet_corrector(
                low_features[-1],
                high_features[-1],
                predicted_high_raw,
                mri_high,
            )
        else:
            predicted_high_residual = predicted_high_raw
        return predicted_low_residual, predicted_high_residual, {
            "low_model_output": low_output,
            "high_model_output": high_output_tensor,
            "low_features": low_features,
            "high_features": high_features,
            "high_pred_raw": predicted_high_raw,
            "high_pred_corrected": predicted_high_residual,
            "corrector_gains": gains,
        }

    def forward_train(
        self,
        mri: Tensor,
        pet: Tensor,
        covariance_low: float | Tensor,
        covariance_high: float | Tensor,
        *,
        t: Tensor | None = None,
        t_sampling: str = "endpoint_mixture",
        endpoint_probability: float = 0.15,
        noise_low: Tensor | None = None,
        noise_high: Tensor | None = None,
        roi: Tensor | None = None,
        auxiliary_weights: AuxiliaryWeights | dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if mri.shape != pet.shape:
            raise ValueError("paired MRI/PET tensors must have identical shapes")
        mri_low, mri_high, meta = self.dwt(mri)
        pet_low, pet_high, pet_meta = self.dwt(pet)
        if pet_meta.original_shape != meta.original_shape:
            raise ValueError("paired wavelet metadata mismatch")
        target = compute_residuals(mri_low, mri_high, pet_low, pet_high)
        if t is None:
            t = self.bridge.sample_training_timesteps(
                mri.shape[0],
                mri.device,
                mode=t_sampling,
                endpoint_probability=endpoint_probability,
            )
        else:
            t = t.to(device=mri.device, dtype=torch.long).flatten()
        state_low, epsilon_low = self.bridge.q_sample(
            target.low, t, covariance_low, noise=noise_low
        )
        state_high, epsilon_high = self.bridge.q_sample(
            target.high, t, covariance_high, noise=noise_high
        )
        predicted_low, predicted_high, network_info = self._predict_residual_endpoints(
            state_low,
            state_high,
            mri_low,
            mri_high,
            t,
            covariance_low,
            covariance_high,
        )
        predicted_pet_low, predicted_pet_high = add_residuals(
            mri_low, mri_high, predicted_low, predicted_high
        )
        raw_pet = self.idwt(predicted_pet_low, predicted_pet_high, meta)
        image_for_loss = self.projection(raw_pet) if self.loss_image_domain == "projected" else None
        total, logs = self.loss_function(
            predicted_low,
            target.low,
            predicted_high,
            target.high,
            raw_pet,
            pet,
            image_for_loss=image_for_loss,
        )
        components = self.loss_function.components(
            predicted_low,
            target.low,
            predicted_high,
            target.high,
            raw_pet,
            pet,
            image_for_loss=image_for_loss,
        )
        if auxiliary_weights is None:
            weights = AuxiliaryWeights()
        elif isinstance(auxiliary_weights, AuxiliaryWeights):
            weights = auxiliary_weights
        else:
            weights = AuxiliaryWeights(**auxiliary_weights)
        zero = raw_pet.float().sum() * 0.0
        hot_components: dict[str, Tensor] = {
            "hotspot": zero,
            "underestimation": zero,
            "overestimation": zero,
            "valid_case_fraction": zero,
        }
        hotspot_metadata: dict[str, Tensor] = {}
        if self.hotspot_enabled:
            hotspot_mask, valid_hotspot, hotspot_metadata = build_hotspot_mask(
                pet,
                roi=roi,
                foreground_threshold=self.hotspot_foreground_threshold,
                quantile=self.hotspot_quantile,
                temperature_scale=self.hotspot_temperature_scale,
            )
            hot_components = hotspot_losses(
                raw_pet,
                pet,
                hotspot_mask,
                valid_hotspot,
                charbonnier_epsilon=self.loss_function.epsilon,
            )
        directional: dict[str, Tensor] = {
            "aligned_amplitude": zero,
            "orthogonal_error": zero,
            "beta": torch.zeros(
                (mri.shape[0], 7, self.input_channels), device=mri.device
            ),
            "orthogonal_ratio": torch.zeros(
                (mri.shape[0], 7, self.input_channels), device=mri.device
            ),
            "valid": torch.zeros(
                (mri.shape[0], 7, self.input_channels), device=mri.device, dtype=torch.bool
            ),
            "valid_band_fraction": zero,
        }
        if self.aligned_enabled:
            directional = aligned_high_losses(
                predicted_high,
                target.high,
                self.aligned_band_epsilon,
                smooth_l1_beta=self.aligned_smooth_l1_beta,
            )
        gain_loss = (
            gain_regularization(network_info["corrector_gains"])
            if self.corrector_enabled
            else zero
        )
        total = (
            total
            + float(weights.hotspot) * hot_components["hotspot"]
            + float(weights.underestimation) * hot_components["underestimation"]
            + float(weights.aligned_amplitude) * directional["aligned_amplitude"]
            + float(weights.orthogonal_error) * directional["orthogonal_error"]
            + self.gain_regularization_weight * gain_loss
        )
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite S1/S2 total loss")
        logs.update(
            {
                "loss_total": total.detach(),
                "loss_hotspot": hot_components["hotspot"].detach(),
                "loss_hotspot_underestimation": hot_components["underestimation"].detach(),
                "hotspot_overestimation": hot_components["overestimation"].detach(),
                "hotspot_valid_case_fraction": hot_components["valid_case_fraction"].detach(),
                "loss_aligned_amplitude": directional["aligned_amplitude"].detach(),
                "loss_orthogonal_error": directional["orthogonal_error"].detach(),
                "aligned_valid_band_fraction": directional["valid_band_fraction"].detach(),
                "loss_gain_regularization": gain_loss.detach(),
                "lambda_hotspot_actual": torch.tensor(weights.hotspot, device=mri.device),
                "lambda_underestimation_actual": torch.tensor(
                    weights.underestimation, device=mri.device
                ),
                "lambda_aligned_actual": torch.tensor(
                    weights.aligned_amplitude, device=mri.device
                ),
                "lambda_orthogonal_actual": torch.tensor(
                    weights.orthogonal_error, device=mri.device
                ),
            }
        )
        for key, value in hotspot_metadata.items():
            finite = value[torch.isfinite(value)]
            logs[key] = (
                finite.mean().detach()
                if finite.numel()
                else torch.tensor(0.0, device=mri.device)
            )
        for band in range(7):
            valid_band = directional["valid"][:, band]
            valid_float = valid_band.float()
            denominator = valid_float.sum().clamp_min(1.0)
            logs[f"high_band_{band}_aligned_amplitude_beta"] = (
                directional["beta"][:, band] * valid_float
            ).sum().detach() / denominator
            logs[f"high_band_{band}_orthogonal_error_ratio"] = (
                directional["orthogonal_ratio"][:, band] * valid_float
            ).sum().detach() / denominator
            logs[f"high_band_{band}_aligned_valid_fraction"] = valid_float.mean().detach()
        gains = network_info["corrector_gains"]
        if gains is not None:
            for band in range(7):
                logs[f"corrector_gain_band_{band}"] = gains[:, band].mean().detach()
                logs[f"corrector_energy_multiplier_band_{band}"] = (
                    gains[:, band].square().mean().detach()
                )
        low_metrics = residual_metrics(predicted_low, target.low)
        logs.update({f"low_residual_{key}": value.detach() for key, value in low_metrics.items()})
        logs.update({key: value.detach() for key, value in high_band_metrics(predicted_high, target.high).items()})
        m = self.bridge.bridge_time(t, state_low).flatten()
        logs.update(
            {
                "low_residual_gt_mean": target.low.detach().mean(),
                "low_residual_gt_std": target.low.detach().float().std(unbiased=False),
                "low_residual_pred_mean": predicted_low.detach().mean(),
                "low_residual_pred_std": predicted_low.detach().float().std(unbiased=False),
                "endpoint_sample_ratio": (t == self.bridge.num_timesteps).float().mean(),
                "t_mean": t.float().mean(),
                "t_std": t.float().std(unbiased=False),
                "t_min": t.min(),
                "t_max": t.max(),
                "m_t_mean": m.mean(),
                "m_t_std": m.std(unbiased=False),
            }
        )
        return {
            "loss": total,
            "logs": logs,
            "loss_components": {
                **components,
                "hotspot": hot_components["hotspot"],
                "underestimation": hot_components["underestimation"],
                "aligned_amplitude": directional["aligned_amplitude"],
                "orthogonal_error": directional["orthogonal_error"],
                "gain_regularization": gain_loss,
            },
            "B_raw": raw_pet,
            "B_views": self.projection.all_views(raw_pet),
            "predicted_low_residual": predicted_low,
            "predicted_high_residual": predicted_high,
            "target_low_residual": target.low,
            "target_high_residual": target.high,
            "state_low": state_low,
            "state_high": state_high,
            "t": t,
            "t_low": t,
            "t_high": t,
            "noise_low": epsilon_low,
            "noise_high": epsilon_high,
            **network_info,
        }

    forward = forward_train

    @torch.no_grad()
    def infer(
        self,
        mri: Tensor,
        covariance_low: float | Tensor,
        covariance_high: float | Tensor,
        *,
        num_steps: int = 15,
        stochastic: bool = False,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ) -> dict[str, Any]:
        """MRI-only inference.  The signature deliberately has no PET/GT argument."""
        mri_low, mri_high, meta = self.dwt(mri)
        state_low = torch.zeros_like(mri_low)
        state_high = torch.zeros_like(mri_high)
        steps = self.bridge.sampling_timesteps(num_steps, mri.device)
        trajectory: list[dict[str, Tensor | int]] | None = [] if return_trajectory else None
        predicted_low = state_low
        predicted_high = state_high
        final_network_info: dict[str, Any] = {}
        for current, nxt in zip(steps[:-1], steps[1:]):
            t = current.repeat(mri.shape[0])
            predicted_low, predicted_high, final_network_info = self._predict_residual_endpoints(
                state_low,
                state_high,
                mri_low,
                mri_high,
                t,
                covariance_low,
                covariance_high,
            )
            state_low = self.bridge.sample_step(
                state_low,
                predicted_low,
                current,
                nxt,
                covariance_low,
                stochastic=stochastic,
                generator=generator,
            )
            state_high = self.bridge.sample_step(
                state_high,
                predicted_high,
                current,
                nxt,
                covariance_high,
                stochastic=stochastic,
                generator=generator,
            )
            if trajectory is not None:
                trajectory.append(
                    {
                        "t_current": int(current.item()),
                        "t_next": int(nxt.item()),
                        "low_state": state_low.clone(),
                        "high_state": state_high.clone(),
                    }
                )
        pet_low, pet_high = add_residuals(mri_low, mri_high, state_low, state_high)
        raw_pet = self.idwt(pet_low, pet_high, meta)
        return {
            **self.projection.all_views(raw_pet),
            "predicted_low_residual": state_low,
            "predicted_high_residual": state_high,
            "high_pred_raw": final_network_info.get("high_pred_raw", state_high),
            "high_pred_corrected": final_network_info.get(
                "high_pred_corrected", state_high
            ),
            "corrector_gains": final_network_info.get("corrector_gains"),
            "sampling_timesteps": [int(v) for v in steps.cpu().tolist()],
            "stochastic": bool(stochastic),
            "trajectory": trajectory,
        }
