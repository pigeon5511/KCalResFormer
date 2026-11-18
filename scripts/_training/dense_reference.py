"""Physics-driven LMMSE estimation for variable OFDM grids and pilot layouts.

The estimator works on physical frequency/time coordinates rather than fixed
tensor dimensions.  A batch may therefore contain different pilot counts by
padding the pilot tensors and marking valid entries with ``pilot_valid_mask``.
Grid sizes can vary between batches; mixed grid sizes in one batch additionally
require padding the target coordinates and using ``target_valid_mask``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from structured_error_lmmse import FrozenStructuredErrorBank


class _DifferentiableBesselJ0(torch.autograd.Function):
    """Bessel J0 with its analytic first derivative for autograd.

    PyTorch 2.13 evaluates ``torch.special.bessel_j0`` but does not attach an
    autograd function.  The identity dJ0(x)/dx = -J1(x) supplies the exact
    first-order derivative needed to learn a Doppler parameter upstream.
    """

    @staticmethod
    def forward(ctx: object, argument: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(argument)
        return torch.special.bessel_j0(argument)

    @staticmethod
    def backward(ctx: object, output_gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (argument,) = ctx.saved_tensors
        return (output_gradient * -torch.special.bessel_j1(argument),)


def _bessel_j0(argument: torch.Tensor) -> torch.Tensor:
    return _DifferentiableBesselJ0.apply(argument)


def make_grid_coordinates(
    subcarrier_frequencies_hz: torch.Tensor,
    symbol_times_s: torch.Tensor,
) -> torch.Tensor:
    """Return flattened ``(frequency_hz, time_s)`` coordinates for an OFDM grid.

    The order is subcarrier-major and then symbol-major, matching
    ``channel.reshape(-1)`` for a channel stored as ``[num_scs, num_symbols]``.
    Explicit coordinate vectors are required because actual OFDM symbol times
    can include non-uniform cyclic-prefix durations.
    """

    if subcarrier_frequencies_hz.ndim != 1 or symbol_times_s.ndim != 1:
        raise ValueError("frequency and time coordinate tensors must be one-dimensional")
    if subcarrier_frequencies_hz.numel() == 0 or symbol_times_s.numel() == 0:
        raise ValueError("frequency and time coordinate tensors must be non-empty")
    if subcarrier_frequencies_hz.device != symbol_times_s.device:
        raise ValueError("frequency and time coordinate tensors must use the same device")

    dtype = torch.promote_types(
        subcarrier_frequencies_hz.dtype, symbol_times_s.dtype
    )
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()

    frequencies = subcarrier_frequencies_hz.to(dtype=dtype)
    times = symbol_times_s.to(dtype=dtype)
    frequency_grid, time_grid = torch.meshgrid(frequencies, times, indexing="ij")
    return torch.stack((frequency_grid.reshape(-1), time_grid.reshape(-1)), dim=-1)


class VariablePilotLMMSE(nn.Module):
    r"""LMMSE estimator with separable frequency/time channel covariance.

    The frequency correlation is derived from an exponential power-delay
    profile,

    .. math:: R_f(\Delta f) = (1 + j2\pi\tau_{rms}\Delta f)^{-1},

    and the time correlation uses the isotropic Jakes model,

    .. math:: R_t(\Delta t) = J_0(2\pi f_D\Delta t).

    These assumptions are approximations for the current TDL-A data and must be
    validated empirically before being treated as an accurate channel model.
    """

    def __init__(
        self,
        *,
        diagonal_jitter: float = 1e-6,
        minimum_noise_variance: float = 1e-10,
        effective_noise_variance_enabled: bool = False,
        effective_variance_coefficient: float = 0.0,
        effective_variance_time_scale_s: Optional[float] = None,
        effective_variance_doppler_exponent: float = 2.0,
        effective_variance_delay_reference_s: float = 100e-9,
        effective_variance_delay_exponent: float = 0.0,
        structured_error_bank: Optional[FrozenStructuredErrorBank] = None,
    ) -> None:
        super().__init__()
        if diagonal_jitter < 0:
            raise ValueError("diagonal_jitter must be non-negative")
        if minimum_noise_variance < 0:
            raise ValueError("minimum_noise_variance must be non-negative")
        if effective_variance_coefficient < 0:
            raise ValueError("effective_variance_coefficient must be non-negative")
        if effective_variance_doppler_exponent <= 0:
            raise ValueError("effective_variance_doppler_exponent must be positive")
        if effective_variance_delay_reference_s <= 0:
            raise ValueError("effective_variance_delay_reference_s must be positive")
        if effective_variance_delay_exponent < 0:
            raise ValueError("effective_variance_delay_exponent must be non-negative")
        if effective_variance_time_scale_s is not None:
            if effective_variance_time_scale_s <= 0:
                raise ValueError("effective_variance_time_scale_s must be positive")
        elif effective_noise_variance_enabled:
            raise ValueError(
                "effective_variance_time_scale_s is required when effective "
                "noise variance is enabled"
            )
        self.diagonal_jitter = float(diagonal_jitter)
        self.minimum_noise_variance = float(minimum_noise_variance)
        self.effective_noise_variance_enabled = bool(
            effective_noise_variance_enabled
        )
        self.effective_variance_coefficient = float(
            effective_variance_coefficient
        )
        self.effective_variance_time_scale_s = (
            None
            if effective_variance_time_scale_s is None
            else float(effective_variance_time_scale_s)
        )
        self.effective_variance_doppler_exponent = float(
            effective_variance_doppler_exponent
        )
        self.effective_variance_delay_reference_s = float(
            effective_variance_delay_reference_s
        )
        self.effective_variance_delay_exponent = float(
            effective_variance_delay_exponent
        )
        self.structured_error_bank = structured_error_bank

    def compute_effective_noise_variance(
        self,
        noise_variance: torch.Tensor,
        rms_delay_spread_s: torch.Tensor,
        max_doppler_hz: torch.Tensor,
        channel_variance: torch.Tensor,
    ) -> torch.Tensor:
        r"""Add a condition-dependent modeling-error floor to ``var_hat``.

        When enabled, the additional variance is

        .. math::
            \sigma_{dynamic}^2 = P_H C(f_D T)^p
            (1 + \tau_{rms}/\tau_0)^q.

        ``T`` is a configured physical time scale (normally the OFDM symbol
        period), while ``P_H`` is ``channel_variance``.  Keeping the correction
        separate from ``noise_variance`` preserves the original ``var_hat``
        estimate.  Disabling the feature returns the input tensor directly so
        that the legacy numerical path remains unchanged.
        """

        if not self.effective_noise_variance_enabled:
            return noise_variance

        if self.effective_variance_time_scale_s is None:
            raise RuntimeError("effective variance time scale was not configured")

        normalized_doppler = (
            max_doppler_hz * self.effective_variance_time_scale_s
        )
        delay_factor = (
            1.0
            + rms_delay_spread_s / self.effective_variance_delay_reference_s
        )
        dynamic_variance = (
            channel_variance
            * self.effective_variance_coefficient
            * normalized_doppler.pow(self.effective_variance_doppler_exponent)
            * delay_factor.pow(self.effective_variance_delay_exponent)
        )
        return noise_variance + dynamic_variance

    @staticmethod
    def _as_batched_coordinates(
        coordinates: torch.Tensor,
        batch_size: int,
        name: str,
    ) -> torch.Tensor:
        if coordinates.ndim == 2:
            if coordinates.shape[-1] != 2:
                raise ValueError(f"{name} must have a final dimension of size 2")
            coordinates = coordinates.unsqueeze(0).expand(batch_size, -1, -1)
        elif coordinates.ndim == 3:
            if coordinates.shape[0] != batch_size or coordinates.shape[-1] != 2:
                raise ValueError(
                    f"{name} must have shape [batch, points, 2] or [points, 2]"
                )
        else:
            raise ValueError(
                f"{name} must have shape [batch, points, 2] or [points, 2]"
            )
        if not coordinates.dtype.is_floating_point:
            raise TypeError(f"{name} must use a real floating-point dtype")
        return coordinates

    @staticmethod
    def _as_batch_parameter(
        value: torch.Tensor | float,
        batch_size: int,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch_size)
        elif tensor.ndim == 2 and tensor.shape == (batch_size, 1):
            tensor = tensor[:, 0]
        elif tensor.ndim != 1 or tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must be scalar or contain one value per batch item")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor

    @staticmethod
    def _kernel(
        target_coordinates: torch.Tensor,
        source_coordinates: torch.Tensor,
        rms_delay_spread_s: torch.Tensor,
        max_doppler_hz: torch.Tensor,
        channel_variance: torch.Tensor,
        complex_dtype: torch.dtype,
    ) -> torch.Tensor:
        delta_frequency = (
            target_coordinates[:, 0].unsqueeze(1)
            - source_coordinates[:, 0].unsqueeze(0)
        )
        delta_time = (
            target_coordinates[:, 1].unsqueeze(1)
            - source_coordinates[:, 1].unsqueeze(0)
        )

        frequency_argument = (
            2.0 * math.pi * rms_delay_spread_s * delta_frequency
        )
        frequency_correlation = 1.0 / (
            1.0 + 1j * frequency_argument.to(dtype=complex_dtype)
        )
        time_correlation = _bessel_j0(
            2.0 * math.pi * max_doppler_hz * delta_time
        )
        return (
            channel_variance.to(dtype=complex_dtype)
            * frequency_correlation
            * time_correlation.to(dtype=complex_dtype)
        )

    def forward(
        self,
        pilot_values: torch.Tensor,
        pilot_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        *,
        noise_variance: torch.Tensor | float,
        rms_delay_spread_s: torch.Tensor | float,
        max_doppler_hz: torch.Tensor | float,
        pilot_valid_mask: Optional[torch.Tensor] = None,
        target_valid_mask: Optional[torch.Tensor] = None,
        channel_variance: torch.Tensor | float = 1.0,
        return_pilot_estimate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Estimate channels at arbitrary target coordinates.

        Args:
            pilot_values: Complex tensor ``[batch, max_pilots]`` or ``[max_pilots]``.
            pilot_coordinates: Real physical coordinates in Hz and seconds with
                shape ``[batch, max_pilots, 2]`` or ``[max_pilots, 2]``.
            target_coordinates: Real physical coordinates in Hz and seconds with
                shape ``[batch, max_targets, 2]`` or ``[max_targets, 2]``.
            noise_variance: LS-estimate noise variance in channel units.
            rms_delay_spread_s: RMS delay spread in seconds, not nanoseconds.
            max_doppler_hz: Maximum Doppler shift in hertz.
            pilot_valid_mask: Boolean tensor selecting valid padded pilots.
            target_valid_mask: Boolean tensor selecting valid padded targets.
            channel_variance: Prior channel power. Defaults to one.
            return_pilot_estimate: Also return the estimate at valid pilot
                coordinates. The analytic path reuses the target-domain solve.

        Returns:
            Complex estimates with shape ``[batch, max_targets]``. When
            ``return_pilot_estimate`` is true, returns ``(target_estimate,
            pilot_estimate)``. Unbatched inputs omit the batch dimension and
            padded targets or pilots are filled with zero.
        """

        if return_pilot_estimate and self.structured_error_bank is not None:
            # The frozen structured bank only applies when target coordinates
            # match its fixed full grid. A second pilot-target call therefore
            # has different fallback semantics and cannot safely share the
            # structured solve. Preserve that experimental path exactly.
            target_estimate = self(
                pilot_values,
                pilot_coordinates,
                target_coordinates,
                noise_variance=noise_variance,
                rms_delay_spread_s=rms_delay_spread_s,
                max_doppler_hz=max_doppler_hz,
                pilot_valid_mask=pilot_valid_mask,
                target_valid_mask=target_valid_mask,
                channel_variance=channel_variance,
            )
            pilot_estimate = self(
                pilot_values,
                pilot_coordinates,
                pilot_coordinates,
                noise_variance=noise_variance,
                rms_delay_spread_s=rms_delay_spread_s,
                max_doppler_hz=max_doppler_hz,
                pilot_valid_mask=pilot_valid_mask,
                target_valid_mask=pilot_valid_mask,
                channel_variance=channel_variance,
            )
            return target_estimate, pilot_estimate

        input_was_unbatched = pilot_values.ndim == 1
        if input_was_unbatched:
            pilot_values = pilot_values.unsqueeze(0)
        if pilot_values.ndim != 2:
            raise ValueError("pilot_values must have shape [batch, pilots] or [pilots]")
        if not pilot_values.is_complex():
            raise TypeError("pilot_values must use a complex dtype")

        batch_size, max_pilots = pilot_values.shape
        real_dtype = pilot_values.real.dtype
        device = pilot_values.device
        pilot_coordinates = self._as_batched_coordinates(
            pilot_coordinates, batch_size, "pilot_coordinates"
        ).to(device=device, dtype=real_dtype)
        target_coordinates = self._as_batched_coordinates(
            target_coordinates, batch_size, "target_coordinates"
        ).to(device=device, dtype=real_dtype)
        if pilot_coordinates.shape[1] != max_pilots:
            raise ValueError("pilot_values and pilot_coordinates disagree on pilot count")

        max_targets = target_coordinates.shape[1]
        if pilot_valid_mask is None:
            pilot_valid_mask = torch.ones(
                batch_size, max_pilots, device=device, dtype=torch.bool
            )
        else:
            if pilot_valid_mask.ndim == 1 and input_was_unbatched:
                pilot_valid_mask = pilot_valid_mask.unsqueeze(0)
            if pilot_valid_mask.shape != (batch_size, max_pilots):
                raise ValueError("pilot_valid_mask must match pilot_values")
            pilot_valid_mask = pilot_valid_mask.to(device=device, dtype=torch.bool)

        if target_valid_mask is None:
            target_valid_mask = torch.ones(
                batch_size, max_targets, device=device, dtype=torch.bool
            )
        else:
            if target_valid_mask.ndim == 1 and input_was_unbatched:
                target_valid_mask = target_valid_mask.unsqueeze(0)
            if target_valid_mask.shape != (batch_size, max_targets):
                raise ValueError("target_valid_mask must match target_coordinates")
            target_valid_mask = target_valid_mask.to(device=device, dtype=torch.bool)

        noise_variance = self._as_batch_parameter(
            noise_variance,
            batch_size,
            name="noise_variance",
            device=device,
            dtype=real_dtype,
        )
        rms_delay_spread_s = self._as_batch_parameter(
            rms_delay_spread_s,
            batch_size,
            name="rms_delay_spread_s",
            device=device,
            dtype=real_dtype,
        )
        max_doppler_hz = self._as_batch_parameter(
            max_doppler_hz,
            batch_size,
            name="max_doppler_hz",
            device=device,
            dtype=real_dtype,
        )
        channel_variance = self._as_batch_parameter(
            channel_variance,
            batch_size,
            name="channel_variance",
            device=device,
            dtype=real_dtype,
        )
        if (noise_variance < 0).any():
            raise ValueError("noise_variance must be non-negative")
        if (rms_delay_spread_s < 0).any():
            raise ValueError("rms_delay_spread_s must be non-negative")
        if (max_doppler_hz < 0).any():
            raise ValueError("max_doppler_hz must be non-negative")
        if (channel_variance <= 0).any():
            raise ValueError("channel_variance must be positive")

        effective_noise_variance = self.compute_effective_noise_variance(
            noise_variance,
            rms_delay_spread_s,
            max_doppler_hz,
            channel_variance,
        )

        output = torch.zeros(
            batch_size,
            max_targets,
            device=device,
            dtype=pilot_values.dtype,
        )
        pilot_output = (
            torch.zeros(
                batch_size,
                max_pilots,
                device=device,
                dtype=pilot_values.dtype,
            )
            if return_pilot_estimate
            else None
        )
        for batch_index in range(batch_size):
            valid_pilots = pilot_valid_mask[batch_index]
            valid_targets = target_valid_mask[batch_index]
            if not valid_pilots.any():
                raise ValueError("every batch item must contain at least one valid pilot")
            if not valid_targets.any() and pilot_output is None:
                continue

            pilot_coords = pilot_coordinates[batch_index, valid_pilots]
            target_coords = target_coordinates[batch_index, valid_targets]
            values = pilot_values[batch_index, valid_pilots]

            covariance_pp = self._kernel(
                pilot_coords,
                pilot_coords,
                rms_delay_spread_s[batch_index],
                max_doppler_hz[batch_index],
                channel_variance[batch_index],
                pilot_values.dtype,
            )
            covariance_tp = self._kernel(
                target_coords,
                pilot_coords,
                rms_delay_spread_s[batch_index],
                max_doppler_hz[batch_index],
                channel_variance[batch_index],
                pilot_values.dtype,
            )

            if self.structured_error_bank is not None:
                blended_covariances = self.structured_error_bank.blend_covariances(
                    covariance_pp,
                    covariance_tp,
                    pilot_coords,
                    target_coords,
                    rms_delay_spread_s[batch_index],
                    max_doppler_hz[batch_index],
                    channel_variance[batch_index],
                )
                if blended_covariances is not None:
                    covariance_pp, covariance_tp = blended_covariances

            # Remove round-off asymmetry before solving the Hermitian system.
            covariance_pp = 0.5 * (
                covariance_pp + covariance_pp.transpose(-2, -1).conj()
            )
            regularization = torch.clamp(
                effective_noise_variance[batch_index],
                min=self.minimum_noise_variance,
            ) + self.diagonal_jitter
            covariance_pp = covariance_pp + regularization.to(
                pilot_values.dtype
            ) * torch.eye(
                covariance_pp.shape[0],
                device=device,
                dtype=pilot_values.dtype,
            )
            weights = torch.linalg.solve(covariance_pp, values.unsqueeze(-1))
            output[batch_index, valid_targets] = (
                covariance_tp @ weights
            ).squeeze(-1)

            if pilot_output is not None:
                covariance_pilot_target = self._kernel(
                    pilot_coords,
                    pilot_coords,
                    rms_delay_spread_s[batch_index],
                    max_doppler_hz[batch_index],
                    channel_variance[batch_index],
                    pilot_values.dtype,
                )
                pilot_output[batch_index, valid_pilots] = (
                    covariance_pilot_target @ weights
                ).squeeze(-1)

        resolved_output = output[0] if input_was_unbatched else output
        if pilot_output is None:
            return resolved_output
        resolved_pilot_output = (
            pilot_output[0] if input_was_unbatched else pilot_output
        )
        return resolved_output, resolved_pilot_output


__all__ = ["VariablePilotLMMSE", "make_grid_coordinates"]
