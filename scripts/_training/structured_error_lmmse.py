"""Frozen structured covariance correction for the physical LMMSE front end."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn


class FrozenStructuredErrorBank(nn.Module):
    """Read-only rank-8 covariance factors calibrated on a fixed pilot layout.

    Unsupported physical conditions, pilot layouts, target grids or channel
    variances return ``None`` so the caller can use the analytic LMMSE path.
    A missing or corrupted configured artifact raises during construction.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        bank_path: str | Path,
        *,
        expected_sha256: Optional[str] = None,
    ) -> None:
        super().__init__()
        path = Path(bank_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Structured-error bank not found: {path}")
        actual_sha256 = self._sha256(path)
        if expected_sha256 is not None:
            normalized_expected = expected_sha256.strip().upper()
            if actual_sha256 != normalized_expected:
                raise ValueError(
                    "Structured-error bank SHA256 mismatch: "
                    f"expected {normalized_expected}, got {actual_sha256}"
                )

        with np.load(path, allow_pickle=False) as bank:
            required = {
                "format_version",
                "candidate_name",
                "rank",
                "shrinkage",
                "delay_spread_ns",
                "max_doppler_hz",
                "pilot_coordinates",
                "target_coordinates",
                "pilot_bases",
                "pilot_eigenvalues",
                "target_factors",
            }
            missing = sorted(required - set(bank.files))
            if missing:
                raise ValueError(f"Structured-error bank is missing arrays: {missing}")
            format_version = int(bank["format_version"].item())
            candidate_name = str(bank["candidate_name"].item())
            rank = int(bank["rank"].item())
            shrinkage = float(bank["shrinkage"].item())
            delay_spread_ns = np.asarray(bank["delay_spread_ns"], dtype=np.float64)
            max_doppler_hz = np.asarray(bank["max_doppler_hz"], dtype=np.float64)
            pilot_coordinates = np.asarray(
                bank["pilot_coordinates"], dtype=np.float64
            )
            target_coordinates = np.asarray(
                bank["target_coordinates"], dtype=np.float64
            )
            pilot_bases = np.asarray(bank["pilot_bases"], dtype=np.complex128)
            pilot_eigenvalues = np.asarray(
                bank["pilot_eigenvalues"], dtype=np.float64
            )
            target_factors = np.asarray(
                bank["target_factors"], dtype=np.complex128
            )

        if format_version != self.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported structured-error bank format {format_version}"
            )
        if candidate_name != "generalized_rank8_a0.05":
            raise ValueError(f"Unexpected structured-error candidate: {candidate_name}")
        if rank <= 0 or not 0.0 <= shrinkage <= 1.0:
            raise ValueError("Invalid rank or shrinkage in structured-error bank")
        if pilot_coordinates.ndim != 2 or pilot_coordinates.shape[1] != 2:
            raise ValueError("pilot_coordinates must have shape [pilots, 2]")
        if target_coordinates.ndim != 2 or target_coordinates.shape[1] != 2:
            raise ValueError("target_coordinates must have shape [targets, 2]")
        condition_count = delay_spread_ns.size
        expected_shapes = {
            "max_doppler_hz": (condition_count,),
            "pilot_bases": (condition_count, pilot_coordinates.shape[0], rank),
            "pilot_eigenvalues": (condition_count, rank),
            "target_factors": (condition_count, target_coordinates.shape[0], rank),
        }
        actual_shapes = {
            "max_doppler_hz": max_doppler_hz.shape,
            "pilot_bases": pilot_bases.shape,
            "pilot_eigenvalues": pilot_eigenvalues.shape,
            "target_factors": target_factors.shape,
        }
        for name, expected_shape in expected_shapes.items():
            if actual_shapes[name] != expected_shape:
                raise ValueError(
                    f"{name} has shape {actual_shapes[name]}, expected {expected_shape}"
                )
        arrays = (
            delay_spread_ns,
            max_doppler_hz,
            pilot_coordinates,
            target_coordinates,
            pilot_bases,
            pilot_eigenvalues,
            target_factors,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Structured-error bank contains non-finite values")
        if (pilot_eigenvalues < 0).any():
            raise ValueError("Structured-error bank contains negative eigenvalues")

        self.bank_path = str(path)
        self.bank_sha256 = actual_sha256
        self.candidate_name = candidate_name
        self.rank = rank
        self.shrinkage = shrinkage
        self.register_buffer(
            "delay_spread_s",
            torch.as_tensor(delay_spread_ns * 1e-9, dtype=torch.float64),
        )
        self.register_buffer(
            "max_doppler_hz",
            torch.as_tensor(max_doppler_hz, dtype=torch.float64),
        )
        self.register_buffer(
            "pilot_coordinates",
            torch.as_tensor(pilot_coordinates, dtype=torch.float64),
        )
        self.register_buffer(
            "target_coordinates",
            torch.as_tensor(target_coordinates, dtype=torch.float64),
        )
        self.register_buffer(
            "pilot_bases",
            torch.as_tensor(pilot_bases, dtype=torch.complex128),
        )
        self.register_buffer(
            "pilot_eigenvalues",
            torch.as_tensor(pilot_eigenvalues, dtype=torch.float64),
        )
        self.register_buffer(
            "target_factors",
            torch.as_tensor(target_factors, dtype=torch.complex128),
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest().upper()

    @staticmethod
    def _coordinates_match(
        supplied: torch.Tensor,
        reference: torch.Tensor,
    ) -> bool:
        if supplied.shape != reference.shape:
            return False
        supplied = supplied.to(device=reference.device, dtype=torch.float64)
        frequency_matches = torch.all(
            torch.abs(supplied[:, 0] - reference[:, 0]) <= 0.5
        )
        time_matches = torch.all(
            torch.abs(supplied[:, 1] - reference[:, 1]) <= 1e-9
        )
        return bool(frequency_matches and time_matches)

    def condition_index(
        self,
        rms_delay_spread_s: torch.Tensor,
        max_doppler_hz: torch.Tensor,
    ) -> Optional[int]:
        delay = rms_delay_spread_s.detach().to(
            device=self.delay_spread_s.device, dtype=torch.float64
        )
        doppler = max_doppler_hz.detach().to(
            device=self.max_doppler_hz.device, dtype=torch.float64
        )
        matches = (
            (torch.abs(self.delay_spread_s - delay) <= 0.5e-9)
            & (torch.abs(self.max_doppler_hz - doppler) <= 0.5)
        )
        indices = torch.nonzero(matches, as_tuple=False).flatten()
        if indices.numel() == 0:
            return None
        if indices.numel() != 1:
            raise RuntimeError("Structured-error bank contains ambiguous conditions")
        return int(indices.item())

    def blend_covariances(
        self,
        analytic_pp: torch.Tensor,
        analytic_tp: torch.Tensor,
        pilot_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        rms_delay_spread_s: torch.Tensor,
        max_doppler_hz: torch.Tensor,
        channel_variance: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return frozen blended covariances, or ``None`` for baseline fallback."""

        condition = self.condition_index(rms_delay_spread_s, max_doppler_hz)
        if condition is None:
            return None
        if not bool(torch.isclose(channel_variance.detach(), torch.ones_like(channel_variance), rtol=0.0, atol=1e-6)):
            return None
        if not self._coordinates_match(pilot_coordinates, self.pilot_coordinates):
            return None
        if not self._coordinates_match(target_coordinates, self.target_coordinates):
            return None

        complex_dtype = analytic_pp.dtype
        device = analytic_pp.device
        basis = self.pilot_bases[condition].to(
            device=device, dtype=complex_dtype
        )
        eigenvalues = self.pilot_eigenvalues[condition].to(
            device=device, dtype=analytic_pp.real.dtype
        )
        target_factor = self.target_factors[condition].to(
            device=device, dtype=complex_dtype
        )
        empirical_pp = (basis * eigenvalues) @ basis.transpose(-2, -1).conj()
        empirical_tp = target_factor @ basis.transpose(-2, -1).conj()
        alpha = self.shrinkage
        return (
            (1.0 - alpha) * analytic_pp + alpha * empirical_pp,
            (1.0 - alpha) * analytic_tp + alpha * empirical_tp,
        )


__all__ = ["FrozenStructuredErrorBank"]
