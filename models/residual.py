"""Matched CNN/Transformer recipes for fixed-grid residual development.

All recipes use the same seven input maps, patch encoder, positional embedding,
output decoder and six pre-normalized blocks.  Only the token mixer changes.
The module is deliberately independent of the legacy estimator and its YAML
schema; a development runner supplies a frozen complex estimate and correction.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


class UnifiedMixerBlock(nn.Module):
    """One pre-norm recipe with identical common parameter names.

    A runner can pair common initialization across architectures by copying all
    state tensors except keys containing ``.mixer.``.  This keeps the encoder,
    positions, normalizations and feed-forward weights identical while allowing
    the inherently different token mixers to have their own initialization.
    """

    def __init__(
        self, mixer_kind: str, width: int, heads: int, hidden: int, ff_dim: int, dropout: float,
        token_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        if mixer_kind not in {"cnn", "transformer"}:
            raise ValueError("mixer_kind must be cnn or transformer")
        self.mixer_kind = mixer_kind
        self.token_shape = token_shape
        self.norm1 = nn.LayerNorm(width)
        if mixer_kind == "cnn":
            self.mixer = nn.Sequential(
                nn.Conv2d(width, hidden, kernel_size=3, padding=1),
                nn.GELU(), nn.Dropout(dropout),
                nn.Conv2d(hidden, width, kernel_size=1),
            )
        else:
            self.mixer = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ff_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ff_dim, width))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, _, width = tokens.shape
        mixed = self.norm1(tokens)
        if self.mixer_kind == "cnn":
            mixed = mixed.transpose(1, 2).reshape(batch_size, width, *self.token_shape)
            mixed = self.mixer(mixed).flatten(2).transpose(1, 2)
        else:
            mixed, _ = self.mixer(mixed, mixed, mixed, need_weights=False)
        tokens = tokens + self.dropout1(mixed)
        mixed = self.ffn(self.norm2(tokens))
        return tokens + self.dropout2(mixed)


class FixedGridComponentResidual(nn.Module):
    """Predict a complex correction with a matched CNN, Transformer or hybrid.

    ``base_grid`` is the frozen estimate (LMMSE plus a low-rank correction when
    enabled); ``low_rank_grid`` is that correction, or an all-zero map in the
    low-rank ablation.  Neither input includes the target channel.  Real and
    imaginary parts pass separately through the *same* network.

    The seven maps are the selected component of base and low-rank estimates,
    frequency and time coordinates spanning [-1, 1], SNR/30, delay_ns/300 and
    Doppler_Hz/1000.  These scales are fixed, with no data-fitted normalization
    or clipping.  Non-overlapping patches are ordered frequency first, then
    time; within a patch, channels precede frequency and time.

    At the defaults, trainable parameter counts are 61,684 (CNN), 61,798
    (Transformer) and 61,741 (hybrid).  Near-matched capacity does not imply
    equal computation or isolate every architectural difference.
    """

    INPUT_CHANNELS = 7

    def __init__(
        self,
        architecture: str = "hybrid",
        grid_shape: Sequence[int] = (120, 14),
        patch_shape: Sequence[int] = (3, 2),
        width: int = 32,
        depth: int = 6,
        heads: int = 4,
        ff_dim: int = 64,
        cnn_hidden: int = 13,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if architecture not in {"cnn", "transformer", "hybrid"}:
            raise ValueError("architecture must be cnn, transformer or hybrid")
        self.grid_shape = self._shape_pair(grid_shape, "grid_shape")
        self.patch_shape = self._shape_pair(patch_shape, "patch_shape")
        if any(size % patch for size, patch in zip(self.grid_shape, self.patch_shape)):
            raise ValueError("grid_shape must be divisible by patch_shape")
        for name, value in (("width", width), ("depth", depth), ("heads", heads),
                            ("ff_dim", ff_dim), ("cnn_hidden", cnn_hidden)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")

        self.architecture = architecture
        self.width = width
        self.depth = depth
        self.token_shape = tuple(size // patch for size, patch in
                                 zip(self.grid_shape, self.patch_shape))
        self.num_tokens = math.prod(self.token_shape)
        self.patch_area = math.prod(self.patch_shape)
        self.patch_encoder = nn.Linear(self.INPUT_CHANNELS * self.patch_area, width)
        self.position_embedding = nn.Parameter(torch.empty(1, self.num_tokens, width))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        blocks: list[nn.Module] = []
        for index in range(depth):
            use_cnn = architecture == "cnn" or (architecture == "hybrid" and index % 2 == 0)
            blocks.append(UnifiedMixerBlock(
                "cnn" if use_cnn else "transformer", width, heads, cnn_hidden,
                ff_dim, dropout, self.token_shape,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.output_projection = nn.Linear(width, self.patch_area)
        # Every recipe initially preserves the supplied baseline exactly.
        # The first decoder update subsequently unlocks gradients to the body.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        frequency = self._coordinate_axis(self.grid_shape[0])[:, None].expand(self.grid_shape)
        time = self._coordinate_axis(self.grid_shape[1])[None, :].expand(self.grid_shape)
        self.register_buffer("coordinate_maps", torch.stack((frequency, time)).unsqueeze(0))

    @staticmethod
    def _shape_pair(value: Sequence[int], name: str) -> tuple[int, int]:
        if len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, int)
                                  or item <= 0 for item in value):
            raise ValueError(f"{name} must contain two positive integers")
        return (value[0], value[1])

    @staticmethod
    def _coordinate_axis(length: int) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, length) if length > 1 else torch.zeros(1)

    @staticmethod
    def _condition_vector(
        value: torch.Tensor | float,
        *, batch_size: int, device: torch.device, dtype: torch.dtype, name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tensor.is_complex():
            raise TypeError(f"{name} must be real-valued")
        # Rebuild from the original value so Python floats/lists retain double
        # precision when the module and complex inputs have been converted.
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch_size)
        elif tensor.ndim == 2 and tensor.shape == (batch_size, 1):
            tensor = tensor[:, 0]
        elif tensor.ndim != 1 or tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must be scalar or contain one value per batch item")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")
        return tensor

    def _patchify(self, maps: torch.Tensor) -> torch.Tensor:
        """Map [B,C,H,W] to [B,frequency_tokens*time_tokens,C*patch_area]."""
        if maps.ndim != 4 or tuple(maps.shape[2:]) != self.grid_shape:
            raise ValueError("maps must have shape [batch, channels, *grid_shape]")
        batch_size, channels = maps.shape[:2]
        frequency_tokens, time_tokens = self.token_shape
        patch_frequency, patch_time = self.patch_shape
        return maps.reshape(batch_size, channels, frequency_tokens, patch_frequency,
                            time_tokens, patch_time).permute(0, 2, 4, 1, 3, 5).reshape(
                                batch_size, self.num_tokens, channels * self.patch_area)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Inverse of the single-channel patch ordering, returning [B,H,W]."""
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (self.num_tokens, self.patch_area):
            raise ValueError("patches must have shape [batch, num_tokens, patch_area]")
        batch_size = patches.shape[0]
        frequency_tokens, time_tokens = self.token_shape
        patch_frequency, patch_time = self.patch_shape
        return patches.reshape(batch_size, frequency_tokens, time_tokens,
                               patch_frequency, patch_time).permute(0, 1, 3, 2, 4).reshape(
                                   batch_size, *self.grid_shape)

    def _component_maps(
        self, base: torch.Tensor, low_rank: torch.Tensor, conditions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = base.shape[0]
        coordinates = self.coordinate_maps.to(device=base.device, dtype=base.dtype).expand(
            batch_size, -1, -1, -1)
        condition_maps = conditions[:, :, None, None].expand(batch_size, -1, *self.grid_shape)
        return torch.cat((base[:, None], low_rank[:, None], coordinates, condition_maps), dim=1)

    def _forward_component(
        self, base: torch.Tensor, low_rank: torch.Tensor, conditions: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.patch_encoder(self._patchify(self._component_maps(base, low_rank, conditions)))
        tokens = tokens + self.position_embedding
        for block in self.blocks:
            tokens = block(tokens)
        return self._unpatchify(self.output_projection(tokens))

    def forward(
        self,
        base_grid: torch.Tensor,
        low_rank_grid: torch.Tensor,
        *,
        snr_db: torch.Tensor | float,
        rms_delay_spread_s: torch.Tensor | float,
        max_doppler_hz: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return a complex residual with exactly the input grid's shape."""
        for name, tensor in (("base_grid", base_grid), ("low_rank_grid", low_rank_grid)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != self.grid_shape:
                raise ValueError(f"{name} must have shape [batch, {self.grid_shape[0]}, {self.grid_shape[1]}]")
            if not tensor.is_complex():
                raise TypeError(f"{name} must use a complex dtype")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain only finite values")
        if base_grid.shape != low_rank_grid.shape:
            raise ValueError("base_grid and low_rank_grid must have matching shapes")
        if base_grid.device != low_rank_grid.device or base_grid.dtype != low_rank_grid.dtype:
            raise ValueError("base_grid and low_rank_grid must have matching device and dtype")
        batch_size = base_grid.shape[0]
        if batch_size == 0:
            raise ValueError("the input batch must be non-empty")
        condition_args = dict(batch_size=batch_size, device=base_grid.device,
                              dtype=base_grid.real.dtype)
        snr = self._condition_vector(snr_db, name="snr_db", **condition_args) / 30.0
        delay = self._condition_vector(rms_delay_spread_s, name="rms_delay_spread_s",
                                       **condition_args) * (1e9 / 300.0)
        doppler = self._condition_vector(max_doppler_hz, name="max_doppler_hz",
                                         **condition_args) / 1000.0
        conditions = torch.stack((snr, delay, doppler), dim=1)
        if not bool(torch.isfinite(conditions).all()):
            raise ValueError("normalized conditions must contain only finite values")
        real = self._forward_component(base_grid.real, low_rank_grid.real, conditions)
        imag = self._forward_component(base_grid.imag, low_rank_grid.imag, conditions)
        return torch.complex(real, imag)


__all__ = ["FixedGridComponentResidual"]
