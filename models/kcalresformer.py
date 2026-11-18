"""KCalResFormer: calibrated exact K-LMMSE and a shared neural residual."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .frontend import CartesianKroneckerFrontend
from .residual import FixedGridComponentResidual

FORMAT_VERSION = 'kcalresformer-v1'
CONDITIONS = ('snr_db', 'rms_delay_spread_s', 'max_doppler_hz')
LABEL_FIELDS = {'target', 'target_values', 'H_true', 'H_ideal', 'truth', 'tx_bits'}


class _KroneckerResidual(nn.Module):
    model_name = None
    frontend_mode = None

    def __init__(self, config):
        super().__init__()
        config = deepcopy(config)
        if config['model_name'] != self.model_name or config['frontend_mode'] != self.frontend_mode:
            raise ValueError('configuration belongs to a different model variant')
        if config.get('format_version') != FORMAT_VERSION:
            raise ValueError('unsupported model configuration format')
        if config['component_kwargs']['grid_shape'] != [120, 14]:
            raise ValueError('these trained variants require the fixed 120x14 grid')
        alpha = config['alpha']
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be finite and in [0,1]')
        self.config = config
        self.branch = FixedGridComponentResidual(**config['component_kwargs'])
        self.frontend = CartesianKroneckerFrontend(self.frontend_mode, config['coefficients'],
                                                  config.get('cascade', [1, 1]))
        # Keep the original Python-double scalar convention; no float32 refit.
        self.alpha = float(alpha)
        self.checkpoint_metadata = {}

    @property
    def device(self):
        return next(self.branch.parameters()).device

    @staticmethod
    def _guard(batch):
        if LABEL_FIELDS.intersection(batch):
            raise ValueError('remove supervision and transmitted bits from model inputs')

    @staticmethod
    def _array(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _parameter(self, batch, name, size):
        value = torch.as_tensor(batch[name], device=self.device, dtype=torch.float32)
        if value.ndim == 0:
            value = value.expand(size)
        elif value.shape == (size, 1):
            value = value[:, 0]
        if value.shape != (size,) or not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must be finite scalar or contain one value per observation')
        return value

    def physical_grid(self, batch):
        """Padded mixed-layout pilots -> [B,120,14]; CPU frontend, no labels."""
        self._guard(batch)
        values = self._array(batch['pilot_values'])
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not np.iscomplexobj(values):
            raise ValueError('pilot_values must be complex [B,P] or [P] LS estimates')
        size, pilots = values.shape
        if size == 0:
            raise ValueError('empty observation batch')
        pc = self._array(batch['pilot_coordinates'])
        qc = self._array(batch['target_coordinates'])
        if pc.ndim == 2:
            pc = np.broadcast_to(pc, (size, *pc.shape))
        if qc.ndim == 2:
            qc = np.broadcast_to(qc, (size, *qc.shape))
        valid = self._array(batch.get('pilot_valid_mask', np.ones((size, pilots), dtype=bool)))
        if valid.ndim == 1 and size == 1:
            valid = valid[None, :]
        if pc.shape != (size, pilots, 2) or qc.shape != (size, 1680, 2) or valid.shape != values.shape or valid.dtype != bool:
            raise ValueError('incompatible pilot/coordinate/mask shapes; target grid requires 1680 points')
        if 'target_valid_mask' in batch and not self._array(batch['target_valid_mask']).all():
            raise ValueError('padded/partial target grids are unsupported by the neural residual')
        params = {n: self._parameter(batch, n, size).detach().cpu().numpy()
                  for n in ('noise_variance', 'rms_delay_spread_s', 'max_doppler_hz')}
        result = []
        for i in range(size):
            f, t, order = self.frontend.cartesian(qc[i], 'target_coordinates')
            if len(f) != 120 or len(t) != 14 or not np.array_equal(order, np.arange(1680)):
                raise ValueError('neural grid requires frequency-major ascending 120x14 coordinates')
            result.append(self.frontend.predict(values[i, valid[i]], pc[i, valid[i]], qc[i],
                          **{name: float(x[i]) for name, x in params.items()}).reshape(120, 14))
        return torch.from_numpy(np.stack(result)).to(self.device)

    def forward_grid(self, base, *, snr_db, rms_delay_spread_s, max_doppler_hz, mode='full'):
        """Trainable residual over grids from this variant's frozen frontend.

        mode='raw' uses alpha=1 for training; mode='physical' bypasses the NN;
        mode='full' uses the saved source-calibrated global alpha.
        """
        if mode not in ('full', 'physical', 'raw'):
            raise ValueError('mode must be full, physical or raw')
        base = torch.as_tensor(base, device=self.device)
        if base.ndim != 3 or tuple(base.shape[1:]) != (120, 14) or not base.is_complex():
            raise ValueError('base must have complex shape [B,120,14]')
        if mode == 'physical':
            return base
        delta = self.branch(base, torch.zeros_like(base), snr_db=snr_db,
                            rms_delay_spread_s=rms_delay_spread_s, max_doppler_hz=max_doppler_hz)
        raw = base + delta
        if mode == 'raw':
            return raw
        # Same complex64 rounding sequence as the original experiment.
        return base + self.alpha * (raw - base)

    def forward(self, batch, *, mode='full'):
        self._guard(batch)
        base = self.physical_grid(batch)
        params = {name: self._parameter(batch, name, len(base)) for name in CONDITIONS}
        return self.forward_grid(base, **params, mode=mode)

    forward_batch = forward

    def save_pretrained(self, path, *, metadata=None):
        """Save config, residual weights and global alpha in one typed checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = deepcopy(self.config)
        config['alpha'] = self.alpha
        config['weights_included'] = True
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError('alpha must be calibrated to [0,1] before export')
        torch.save(dict(format_version=FORMAT_VERSION, model_name=self.model_name,
                        config=config, state_dict=self.state_dict(),
                        metadata=deepcopy(self.checkpoint_metadata if metadata is None else metadata)), path)

    @classmethod
    def from_pretrained(cls, path, *, device='cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format_version') != FORMAT_VERSION or payload.get('model_name') != cls.model_name:
            raise ValueError('checkpoint belongs to a different model variant or format')
        model = cls(payload['config'])
        model.load_state_dict(payload['state_dict'], strict=True)
        model.checkpoint_metadata = payload.get('metadata', {})
        return model.to(device).eval()


class KCalResFormer(_KroneckerResidual):
    """Fixed 120x14 shared model; exact solve for the assumed covariance."""
    model_name = 'kcalresformer'
    frontend_mode = 'exact'


def create_model(config='configs/model.json'):
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding='utf-8-sig'))
    return KCalResFormer(config)


def load_model(path, *, device='cpu'):
    return KCalResFormer.from_pretrained(path, device=device)
