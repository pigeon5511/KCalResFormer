"""Lazy source A/C views matching PhysicalMatDataset; separate BER payload API."""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path
import sys
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

CONFIG_ROOT = Path(__file__).resolve().parents[2] / 'configs'


def make_grid_coordinates(frequencies, times):
    frequency, time = torch.meshgrid(frequencies, times, indexing='ij')
    return torch.stack((frequency.reshape(-1), time.reshape(-1)), dim=-1)


class FormalDMRSDataset(Dataset):
    """One observation per index, two paired views per independent channel.

    split: calibration/train/validation_selection/validation_alpha/validation.
    max_channels: train-only balanced nested prefix (10000,30000,100000).
    No B/D source observation API; no full Y/bits in training dictionaries.
    """
    def __init__(self, split='train', max_channels=None, cache_shards=1, data_root=None):
        if data_root is None:
            raise ValueError('data_root is required')
        self.root = Path(data_root)
        self.protocol = json.loads((self.root/'protocol.json').read_text(encoding='utf-8-sig'))
        partition = self.protocol.get('validation_partition',dict(selection_channels=8000,alpha_channels=2000))
        self.role = split
        if split == 'validation_selection':
            self.split, self.start, self.channels = 'validation', 0, partition['selection_channels']
        elif split == 'validation_alpha':
            self.split, self.start, self.channels = 'validation', partition['selection_channels'], partition['alpha_channels']
        else:
            if split not in self.protocol['splits']:
                raise ValueError(f'Unknown split: {split}')
            self.split, self.start = split, 0
            self.channels = self.protocol['splits'][split]['channels']
        if max_channels is not None:
            if split != 'train' or max_channels not in (10000, 30000, 100000):
                raise ValueError('Only frozen nested train prefixes 10000/30000/100000 are supported')
            self.channels = max_channels
        if cache_shards < 1:
            raise ValueError('cache_shards must be >=1')
        self.cache_limit = cache_shards
        self.cache = OrderedDict()
        geometry_path = self.root/'geometry.mat' if 'generation_mode' in self.protocol else CONFIG_ROOT/'geometry.mat'
        self.geometry = sio.loadmat(geometry_path)
        self.coordinates = make_grid_coordinates(
            torch.tensor(self.geometry['subcarrier_frequencies_hz'].reshape(-1), dtype=torch.float32),
            torch.tensor(self.geometry['symbol_times_s'].reshape(-1), dtype=torch.float32))

    def __len__(self):
        return 2*self.channels

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = OrderedDict()
        return state

    def _channel(self, channel_index):
        if not 0 <= channel_index < self.channels:
            raise IndexError(channel_index)
        absolute = self.start + channel_index
        shard, within = divmod(absolute, self.protocol['shard_size'])
        if shard not in self.cache:
            path = self.root/self.split/f'shard_{shard+1:04d}.mat'
            if not Path(str(path)+'.complete.json').exists():
                raise RuntimeError(f'Shard not committed: {path}')
            self.cache[shard] = sio.loadmat(path)
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return self.cache[shard], within, absolute+1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        channel, view = divmod(index, 2)
        m, k, rid = self._channel(channel)
        layout_index = (0,2)[view]
        layout = ('A','C')[view]
        mask = self.geometry['pilot_masks'][:,:,layout_index].astype(bool)
        # Copies keep outputs safe after a cached shard is evicted or a caller mutates a tensor.
        truth = torch.tensor(m['H_true'][:,:,k].copy(), dtype=torch.complex64).reshape(-1)
        ls = np.zeros((120,14), np.complex64)
        ls[mask] = m['Y'][:,:,k][mask] / m['X'][:,:,k][mask]
        flat = torch.tensor(mask.reshape(-1), dtype=torch.bool)
        ls_flat = torch.from_numpy(ls.reshape(-1))
        scalar = lambda key: float(m[key].reshape(-1)[k])
        return dict(pilot_values=ls_flat[flat], pilot_coordinates=self.coordinates[flat],
            pilot_valid_mask=torch.ones(int(flat.sum()), dtype=torch.bool),
            target_values=truth, target_coordinates=self.coordinates.clone(),
            target_valid_mask=torch.ones(1680, dtype=torch.bool), ls_grid_values=ls_flat,
            pilot_grid_mask=flat, noise_variance=torch.tensor(scalar('noise_variance'),dtype=torch.float32),
            rms_delay_spread_s=torch.tensor(scalar('delay_spread_ns')*1e-9,dtype=torch.float32),
            max_doppler_hz=torch.tensor(scalar('max_doppler_hz'),dtype=torch.float32),
            snr_db=torch.tensor(scalar('snr_db'),dtype=torch.float32),
            grid_shape=torch.tensor([120,14],dtype=torch.long), pilot_layout_id=layout,
            layout_family='NR_PDSCH_mappingA_type2_single',dataset_version=self.protocol['dataset_version'],
            split_name=self.role,file_name=f'{self.split}_{rid:06d}_{layout}')

    def ber_payload(self, channel_index):
        """Evaluation-only raw waveform grids and reference bits for one channel."""
        m,k,rid = self._channel(channel_index)
        return dict(H_true=m['H_true'][:,:,k].copy(), X=m['X'][:,:,k].copy(),
                    Y=m['Y'][:,:,k].copy(),Y_noiseless=m['Y_noiseless'][:,:,k].copy(),
                    tx_bits=m['tx_bits'][:,:,:,k].copy(),
                    data_mask=self.geometry['data_mask'].astype(bool),channel_id=rid,
                    target_energy=float(m['target_energy'].reshape(-1)[k]))


def channel_metrics(prediction, truth):
    """Per-observation true complex MSE/NMSE; inputs [...,120,14] or [...,1680]."""
    pred = np.asarray(prediction, dtype=np.complex128)
    target = np.asarray(truth, dtype=np.complex128)
    if pred.shape != target.shape:
        raise ValueError('Prediction and truth shapes differ')
    if pred.shape[-2:] == (120,14):
        axes = (-2,-1)
    elif pred.shape[-1:] == (1680,):
        axes = (-1,)
    else:
        raise ValueError('Expected trailing grid [120,14] or [1680]')
    if not (np.isfinite(pred).all() and np.isfinite(target).all()):
        raise ValueError('Nonfinite prediction or truth')
    mse = np.mean(np.abs(pred-target)**2, axis=axes, dtype=np.float64)
    energy = np.mean(np.abs(target)**2, axis=axes, dtype=np.float64)
    if np.any(energy <= 0):
        raise ValueError('Zero-energy channel')
    return dict(mse=mse,nmse=mse/energy,target_energy=energy)


def qpsk_bit_errors(y, h_hat, tx_bits, data_mask, noise_variance=0.0):
    """Uncoded hard QPSK bit errors; variance=0 ZF, positive scalar variance MMSE.

    Return integer counts to aggregate across channels. No decoding/BLER.
    """
    y=np.asarray(y,np.complex128); h=np.asarray(h_hat,np.complex128)
    bits=np.asarray(tx_bits); mask=np.asarray(data_mask,bool)
    if y.shape!=(120,14) or h.shape!=y.shape or bits.shape!=(120,14,2) or mask.shape!=y.shape:
        raise ValueError('BER payload shape mismatch')
    if not(np.isfinite(y).all() and np.isfinite(h).all()) or noise_variance<0:
        raise ValueError('Invalid equalizer input')
    denominator=np.maximum(np.abs(h)**2+noise_variance,np.finfo(np.float64).tiny)
    equalized=y*h.conj()/denominator
    decisions=np.stack((equalized.real<0,equalized.imag<0),axis=-1)
    return dict(bit_errors=int(np.count_nonzero(decisions[mask]!=bits[mask])),
                bit_count=int(2*np.count_nonzero(mask)))
