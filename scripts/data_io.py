"""Minimal formal-shard adapter and metrics, independent of experiment scripts."""
from pathlib import Path
import hashlib
import json

import numpy as np
import scipy.io as sio

INPUT_KEYS = ('pilot_values', 'pilot_coordinates', 'pilot_valid_mask', 'target_coordinates',
              'noise_variance', 'snr_db', 'rms_delay_spread_s', 'max_doppler_hz')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def make_coordinates(geometry):
    # Match the original float32 coordinate and condition contract exactly.
    f = geometry['subcarrier_frequencies_hz'].reshape(-1).astype(np.float32)
    t = geometry['symbol_times_s'].reshape(-1).astype(np.float32)
    ff, tt = np.meshgrid(f, t, indexing='ij')
    return np.stack((ff.reshape(-1), tt.reshape(-1)), axis=-1)


def select_input(payload, ids):
    """Only allow inference fields; labels/payloads remain outside model inputs."""
    n = len(payload['pilot_values'])
    out = {}
    for key in INPUT_KEYS:
        value = np.asarray(payload[key])
        if key in ('pilot_coordinates', 'target_coordinates') and value.ndim == 2:
            out[key] = value
        elif value.ndim == 0:
            out[key] = value
        elif value.shape[0] == n:
            out[key] = value[ids]
        else:
            raise ValueError(f'{key} does not match the observation count')
    return out


def load_formal(root, geometry_path, split, channel_ids, layouts):
    """Explicit one-based channel IDs; every loaded shard is hash-verified.

    Test data is read only when split='test' is explicitly requested and exists.
    The caller must establish test provenance; reading a test directory does not establish independence.
    """
    root = Path(root)
    protocol_path = root/'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8-sig')) if protocol_path.exists() else {}
    shard_size = protocol.get('shard_size',500)
    ids = list(channel_ids)
    if not ids or min(ids) < 1 or len(set(ids)) != len(ids):
        raise ValueError('channel IDs must be unique positive one-based integers')
    if not layouts or len(set(layouts)) != len(layouts) or any(x not in 'ABCD' for x in layouts):
        raise ValueError('layouts must be unique members of A B C D')
    geometry = sio.loadmat(geometry_path)
    coords = make_coordinates(geometry)
    if coords.shape != (1680, 2) or geometry['pilot_masks'].shape != (120, 14, 4):
        raise ValueError('geometry must match the fixed 120x14 four-layout contract')
    values = []; pc = []; valid = []; qc = []; noise = []; snr = []; delay = []; doppler = []
    truths = []; received = []; bits = []; rids = []; labels = []; streams = []
    hashes = {}; mats = {}
    max_pilots = max(int(geometry['pilot_masks'][:, :, 'ABCD'.index(x)].sum()) for x in layouts)
    for rid in ids:
        shard, k = divmod(rid - 1, shard_size)
        if shard not in mats:
            # One cached shard, to bound RAM when testing large datasets.
            path = root / split / f'shard_{shard + 1:04d}.mat'
            complete = json.loads(Path(str(path) + '.complete.json').read_text(encoding='utf-8-sig'))
            digest = sha256(path)
            if digest != complete['sha256']:
                raise ValueError(f'shard checksum mismatch: {path}')
            hashes[str(path)] = digest
            mats = {shard: sio.loadmat(path)}
        m = mats[shard]
        scalar = lambda key: float(m[key].reshape(-1)[k])
        for layout in layouts:
            mask = geometry['pilot_masks'][:, :, 'ABCD'.index(layout)].astype(bool)
            count = int(mask.sum())
            y = m['Y'][:, :, k][mask] / m['X'][:, :, k][mask]
            padded = np.zeros(max_pilots, np.complex64); padded[:count] = y
            padded_coords = np.zeros((max_pilots, 2), np.float32); padded_coords[:count] = coords[mask.reshape(-1)]
            values.append(padded); pc.append(padded_coords); valid.append(np.arange(max_pilots) < count)
            noise.append(scalar('noise_variance')); snr.append(scalar('snr_db'))
            delay.append(scalar('delay_spread_ns') * 1e-9); doppler.append(scalar('max_doppler_hz'))
            truths.append(m['H_true'][:, :, k]); received.append(m['Y'][:, :, k]); bits.append(m['tx_bits'][:, :, :, k])
            rids.append(rid); labels.append(layout)
            streams.append([int(m[name].reshape(-1)[k]) for name in ('channel_seed', 'data_seed', 'noise_seed')])
    return dict(pilot_values=np.array(values), pilot_coordinates=np.array(pc), pilot_valid_mask=np.array(valid),
                target_coordinates=coords, noise_variance=np.array(noise, np.float32), snr_db=np.array(snr, np.float32),
                rms_delay_spread_s=np.array(delay, np.float32), max_doppler_hz=np.array(doppler, np.float32),
                H_true=np.array(truths), Y=np.array(received), tx_bits=np.array(bits),
                data_mask=geometry['data_mask'].astype(bool), channel_ids=np.array(rids), layouts=np.array(labels),
                generation_stream_seeds=np.array(streams), source_hashes=hashes)


def metric_arrays(prediction, truth, received, bits, data_mask):
    pred = np.asarray(prediction, np.complex128)
    truth = np.asarray(truth, np.complex128)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[1:] != (120, 14):
        raise ValueError('predictions/truth must have identical [N,120,14] shapes')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('nonfinite channel estimate or truth')
    error = np.mean(abs(pred - truth) ** 2, axis=(1, 2))
    energy = np.mean(abs(truth) ** 2, axis=(1, 2))
    if np.any(energy <= 0):
        raise ValueError('zero-energy target channel')
    z = np.asarray(received, np.complex128) * pred.conj()
    decisions = np.stack((z.real < 0, z.imag < 0), axis=-1)
    errors = np.count_nonzero(decisions[:, data_mask, :] != bits[:, data_mask, :], axis=(1, 2))
    return dict(mse=error, nmse=error / energy, bit_errors=errors,
                bit_count=np.full(len(pred), int(2 * data_mask.sum()), dtype=np.int64))
