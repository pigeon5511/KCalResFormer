"""Make synthetic pilot observations for interface checks; not a benchmark dataset."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy.io as sio
from scripts.data_io import make_coordinates


def make_example():
    geometry = sio.loadmat(ROOT/'configs/geometry.mat')
    coordinates = make_coordinates(geometry)
    rng = np.random.default_rng(20261118)
    frequency = coordinates[:,0].reshape(120,14)
    time = coordinates[:,1].reshape(120,14)
    truth = (np.exp(-2j*np.pi*frequency*65e-9) * np.exp(2j*np.pi*time*120)).astype(np.complex64)
    noise_variance = np.float32(0.01)
    noise = (np.sqrt(noise_variance/2)*(rng.normal(size=(120,14))+1j*rng.normal(size=(120,14)))).astype(np.complex64)
    values=[]; pilot_coordinates=[]; masks=[]
    for layout in range(4):
        mask = geometry['pilot_masks'][:,:,layout].astype(bool)
        count = int(mask.sum())
        padded = np.zeros(160, np.complex64)
        padded[:count] = (truth+noise)[mask]
        pc = np.zeros((160,2), np.float32)
        pc[:count] = coordinates[mask.reshape(-1)]
        values.append(padded); pilot_coordinates.append(pc); masks.append(np.arange(160)<count)
    return dict(pilot_values=np.array(values), pilot_coordinates=np.array(pilot_coordinates),
        pilot_valid_mask=np.array(masks), target_coordinates=coordinates,
        noise_variance=np.full(4,noise_variance,np.float32), snr_db=np.full(4,20,np.float32),
        rms_delay_spread_s=np.full(4,65e-9,np.float32), max_doppler_hz=np.full(4,120,np.float32),
        H_true=np.stack([truth]*4), layouts=np.array(list('ABCD')), channel_ids=np.ones(4,np.int64),
        dataset_scope=np.array('SYNTHETIC_INTERFACE_EXAMPLE_NOT_TDL_BENCHMARK'))


if __name__ == '__main__':
    target = Path(__file__).resolve().parent/'synthetic_pilots.npz'
    if target.exists():
        raise FileExistsError(target)
    np.savez_compressed(target, **make_example())
    print(target)
