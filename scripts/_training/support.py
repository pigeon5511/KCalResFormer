"""Portable, source-only training adapters; no external checkpoints."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[_name]='2'
import numpy as np
import scipy.linalg
import torch
from torch import nn
from torch.utils.data import Dataset,Sampler
HERE=Path(__file__).resolve().parent
PROJECT=HERE
REPO_ROOT=HERE.parents[1]
CONFIG_ROOT=REPO_ROOT/'configs'
ROOT=HERE
sys.path.insert(0,str(REPO_ROOT))
def module_from(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in value["cuda"]])



from data_loader import FormalDMRSDataset
from models.residual import FixedGridComponentResidual
from dense_reference import VariablePilotLMMSE
from models.frontend import CartesianKroneckerFrontend
K_ARMS=('exact_klmmse_residual',)
METHODS=K_ARMS
ACTIVE_LIMITS={}

def configure_limits(config):
    ACTIVE_LIMITS.clear()
    if config.get('smoke'):
        ACTIVE_LIMITS.update(train=8,calibration=6,validation_selection=6,validation_alpha=4)

def source_base(split,config):
    dataset=FormalDMRSDataset(split,data_root=config['data_root'],cache_shards=1)
    if split in ACTIVE_LIMITS:dataset.channels=min(dataset.channels,ACTIVE_LIMITS[split])
    return dataset

def mode_for(method):
    if method not in K_ARMS:raise ValueError('Unknown K model identity')
    return 'exact' if method==K_ARMS[0] else 'twopass'

def model_create(method,config,device):
    if method not in K_ARMS:
        raise ValueError('Only the shared exact-K KCalResFormer is included')
    return CalResFormer(config).to(device)


def inference(model,method,batch):
    if 'target' in batch or 'target_values' in batch:
        raise ValueError('Targets must not enter inference')
    if method not in K_ARMS:
        raise ValueError('Unknown model')
    return model(batch)


class KronFrontend:
    """Grouped complex128 implementation of the packaged Cartesian solver."""
    def __init__(self,mode):
        self.mode=mode
        self.kron=CartesianKroneckerFrontend(mode,[1,1,1],cache_size=512)
    def factors(self,sample,delay_scale,doppler_scale):
        self.kron.coefficients=(1,delay_scale,doppler_scale)
        return self.kron.factors(sample['pilot_coordinates'].numpy(),sample['target_coordinates'].numpy(),
             float(sample['rms_delay_spread_s']),float(sample['max_doppler_hz']))
    def modal(self,sample,values,coefficients):
        f=self.factors(sample,*coefficients[1:])
        y=np.asarray(values,np.complex128)
        if y.ndim==1:y=y[:,None]
        canonical=np.empty_like(y);canonical[f['po']]=y
        basis=np.kron(f['uf'],f['ut']);mapped=np.kron(f['af'],f['at'])[f['qo']]
        return f,basis.conj().T@canonical,mapped
    def denominator(self,f,variance,noise):
        eta=np.maximum(noise*np.asarray(variance,np.float64).reshape(-1),1e-10)+1e-6
        if self.mode=='exact':return (f['ef'][:,None]*f['et'][None,:]).reshape(-1,1)+eta
        return ((f['ef'][:,None,None]+eta)*(f['et'][None,:,None]+eta)).reshape(-1,len(eta))
    def predict(self,sample,values,variance,coefficients):
        f,z,mapped=self.modal(sample,values,coefficients)
        return (mapped@(z/self.denominator(f,variance,coefficients[0]))).T.astype(np.complex64)

class OrderSampler(Sampler):
    """Shard-local order with a trainer-owned consumed-batch cursor."""
    def __init__(self, dataset, batch_size, seed, shuffle=True):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.shuffle = shuffle
        self.epoch, self.start_batch = 0, 0
        self.total_batches = (len(dataset) + batch_size - 1) // batch_size

    def set_epoch(self, epoch, start_batch=0):
        if epoch < 0 or not 0 <= start_batch <= self.total_batches:
            raise ValueError("Invalid consumed batch cursor")
        self.epoch, self.start_batch = epoch, start_batch

    def __len__(self):
        return self.total_batches - self.start_batch

    def __iter__(self):
        ds = self.dataset
        if not self.shuffle:
            order = np.arange(len(ds))
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
            first, stop, size = ds.start, ds.start + ds.channels, ds.protocol["shard_size"]
            views = ds.views_per_channel
            chunks = []
            for shard in rng.permutation(np.arange(first // size, (stop - 1) // size + 1)):
                lo, hi = max(first, int(shard) * size), min(stop, (int(shard) + 1) * size)
                chunks.append(rng.permutation(np.arange(views * (lo-first), views * (hi-first))))
            order = np.concatenate(chunks)
        for batch in range(self.start_batch, self.total_batches):
            yield order[batch*self.batch_size:(batch+1)*self.batch_size].tolist()


class SourceDataset(Dataset):
    """Only A/C source pools; cached physical predictions contain no labels."""
    views_per_channel = 2

    def __init__(self, split, data_root, cache_root=None, feature=None):
        if split not in ("train", "calibration", "validation_selection", "validation_alpha"):
            raise ValueError("Final tests are inaccessible through the training dataset")
        self.base = FormalDMRSDataset(split, data_root=data_root, cache_shards=2)
        for name in ("protocol", "start", "channels", "role"):
            setattr(self, name, getattr(self.base, name))
        limits=ACTIVE_LIMITS
        if split in limits:
            self.channels=min(self.channels,limits[split]);self.base.channels=self.channels
        self.cache_root, self.feature = cache_root, feature
        self.arrays = OrderedDict()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        item = dict(target=sample["target_values"].reshape(120, 14),
                    snr_db=sample["snr_db"], rms_delay_spread_s=sample["rms_delay_spread_s"],
                    max_doppler_hz=sample["max_doppler_hz"], observation_id=sample["file_name"])
        if self.feature:
            absolute = self.start + index // 2
            shard, offset = divmod(absolute, self.protocol["shard_size"])
            if shard not in self.arrays:
                path = Path(self.cache_root) / self.base.split / f"shard_{shard+1:04d}_{self.feature}.npy"
                self.arrays[shard] = np.load(path, mmap_mode="r", allow_pickle=False)
                while len(self.arrays) > 2:
                    self.arrays.popitem(last=False)
            self.arrays.move_to_end(shard)
            item["input"] = torch.from_numpy(self.arrays[shard][2*offset+index%2].copy())
        return item


class CalResFormer(nn.Module):
    """Frozen calibrated front end is cached; no trainable low-rank branch."""
    def __init__(self, config):
        super().__init__()
        self.branch = FixedGridComponentResidual(**config["component_kwargs"])

    def forward(self, batch):
        if "target" in batch or "target_values" in batch:
            raise ValueError("Remove supervision before model inference")
        base = batch["input"]
        delta = self.branch(base, torch.zeros_like(base), **{name: batch[name] for name in
            ("snr_db", "rms_delay_spread_s", "max_doppler_hz")})
        return base + delta


class SpectralFrontend:
    """Same covariance/regularization as complex128 VariablePilotLMMSE.

    Cache eigensystems by geometry/delay/Doppler; fit/apply never infer a
    coefficient from the observation label. Nominal conditions are inputs.
    """
    def __init__(self):
        self.cache = OrderedDict()

    def factors(self, sample, delay_scale, doppler_scale):
        pc, qc = sample["pilot_coordinates"].double(), sample["target_coordinates"].double()
        key = (sample["pilot_layout_id"], float(sample["rms_delay_spread_s"]),
               float(sample["max_doppler_hz"]), delay_scale, doppler_scale,
               hashlib.sha256(pc.numpy().tobytes()+qc.numpy().tobytes()).hexdigest())
        if key not in self.cache:
            ds = sample["rms_delay_spread_s"].double()*delay_scale
            fd = sample["max_doppler_hz"].double()*doppler_scale
            pp = VariablePilotLMMSE._kernel(pc, pc, ds, fd, torch.tensor(1., dtype=torch.float64), torch.complex128).numpy()
            qp = VariablePilotLMMSE._kernel(qc, pc, ds, fd, torch.tensor(1., dtype=torch.float64), torch.complex128).numpy()
            eigenvalues, basis = scipy.linalg.eigh((pp+pp.conj().T)/2, check_finite=False)
            if eigenvalues.min() < -1e-8:
                raise ValueError("Invalid covariance eigenspectrum")
            self.cache[key] = (eigenvalues, basis, qp @ basis)
            while len(self.cache) > 512:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def predict(self, sample, values, noise_variance, coefficients):
        noise, delay, doppler = coefficients
        eigenvalues, basis, mapped = self.factors(sample, delay, doppler)
        y = np.asarray(values, dtype=np.complex128)
        if y.ndim == 1:
            y = y[:, None]
        variance = np.asarray(noise_variance, dtype=np.float64).reshape(-1)
        projected = basis.conj().T @ y
        denominator = eigenvalues[:, None] + (np.maximum(noise*variance, 1e-10)+1e-6)[None, :]
        return (mapped @ (projected/denominator)).T.astype(np.complex64)
