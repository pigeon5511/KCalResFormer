"""CPU complex128 Cartesian Kronecker solves with a bounded factor cache."""
from collections import OrderedDict
import hashlib
import math

import numpy as np
from scipy.linalg import eigh
from scipy.special import j0


class CartesianKroneckerFrontend:
    """Exact or two-pass separable-covariance estimator, not an oracle prior.

    Inputs must form full Cartesian products. Noise is one homogeneous white
    LS variance per observation; unequal per-pilot variance is not supported.
    Neither covariance hyperparameters nor the CPU solve are differentiable.
    """

    def __init__(self, mode, coefficients, cascade=(1.0, 1.0), cache_size=512):
        if mode not in ('exact', 'twopass'):
            raise ValueError('mode must be exact or twopass')
        if (len(coefficients) != 3 or any(not math.isfinite(x) for x in coefficients)
                or coefficients[0] < 0 or any(x <= 0 for x in coefficients[1:])):
            raise ValueError('nonnegative noise scale and positive delay/Doppler scales required')
        if len(cascade) != 2 or any(not math.isfinite(x) or x <= 0 for x in cascade):
            raise ValueError('two finite positive cascade coefficients required')
        if not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError('cache_size must be a nonnegative integer')
        self.mode = mode
        self.coefficients = tuple(coefficients)
        self.cascade = tuple(cascade)
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def clear_cache(self):
        self.cache.clear()

    @staticmethod
    def cartesian(coordinates, name):
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0 or not np.isfinite(coords).all():
            raise ValueError(f'{name} must be nonempty finite [points,2] coordinates')
        f, fi = np.unique(coords[:, 0], return_inverse=True)
        t, ti = np.unique(coords[:, 1], return_inverse=True)
        order = fi * len(t) + ti
        if len(coords) != len(f) * len(t) or len(np.unique(order)) != len(coords):
            raise ValueError(f'{name} must be a complete Cartesian product without duplicates')
        return f, t, order

    def factors(self, pilot_coordinates, target_coordinates, delay, doppler):
        pc = np.asarray(pilot_coordinates, dtype=np.float64)
        qc = np.asarray(target_coordinates, dtype=np.float64)
        key = (delay, doppler, self.coefficients[1:], pc.shape, qc.shape,
               hashlib.sha256(pc.tobytes() + qc.tobytes()).digest())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        fp, tp, po = self.cartesian(pc, 'pilot_coordinates')
        fq, tq, qo = self.cartesian(qc, 'target_coordinates')
        tau = delay * self.coefficients[1]
        fd = doppler * self.coefficients[2]
        rf = 1 / (1 + 2j * np.pi * tau * (fp[:, None] - fp[None, :]))
        rt = j0(2 * np.pi * fd * (tp[:, None] - tp[None, :]))
        rfp = 1 / (1 + 2j * np.pi * tau * (fq[:, None] - fp[None, :]))
        rtp = j0(2 * np.pi * fd * (tq[:, None] - tp[None, :]))
        ef, uf = eigh((rf + rf.conj().T) / 2, check_finite=False)
        et, ut = eigh((rt + rt.T) / 2, check_finite=False)
        if min(ef.min(), et.min()) < -1e-8:
            raise ValueError('non-positive covariance beyond numerical tolerance')
        factors = dict(ef=ef, et=et, uf=uf, ut=ut, af=rfp @ uf, at=rtp @ ut,
                       po=po, qo=qo, nf=len(fp), nt=len(tp))
        if self.cache_size:
            self.cache[key] = factors
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return factors

    def predict(self, values, pilot_coordinates, target_coordinates, *,
                noise_variance, rms_delay_spread_s, max_doppler_hz):
        params = (noise_variance, rms_delay_spread_s, max_doppler_hz)
        if any(not np.isscalar(x) or not math.isfinite(x) or x < 0 for x in params):
            raise ValueError('noise variance, delay and Doppler must be finite nonnegative scalars')
        y = np.asarray(values, dtype=np.complex128)
        if y.ndim != 1 or not np.isfinite(y).all() or len(y) != len(pilot_coordinates):
            raise ValueError('values must be one finite complex LS value per pilot coordinate')
        f = self.factors(pilot_coordinates, target_coordinates, rms_delay_spread_s, max_doppler_hz)
        eta = max(self.coefficients[0] * noise_variance, 1e-10) + 1e-6
        canonical = np.empty(len(y), dtype=np.complex128)
        canonical[f['po']] = y
        yp = canonical.reshape(f['nf'], f['nt'])
        projected = f['uf'].conj().T @ yp @ f['ut'].conj()
        if self.mode == 'exact':
            denominator = f['ef'][:, None] * f['et'][None, :] + eta
        else:
            cf, ct = self.cascade
            denominator = ((f['ef'][:, None] + max(cf * eta, 1e-10))
                           * (f['et'][None, :] + max(ct * eta, 1e-10)))
        grid = f['af'] @ (projected / denominator) @ f['at'].T
        return grid.reshape(-1)[f['qo']].astype(np.complex64)
