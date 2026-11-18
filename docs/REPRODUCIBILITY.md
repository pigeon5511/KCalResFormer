# Reproduction boundaries

The residual implementation in `models/residual.py` and Cartesian solver in `models/frontend.py` were copied byte-for-byte from the formal shared-model experiment. The public model wrapper is adapted from the existing standalone interface.

Training orchestration is adapted to run only `exact_klmmse_residual` on A/C. The original three seeds, architecture, physical coefficient search, complex-MSE objective, model-selection split, source-alpha fitting, Adam settings, batch sampler and resumable checkpoint logic are retained. Ada specialists, two-pass execution and comparison reporting are removed from this entry. This reorganization has a new source identity and is not a byte-identical server training package. Local smoke checks cannot establish a full formal rerun or the paper's accuracy.

The MATLAB generator retains the original channel-realization function, including OFDM timing, pilot symbols, seed streams, noise placement and stored fields. Its default 115,000-channel schedules match the original per-field arrays. Output directories and orchestration are portable; protocol timestamps, MATLAB versions and MAT headers can change file hashes. `--generated-data` records the new audited identity rather than accepting an unmatched file as the original dataset. Small smoke datasets have a smaller shard size and validation partition and must remain labeled as smoke experiments.

The paper's shared physical coefficients are `[0.03125, 0.5, 1.0]` for noise, delay and Doppler scaling. Its global source alpha is `0.888708221815786`. These values come from matching frozen calibration/alpha records. Creating a model from `configs/model.json` does not load neural weights. After retraining, use the newly fitted coefficients and alpha, which `scripts/import_checkpoint.py --fit-root ...` binds into the exported model.

The public frontend uses CPU complex128 with an eigensystem cache, casts the physical grid to complex64 and sends it to the residual's device. Full inference preserves the original complex64 rounding order `base + alpha * ((base + delta) - base)`. An all-zero map occupies the inherited second component-input slot; it is not an active low-rank branch. Real and imaginary parts pass through the same neural network separately.

Training/selection use complex MSE. Physical calibration and alpha fitting use per-observation NMSE. The descriptive evaluator averages linear NMSE before converting to dB. Neither validation-selected minima nor smoke metrics are independent test estimates. BER, if evaluated, uses the common data mask and uncoded Gray-QPSK sign decisions; NMSE gains alone do not imply BER gains.

The archived workflow specifies Python 3.12, CUDA PyTorch, NumPy and SciPy; it recommends retaining NumPy 1.26.4 when using PyTorch 2.2.2 to avoid the old PyTorch/NumPy 2 ABI mismatch. The complete runtime metadata of the missing three shared checkpoints has not been verified here. No exact reproducibility across software versions or GPUs is claimed.

`scripts/evaluate.py` is a portable single-model subset evaluator. Paper-level bootstrap intervals, baseline comparisons, three-seed aggregation and latency figures require the original experimental analysis, which is outside this package.
