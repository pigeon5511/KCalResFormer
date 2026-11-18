# KCalResFormer

[中文说明](docs/README_zh.md)

KCalResFormer combines calibrated exact Kronecker LMMSE reconstruction with a shared CNN–Transformer residual for OFDM channel estimation on a fixed 120 × 14 grid. Calibration, neural training, model selection and residual scaling use source DM-RS layouts A/C. The fitted model is reused on layouts B/D without retraining.

This repository is a source reorganization of the shared model used by the current manuscript. The neural branch has **61,741 trainable parameters**. The exact solver is exact for the assumed separable covariance; it is not an oracle channel model.

**Checkpoint status:** the three paper checkpoints were not found among the available local files. Their frozen SHA256 hashes, calibration coefficients and alpha are recorded in `configs/checkpoint_manifest.json`. Early development and layout-specialist weights are not substituted. The runnable example uses a randomly initialized neural branch and does not reproduce paper accuracy.

## Directory layout

```text
KCalResFormer/
├── models/                 # Physical front end, residual network and public model API
├── configs/                # Model, training, dataset protocol and frozen checkpoint hashes
│   └── SOURCE_MANIFEST.json # Source-file integrity hashes
├── scripts/                # Data generation/preparation, training, checkpoint import and evaluation
│   ├── matlab/             # MATLAB + 5G Toolbox channel/waveform generator
│   └── _training/          # Adapted formal shared A/C training implementation
├── examples/               # Synthetic input and a runnable interface example
├── checkpoints/            # Place verified, converted model checkpoints here
├── docs/                   # Data contract, generation guide and reproduction details
│   └── README_zh.md         # Chinese guide
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation and quick check

Use Python 3.10 or later; Python 3.12 is used for local verification. Install the PyTorch build suitable for your CPU/CUDA environment, then run from this repository's root:

```bash
python -m pip install -r requirements.txt
python examples/quickstart.py
```

Expected output: 61,741 parameters and a complex64 tensor with shape `(4, 120, 14)`, containing A/B/C/D views of one synthetic channel. The output projection is initially zero, so this first random model returns its physical estimate. This is an interface check, not a trained-model benchmark.

## Inference

Create a new output file for a synthetic interface check:

```bash
python scripts/infer.py --input examples/synthetic_pilots.npz --output runs/example_prediction.npy --random-init
```

After restoring a formal shared checkpoint from the original training server, verify and convert it:

```bash
python scripts/import_checkpoint.py --input /path/to/formal/best.pt --seed 2026102301 --output checkpoints/kcalresformer_seed_2026102301.pt
python scripts/infer.py --input examples/synthetic_pilots.npz --output runs/trained_prediction.npy --checkpoint checkpoints/kcalresformer_seed_2026102301.pt --device cpu
```

The importer checks the original file's SHA256 before loading it. There are three expected seeds: `2026102301`, `2026102302` and `2026102303`. Each seed is a separately trained model, not a model ensemble.

Python API:

```python
import numpy as np
import torch
from models import load_model
from scripts.data_io import select_input

model = load_model('checkpoints/kcalresformer_seed_2026102301.pt', device='cpu')
with np.load('examples/synthetic_pilots.npz', allow_pickle=False) as archive:
    inputs = select_input(dict(archive), slice(None))
with torch.inference_mode():
    estimate = model(inputs)
```

The physical solve uses CPU complex128 and returns complex64 grids. The residual can run on CPU or CUDA. This repository does not establish all-GPU latency. See [the input and data contract](docs/DATA.md).

## Generate your own source dataset

Data generation requires **MATLAB and 5G Toolbox**; a MATLAB license and toolbox installation are supplied by the user. Training, inference and evaluation use Python/PyTorch. No MATLAB Engine Python package is required. The generation pipeline has been checked locally with MATLAB R2021b and 5G Toolbox 2.3.

First check the source-data plan, then generate and audit a small dataset:

```bash
python scripts/generate_data.py plan
python scripts/generate_data.py all --smoke --output-root data/smoke
python scripts/train.py train --smoke --generated-data --device cpu --data-root data/smoke --output-root runs/smoke
```

MATLAB is found on PATH, through `MATLAB_EXECUTABLE`, or through `--matlab /path/to/matlab`. The smoke dataset has 6 calibration, 8 training and 10 validation channels (6 selection + 4 alpha); it verifies the workflow and does not reproduce formal accuracy.

Generate the full 115,000-channel source dataset and train your own weights:

```bash
python scripts/generate_data.py all --output-root data/formal_dmrs --workers 1
python scripts/train.py train --generated-data --device cuda --data-root data/formal_dmrs --output-root runs/formal
python scripts/import_checkpoint.py --fit-root runs/formal --seed 2026102301 --output checkpoints/retrained_seed_2026102301.pt
```

The generator keeps the original SISO TDL-A, OFDM, DM-RS, QPSK, SNR/delay/Doppler schedule and seed streams. It checks geometry, static-channel reconstruction, repeated seeds, stored arrays, noise, disjoint streams and shard hashes. `all` performs setup → MATLAB smoke → freeze → generation → replay → audit. Repeat the same command, or use `resume`, to validate and skip committed shards. Use a dedicated output directory; existing datasets are not overwritten. Full source MAT shards use roughly 6.6 GB before training caches and logs; allow at least 20 GB for a full workflow.

See [the complete data-generation guide](docs/DATA_GENERATION.md), including staged commands, two-worker execution and recovery behavior. An independently frozen final-test generator is outside this source-data package.

## Prepare existing MAT data

MAT data already generated by the scripts, or existing formal data, can be exported into the NPZ interface:

```bash
python scripts/prepare_data.py --data-root /path/to/formal_dmrs --split validation --start-channel 1 --channels 4 --layouts A B C D --output runs/validation_subset.npz
```

Use A/C only for calibration, neural training, model selection and alpha fitting. B/D observations from an existing development set do not establish a new independent test. The independent final set requires its own recorded provenance.

## Training

Review the frozen recipe without starting training:

```bash
python scripts/train.py plan
```

Run an isolated CPU smoke workflow on a small subset of the existing formal source dataset:

```bash
python scripts/train.py train --smoke --device cpu --data-root /path/to/formal_dmrs --output-root runs/smoke
```

Run the full source-only recipe, which audits the declared dataset, calibrates the exact physical front end, caches its outputs, trains three shared models and fits one pooled source alpha:

```bash
python scripts/train.py train --device cuda --data-root /path/to/formal_dmrs --output-root runs/formal
python scripts/import_checkpoint.py --fit-root runs/formal --seed 2026102301 --output checkpoints/retrained_seed_2026102301.pt
```

Training uses complex MSE, batch 512, Adam, exponential learning-rate decay, a 500-epoch cap and patience 10. The source pools contain 100,000 training channels, 5,000 physical-calibration channels, 8,000 model-selection channels and 2,000 alpha-fitting channels. Training checkpoints retain optimizer, scheduler, RNG and consumed-batch state for resumption with the same command and environment. Formal training requires CUDA; the smoke option runs one epoch with one seed on small source subsets.

The entry trains **only the shared exact-K model**. Changing the source layout, grid, recipe or environment defines a new experiment. After editing files, run `python scripts/update_manifest.py` to refresh `configs/SOURCE_MANIFEST.json` before starting a new output directory. Original server checkpoints are imported for inference; their training-resume identities do not belong to this reorganized workflow.

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/kcalresformer_seed_2026102301.pt --input runs/validation_subset.npz --output runs/validation_evaluation
```

The evaluator reports complex MSE, mean per-observation linear NMSE and `10*log10(mean NMSE)`. When received symbols, reference bits and the common data mask are present, it also reports uncoded Gray-QPSK BER. Results are descriptive for the supplied subset. The lightweight evaluator does not reproduce the paper's paired bootstrap, full comparison tables or latency study. NMSE improvements do not establish BER improvements.

## Scope and attribution

See [reproduction notes](docs/REPRODUCIBILITY.md) and [release status](docs/RELEASE_STATUS.md). The separate legacy80 benchmark, early demonstration weights, two-pass experiments and later layout specialists are outside this shared-model package.

This project is licensed under the [MIT License](LICENSE), copyright (c) 2026 Yusheng Zhang. Add the paper link and citation when those publication details are finalized.

## Third-party dependencies

PyTorch, NumPy and SciPy are external dependencies installed separately. Their respective licenses continue to apply; this repository does not distribute their implementations.

Dataset generation calls MATLAB and 5G Toolbox, which must be installed and licensed separately by the user. This repository contains a simulation script that calls their APIs and does not distribute MATLAB or toolbox implementations.

Full datasets and trained checkpoints are not bundled.
