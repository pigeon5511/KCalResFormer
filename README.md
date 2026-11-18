# KCalResFormer: Calibrated Kronecker LMMSE with a CNN–Transformer Residual

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[中文说明](docs/README_zh.md)

Implementation of KCalResFormer for SISO OFDM channel estimation with multiple DM-RS layouts.

## Overview

KCalResFormer combines a calibrated Kronecker LMMSE estimator with a shared CNN–Transformer residual network. The physical estimator reconstructs the channel from pilot observations, and the neural network learns a correction to the reconstructed grid. The model uses nominal SNR, RMS delay spread and maximum Doppler shift as side information.

The implementation operates on a fixed **120 × 14** resource grid and contains **61,741 trainable neural parameters**. Calibration and training use DM-RS layouts A/C; the fitted model is reused on layouts B/D without retraining.

This repository provides model code, a MATLAB dataset generator, and training and inference scripts. Pretrained weights are not bundled.

## Architecture

The estimator has three components:

1. **Calibrated K-LMMSE front end:** reconstructs the channel using an exact Kronecker solve for the assumed separable covariance.
2. **CNN–Transformer residual network:** combines local convolutional processing and attention to predict a channel correction.
3. **Residual fusion:** scales the neural correction and adds it to the physical estimate.

$$
\widehat{H} = \widehat{H}_{\mathrm{K\text{-}LMMSE}} + \alpha\,\Delta_{\mathrm{NN}}.
$$

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/pigeon5511/KCalResFormer.git
cd KCalResFormer
python -m pip install -r requirements.txt
```

Requirements:

- **Python 3.10 or later**, with PyTorch, NumPy and SciPy.
- **MATLAB and 5G Toolbox** for dataset generation.
- **CUDA-enabled PyTorch** for full training. The small workflow check and inference can run on CPU.

Install the PyTorch build appropriate for your CUDA environment before starting GPU training. The MATLAB generator has been checked with MATLAB R2021b and 5G Toolbox 2.3; Python MATLAB Engine is not required.

## Quick Start

Check the installation using the included synthetic pilot observations:

```bash
python examples/quickstart.py
```

The example prints the parameter count and a complex channel tensor with shape `(4, 120, 14)`. It checks the model interface without trained weights.

## Project Structure

```text
KCalResFormer/
├── models/              # Physical estimator and CNN–Transformer residual
├── configs/             # Model, training and dataset configurations
├── scripts/
│   ├── matlab/          # MATLAB channel and OFDM data generator
│   └── _training/       # Training pipeline
├── examples/            # Synthetic input and minimal example
├── checkpoints/         # Exported model weights
├── docs/                # Chinese guide and detailed data documentation
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Run the following commands from the repository root.

## Data Generation

Generate the training, calibration and validation data:

```bash
python scripts/generate_data.py all --output-root data/formal_dmrs
```

The generator uses SISO TDL-A channels, a 120 × 14 OFDM grid, 15 kHz subcarrier spacing and QPSK symbols.

| Split | Independent channels |
|---|---:|
| Training | 100,000 |
| Physical calibration | 5,000 |
| Validation | 10,000 |

For a small workflow check, use `--smoke` with a separate output directory:

```bash
python scripts/generate_data.py all --smoke --output-root data/smoke
```

MATLAB can be selected through `MATLAB_EXECUTABLE` or the `--matlab` argument. See the [data-generation guide](docs/DATA_GENERATION.md) for staged execution and resuming interrupted generation.

## Training

Train the shared KCalResFormer model on the generated data:

```bash
python scripts/train.py train --generated-data --device cuda --data-root data/formal_dmrs --output-root runs/formal
```

Training settings are defined in `configs/training.json`, including the batch size, optimizer, learning rate and early stopping. Model selection and residual-scale fitting use separate portions of the validation split and only A/C observations. To resume an interrupted run, repeat the same command with the same data and output directories.

After training, export a model for inference:

```bash
python scripts/import_checkpoint.py --fit-root runs/formal --seed 2026102301 --output checkpoints/kcalresformer.pt
```

The export includes the trained network, calibrated physical parameters and fitted residual scale.

To check the training workflow on CPU using the small dataset:

```bash
python scripts/train.py train --smoke --generated-data --device cpu --data-root data/smoke --output-root runs/smoke
```

## Inference and Evaluation

The following example prepares a small validation subset, estimates its channels and computes the error metrics:

```bash
python scripts/prepare_data.py --data-root data/formal_dmrs --split validation --channels 4 --layouts A B C D --output runs/validation_subset.npz
python scripts/infer.py --checkpoint checkpoints/kcalresformer.pt --input runs/validation_subset.npz --output runs/channel_estimates.npy --device cpu
python scripts/evaluate.py --checkpoint checkpoints/kcalresformer.pt --input runs/validation_subset.npz --output runs/evaluation --device cpu
```

The evaluator reports complex MSE and NMSE, with per-layout summaries. It also reports uncoded QPSK BER when the required received symbols, reference bits and data mask are available. Use a separate held-out test set for reporting test performance.

The inference output is a complex channel tensor with shape `(batch, 120, 14)`. Input fields and units are described in the [data format guide](docs/DATA.md).

## Configuration

| File | Purpose |
|---|---|
| `configs/model.json` | Model architecture and inference defaults |
| `configs/training.json` | Training, calibration and model-selection settings |
| `configs/dataset_protocol.json` | Channel conditions and dataset sizes |
| `configs/geometry.mat` | Resource-grid coordinates and pilot layouts |

After editing repository files, refresh the source manifest before starting a new training run:

```bash
python scripts/update_manifest.py
```

## Third-party Dependencies

PyTorch, NumPy and SciPy are installed separately and retain their respective licenses. MATLAB and 5G Toolbox must also be installed and licensed separately for data generation. Their implementations are not distributed with this repository.

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Yusheng Zhang.
