"""Check a randomly initialized model; load a checkpoint for trained inference."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models import create_model
from scripts.data_io import select_input

torch.set_num_threads(2)
with np.load(ROOT/'examples/synthetic_pilots.npz', allow_pickle=False) as archive:
    inputs = select_input(dict(archive), slice(None))
model = create_model(ROOT/'configs/model.json').eval()
with torch.inference_mode():
    estimate = model(inputs)
print('Random initialization: interface check only; not the paper result.')
print('Parameters:', sum(parameter.numel() for parameter in model.parameters()))
print('Output:', tuple(estimate.shape), estimate.dtype)
