"""Estimate complex channel grids from pilot observations in an NPZ file."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from models import create_model, load_model
from scripts.data_io import select_input


def predict(model, payload, batch_size=16):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(payload['pilot_values']), batch_size):
            batch = select_input(payload, slice(start, start+batch_size))
            predictions.append(model(batch).cpu().numpy())
    if not predictions:
        raise ValueError('No observations')
    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--random-init', action='store_true', help='Interface check only; not trained inference')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/model.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', default=16, type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.output.exists():
        raise FileExistsError(args.output)
    model = (create_model(args.config).to(args.device).eval() if args.random_init
             else load_model(args.checkpoint, device=args.device))
    with np.load(args.input, allow_pickle=False) as archive:
        prediction = predict(model, dict(archive), args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.save(stream, prediction, allow_pickle=False)
    print(dict(shape=list(prediction.shape), dtype=str(prediction.dtype),
               model_status='RANDOM_INITIALIZATION_INTERFACE_CHECK' if args.random_init else 'LOADED_CHECKPOINT'))


if __name__ == '__main__':
    main()
