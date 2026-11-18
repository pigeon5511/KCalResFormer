"""Report complex MSE and per-observation NMSE for one loaded model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models import load_model
from scripts.data_io import metric_arrays, sha256, write_json
from scripts.infer import predict


def summarize(metrics, mask):
    mse = float(metrics['mse'][mask].mean())
    nmse = float(metrics['nmse'][mask].mean())
    # A perfect estimate has a mathematical dB value of -infinity; JSON uses null.
    result = dict(observations=int(mask.sum()), mse=mse, nmse=nmse,
                  nmse_db=float(10*np.log10(nmse)) if nmse > 0 else None)
    if 'bit_count' in metrics:
        result['uncoded_qpsk_ber'] = float(metrics['bit_errors'][mask].sum()/metrics['bit_count'][mask].sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='A new directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.checkpoint, device=args.device)
    with np.load(args.input, allow_pickle=False) as archive:
        payload = dict(archive)
    truth = np.asarray(payload['H_true'], np.complex128)
    prediction = predict(model, payload, args.batch_size).astype(np.complex128)
    if prediction.shape != truth.shape or not np.isfinite(truth).all():
        raise ValueError('Finite H_true with shape [N,120,14] is required')
    energy = np.mean(abs(truth)**2, axis=(1,2))
    if np.any(energy <= 0):
        raise ValueError('Target channel energy must be positive')
    mse = np.mean(abs(prediction-truth)**2, axis=(1,2))
    metrics = dict(mse=mse, nmse=mse/energy)
    if all(key in payload for key in ('Y','tx_bits','data_mask')):
        metrics = metric_arrays(prediction, truth, payload['Y'], payload['tx_bits'], payload['data_mask'])
    layouts = payload.get('layouts', np.full(len(truth), 'unspecified'))
    ids = payload.get('channel_ids', np.arange(len(truth)))
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json', dict(checkpoint_sha256=sha256(args.checkpoint),
        input_sha256=sha256(args.input), device=args.device, checkpoint_metadata=model.checkpoint_metadata,
        metrics='Mean complex MSE and mean per-observation linear NMSE; dB after averaging',
        scope='Descriptive subset evaluation; no new independent-test or uncertainty claim'))
    report = dict(overall=summarize(metrics, np.ones(len(truth), bool)),
                  by_layout={str(layout):summarize(metrics, layouts==layout) for layout in np.unique(layouts)})
    write_json(args.output/'summary.json', report)
    np.savez_compressed(args.output/'per_observation_metrics.npz', **metrics, layouts=layouts, channel_ids=ids)
    print(report)


if __name__ == '__main__':
    main()
