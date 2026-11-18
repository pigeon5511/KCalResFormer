"""Verify and convert a formal best.pt into a portable KCalResFormer checkpoint."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models import create_model
from scripts.data_io import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='A paper checkpoint listed in configs/checkpoint_manifest.json')
    source.add_argument('--fit-root', type=Path, help='Completed output of scripts/train.py')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    arm = 'exact_klmmse_residual'
    config = deepcopy(read(ROOT/'configs/model.json'))
    if args.fit_root:
        record = read(args.fit_root/'model_freeze.json')
        calibration_path = args.fit_root/'calibration.json'
        alpha_path = args.fit_root/'alpha.json'
        if sha256(calibration_path) != record['calibration_sha256'] or sha256(alpha_path) != record['alpha_sha256']:
            raise ValueError('Frozen calibration or alpha has changed')
        config['coefficients'] = read(calibration_path)['arms'][arm]['coefficients']
        config['alpha'] = record['alpha'][arm]
        binding = read(args.fit_root/'binding.json')
        config['component_kwargs'] = binding['identity']['configuration']['component_kwargs']
        status = record['status']
    else:
        record = read(ROOT/'configs/checkpoint_manifest.json')
        status = 'VERIFIED_PAPER_CHECKPOINT'
    rows = [row for row in record['runs'] if row['seed']==args.seed and row['method']==arm]
    if len(rows) != 1:
        raise ValueError('No unique matching shared exact-K seed in the freeze record')
    row = rows[0]
    path = args.fit_root/row['checkpoint'] if args.fit_root else args.input
    if sha256(path) != row['sha256']:
        raise ValueError('Checkpoint SHA256 does not match the frozen shared model; do not substitute early or specialist weights')
    if args.output.exists():
        raise FileExistsError(args.output)
    # Formal training payloads contain runtime metadata from older PyTorch.
    # Deserialize only after matching the complete file against the frozen hash.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['identity']['method'] != arm or payload['identity']['seed'] != args.seed:
        raise ValueError('Checkpoint model identity mismatch')
    model = create_model(config)
    model.load_state_dict(payload['model'], strict=True)
    model.save_pretrained(args.output, metadata=dict(seed=args.seed, source_checkpoint_sha256=row['sha256'],
        checkpoint_status=status, best_epoch=payload['epoch'], selection_metric=payload['selection_metric'],
        validation_selection_mse=float(payload['validation_mse']), fit_layouts=['A','C']))
    print(dict(output=str(args.output), seed=args.seed, status=status))


if __name__ == '__main__':
    main()
