"""Export a declared subset of formal MAT shards to the portable NPZ interface."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.data_io import load_formal, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=['calibration', 'train', 'validation', 'test'])
    parser.add_argument('--start-channel', type=int, default=1, help='One-based channel ID')
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--layouts', nargs='+', default=list('ABCD'), choices=list('ABCD'))
    parser.add_argument('--geometry', type=Path, help='Defaults to the generated data geometry, then the bundled geometry')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.channels < 1 or args.start_channel < 1:
        parser.error('Positive channel count and one-based start ID are required')
    if args.output.exists():
        raise FileExistsError(args.output)
    geometry = args.geometry or (args.data_root/'geometry.mat' if (args.data_root/'geometry.mat').exists() else ROOT/'configs/geometry.mat')
    payload = load_formal(args.data_root, geometry, args.split,
                          range(args.start_channel, args.start_channel+args.channels), args.layouts)
    hashes = payload.pop('source_hashes')
    payload['split_name'] = np.full(len(payload['pilot_values']), args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, **payload)
    # Relative shard names avoid storing a workstation path in exported data.
    write_json(args.output.with_suffix('.source.json'), dict(split=args.split,
        channels=args.channels, layouts=args.layouts,
        source_sha256={Path(key).name:value for key,value in hashes.items()},
        scope='Caller-declared data subset; independence is not inferred from directory names'))
    print(dict(observations=len(payload['pilot_values']), output=str(args.output)))


if __name__ == '__main__':
    main()
