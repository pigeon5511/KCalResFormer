"""Run the formal shared exact-K recipe, with source-only calibration and alpha fitting."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'check', 'prepare', 'train', 'status'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--generated-data', action='store_true', help='Use a dataset created and audited by scripts/generate_data.py')
    parser.add_argument('--output-root', type=Path, default=Path('runs/formal'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true', help='One seed, one epoch, isolated small source subsets')
    args = parser.parse_args()
    if args.stage not in ('plan', 'status') and args.data_root is None:
        parser.error('--data-root is required')
    sys.path.insert(0, str(ROOT/'scripts/_training'))
    import runner
    sys.argv = [str(Path(__file__).resolve()), args.stage, '--output-root', str(args.output_root.resolve()),
                '--device', args.device]
    if args.data_root is not None:
        sys.argv += ['--data-root', str(args.data_root.resolve())]
    if args.smoke:
        sys.argv.append('--smoke')
    if args.generated_data:
        sys.argv.append('--generated-data')
    runner.main()


if __name__ == '__main__':
    main()
