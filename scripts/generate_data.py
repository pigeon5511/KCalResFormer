"""Generate source MAT shards using MATLAB and 5G Toolbox, then audit them."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dataset_generation import (audit,check_freeze,execution_lock,freeze,matlab_executable,
                                      protocol_for,read,run_matlab,setup,sha)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','setup','smoke','freeze','generate','resume','replay','audit','all'])
    parser.add_argument('--output-root',type=Path,default=Path('data/formal_dmrs'))
    parser.add_argument('--matlab',help='MATLAB executable; otherwise MATLAB_EXECUTABLE or PATH')
    parser.add_argument('--workers',type=int,choices=[1,2],default=1,help='Independent MATLAB processes; no Parallel Computing Toolbox needed')
    parser.add_argument('--smoke',action='store_true',help='24 channels, 12 shards; interface validation only')
    args=parser.parse_args(); root=args.output_root.resolve()
    if args.smoke and args.output_root==Path('data/formal_dmrs'): root=Path('data/smoke').resolve()
    if args.stage=='plan':
        protocol=protocol_for(root,args.smoke)
        print(json.dumps(dict(mode=protocol['generation_mode'],output_root=str(root),
            channels={split:spec['channels'] for split,spec in protocol['splits'].items()},
            validation_partition=protocol['validation_partition'],shard_size=protocol['shard_size'],
            dependencies=['MATLAB','5G Toolbox'],final_test_generated=False),ensure_ascii=False,indent=2)); return
    if root==ROOT or root in ROOT.parents:
        parser.error('Choose a dedicated data output directory')
    with execution_lock(root):
        if args.stage in ('setup','all','resume'): setup(root,args.smoke)
        if args.stage=='setup': print('DATASET_SETUP_COMPLETE'); return
        if args.stage not in ('all','resume'):
            protocol=read(root/'protocol.json')
            if args.smoke != (protocol.get('generation_mode')=='smoke'):
                parser.error('Supply --smoke when working with a smoke dataset')
        if args.stage in ('smoke','all','resume'):
            if not (root/'generation/smoke.json').exists():
                run_matlab(root,'smoke',matlab_executable(args.matlab))
            else:
                if (root/'generation/freeze.json').exists(): check_freeze(root)
                if read(root/'generation/smoke.json')['status']!='PASS': raise ValueError('Previous MATLAB smoke failed')
        if args.stage in ('freeze','all','resume'): freeze(root)
        if args.stage in ('generate','all','resume'):
            run_matlab(root,'source',matlab_executable(args.matlab),args.workers)
        if args.stage in ('replay','all','resume'):
            run_matlab(root,'replay',matlab_executable(args.matlab))
        if args.stage in ('audit','all','resume'):
            result=audit(root)
            print(json.dumps(dict(status='GENERATED_SOURCE_DATA_READY',mode=result['generation_mode'],
                channels=result['independent_channels'],shards=result['shards']),ensure_ascii=False))


if __name__=='__main__':
    main()
