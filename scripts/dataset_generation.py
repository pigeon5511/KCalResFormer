"""Portable source-dataset setup, MATLAB orchestration and stored-array audit."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO/'scripts/matlab/generate_formal.m'
SMOKE_COUNTS = dict(calibration=6, train=8, validation=10)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(temporary,path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def execution_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'generation.lock').open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            if stream.read(1) == b'':
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def protocol_for(root, smoke=False):
    protocol = deepcopy(read(REPO/'configs/dataset_protocol.json'))
    protocol.update(created_utc=datetime.now(timezone.utc).isoformat(), data_root=str(root),
                    generation_mode='smoke' if smoke else 'formal',
                    validation_partition=dict(selection_channels=6 if smoke else 8000,
                                              alpha_channels=4 if smoke else 2000))
    if smoke:
        protocol['shard_size'] = 2
        protocol['storage'] = 'MAT v6 smoke dataset: two independent channels/shard; not a formal accuracy dataset.'
        for split,count in SMOKE_COUNTS.items():
            protocol['splits'][split].update(channels=count,views=2*count,
                role='SMOKE_ONLY: same physical generator on a small source subset')
    return protocol


def validate_protocol(protocol, allow_smoke=False):
    mode = protocol.get('generation_mode')
    require(mode in ('formal','smoke'), 'Missing generated-data mode')
    require(mode != 'smoke' or allow_smoke, 'Smoke data require --smoke; they cannot enter formal training')
    template = protocol_for(Path('.'), mode=='smoke')
    ignored = {'created_utc','data_root'}
    require({k:v for k,v in protocol.items() if k not in ignored} ==
            {k:v for k,v in template.items() if k not in ignored},
            'Generated data differ from the declared source recipe')


def setup(root, smoke=False):
    directory = root/'generation'
    if (root/'protocol.json').exists():
        protocol = read(root/'protocol.json')
        validate_protocol(protocol,allow_smoke=smoke)
        require(protocol['generation_mode']==('smoke' if smoke else 'formal'), 'Existing dataset has another mode')
        require((directory/'setup.json').is_file(),'Incomplete setup; use a new output directory')
        require(sha(directory/'generate_formal.m')==sha(GENERATOR),'Generator changed; use a new output directory')
        for relative,digest in read(directory/'setup.json')['files'].items():
            require(sha(root/relative)==digest,f'Setup file changed: {relative}')
        return protocol
    existing = [path for path in root.iterdir() if path.name!='generation.lock']
    require(not existing, 'Choose a new output directory; existing files will not be overwritten')
    directory.mkdir()
    (directory/'logs').mkdir()
    protocol = protocol_for(root,smoke)
    write(root/'protocol.json',protocol)
    write(directory/'protocol.json',protocol)
    shutil.copyfile(GENERATOR,directory/'generate_formal.m')
    conditions=np.array(list(itertools.product(protocol['conditions']['snr_db'],
        protocol['conditions']['delay_spread_ns'],protocol['conditions']['max_doppler_hz'])),np.int64)
    sio.savemat(directory/'conditions.mat',{'conditions':conditions.astype(np.float64)})
    all_seeds=set(); summary={}
    for split,spec in protocol['splits'].items():
        (root/split).mkdir()
        count=spec['channels']; rng=np.random.default_rng(spec['schedule_seed'])
        indices=np.concatenate([rng.permutation(len(conditions)) for _ in range(math.ceil(count/len(conditions)))])[:count]
        matrix=conditions[indices]; ids=np.arange(1,count+1,dtype=np.uint32)
        role=np.ones(count,np.uint8)
        if split=='validation': role[protocol['validation_partition']['selection_channels']:]=2
        seeds=spec['seed_base']+ids.astype(np.uint64)
        for offset in protocol['seed_stream_offsets'].values():
            assigned=set((seeds+offset).tolist())
            require(not all_seeds.intersection(assigned),'Seed streams overlap')
            all_seeds.update(assigned)
        sio.savemat(directory/f'schedule_{split}.mat',dict(channel_id=ids,
            condition_index=(indices+1).astype(np.uint16),snr_db=matrix[:,0].astype(np.uint8),
            delay_spread_ns=matrix[:,1].astype(np.uint16),max_doppler_hz=matrix[:,2].astype(np.uint16),
            role=role,channel_seed=seeds.astype(np.uint32)))
        counts=np.bincount(indices,minlength=len(conditions))
        require(int(np.ptp(counts))<=1,'Unbalanced condition schedule')
        nested={}
        for size in (10000,30000,100000) if split=='train' and not smoke else ():
            prefix=np.bincount(indices[:size],minlength=len(conditions))
            require(int(np.ptp(prefix))<=1,'Unbalanced nested train prefix')
            nested[str(size)]=dict(min_per_condition=int(prefix.min()),max_per_condition=int(prefix.max()))
        summary[split]=dict(channels=count,min_per_condition=int(counts.min()),
                            max_per_condition=int(counts.max()),nested=nested)
    write(directory/'schedule_audit.json',dict(status='PASS',splits=summary,
        unique_channel_data_noise_seeds=len(all_seeds),
        scope='Disjoint streams within this dataset; historical experiments outside this repository are not searched'))
    paths=[root/'protocol.json',directory/'protocol.json',directory/'generate_formal.m',
           directory/'conditions.mat',directory/'schedule_audit.json',*sorted(directory.glob('schedule_*.mat'))]
    write(directory/'setup.json',dict(files={path.relative_to(root).as_posix():sha(path) for path in paths}))
    return protocol


def check_freeze(root):
    record=read(root/'generation/freeze.json')
    for relative,digest in record['files'].items():
        path=(root/relative).resolve()
        require(root in path.parents and path.is_file(),f'Invalid frozen file: {relative}')
        require(sha(path)==digest,f'Frozen generation file changed: {relative}')
    require(sha(root/'protocol.json')==record['protocol_sha256'],'Root protocol changed')
    require(sha(root/'generation/generate_formal.m')==record['generator_sha256'],'Generator changed')
    return record


def freeze(root):
    path=root/'generation/freeze.json'
    if path.exists(): return check_freeze(root)
    require(read(root/'generation/smoke.json')['status']=='PASS','MATLAB smoke must pass before freezing')
    shutil.copyfile(root/'generation/geometry.mat',root/'geometry.mat')
    setup_files=read(root/'generation/setup.json')['files']
    paths=[*setup_files,'generation/setup.json','generation/smoke.json','generation/geometry.mat','geometry.mat']
    record=dict(format_version='kcalresformer-generated-source-v1',
        protocol_sha256=sha(root/'protocol.json'),generator_sha256=sha(root/'generation/generate_formal.m'),
        files={relative:sha(root/relative) for relative in paths}, final_test_generated=False)
    write(path,record)
    return check_freeze(root)


def matlab_executable(value):
    candidate=value or os.environ.get('MATLAB_EXECUTABLE') or 'matlab'
    resolved=shutil.which(candidate)
    if resolved: return resolved
    if Path(candidate).is_file(): return str(Path(candidate).resolve())
    raise FileNotFoundError('MATLAB not found. Supply --matlab /path/to/matlab or set MATLAB_EXECUTABLE')


def matlab_quote(value):
    return "'"+str(value).replace('\\','/').replace("'","''")+"'"


def run_matlab(root,stage,executable,workers=1):
    directory=root/'generation'
    require(1<=workers<=2,'Use one or two workers')
    if stage!='smoke': check_freeze(root)
    processes=[]; worker_count=workers if stage=='source' else 1
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    def stop_all():
        for process in processes:
            if process.poll() is None:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                else:
                    process.terminate()
        for process in processes:
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
    try:
        for worker in range(1,worker_count+1):
            statement=(f"addpath({matlab_quote(directory)}); generate_formal({matlab_quote(stage)},"
                       f"{worker},{worker_count},{matlab_quote(directory)},{matlab_quote(root)});")
            log_path=directory/'logs'/f'{stamp}_{stage}_worker{worker}.log'
            with log_path.open('wb') as stream:
                process=subprocess.Popen([executable,'-nosplash','-singleCompThread','-batch',statement],
                    cwd=directory,stdout=stream,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            processes.append(process)
        start=time.monotonic(); next_report=0
        while True:
            states=[process.poll() for process in processes]
            if any(code is not None and code!=0 for code in states):
                raise RuntimeError(f'MATLAB stage failed. Inspect {directory / "logs"}')
            if all(code is not None for code in states): break
            if time.monotonic()>=next_report:
                print(json.dumps(dict(stage=stage,elapsed_seconds=round(time.monotonic()-start,1),
                    completed_shards=sum(len(list((root/split).glob('*.complete.json'))) for split in ('train','calibration','validation')))),flush=True)
                next_report=time.monotonic()+20
            time.sleep(0.5)
    except BaseException:
        stop_all()
        raise
    write(directory/f'{stage}_exit.json',dict(status='PASS',exit_codes=states,
                                              elapsed_seconds=time.monotonic()-start))


def string(value):
    return ''.join(np.asarray(value).reshape(-1).astype(str))


def validate_generated_binding(root,allow_smoke=False):
    root=Path(root).resolve(); protocol=read(root/'protocol.json')
    validate_protocol(protocol,allow_smoke)
    frozen=check_freeze(root); manifest=read(root/'manifest.json')
    expected_status='SMOKE_COMPLETE_AND_VERIFIED' if protocol['generation_mode']=='smoke' else 'COMPLETE_AND_VERIFIED'
    require(manifest['status']==expected_status,'Generated dataset has not passed the complete audit')
    require(manifest['protocol_sha256']==sha(root/'protocol.json'),'Dataset protocol changed')
    require(manifest['freeze_sha256']==sha(root/'generation/freeze.json'),'Generation freeze changed')
    require(manifest['audit_sha256']==sha(root/'audit.json'),'Data audit changed')
    require(manifest['dataset_version']==protocol['dataset_version'],'Dataset version mismatch')
    require(sha(root/'geometry.mat')==frozen['files']['geometry.mat'],'Geometry changed')
    require(sha(root/'generation/generate_formal.m')==sha(GENERATOR),
            'Generated data use another generator; declare and audit a separate experiment')
    return protocol,manifest


def audit(root):
    frozen=check_freeze(root); p=read(root/'protocol.json'); validate_protocol(p,allow_smoke=True)
    geometry=sio.loadmat(root/'geometry.mat'); masks=geometry['pilot_masks'].astype(bool)
    expected_rows=np.array([12*block+offset for block in range(10) for offset in (0,1,6,7)])
    for index,symbols in enumerate(([2],[2,11],[2,7,11],[2,5,8,11])):
        expected=np.zeros((120,14),bool); expected[np.ix_(expected_rows,symbols)]=True
        require(np.array_equal(masks[:,:,index],expected),'Unexpected DM-RS geometry')
    union=masks.any(-1); data_mask=geometry['data_mask'].astype(bool)
    require(np.array_equal(data_mask,~union) and data_mask.sum()==1480,'Wrong common BER data mask')
    conditions=sio.loadmat(root/'generation/conditions.mat')['conditions']
    seeds=set(); truths=set(); records=[]; summaries={}; total_noise=0.; noise_res=0
    noise_by_snr={str(snr):[0.,0] for snr in p['conditions']['snr_db']}
    maximum_energy_error=0.; maximum_mapping_error=0.; total=0
    for split,spec in p['splits'].items():
        schedule=sio.loadmat(root/f'generation/schedule_{split}.mat')
        count=spec['channels']; shard_count=math.ceil(count/p['shard_size'])
        paths={f'shard_{index:04d}.mat' for index in range(1,shard_count+1)}
        require({path.name for path in (root/split).glob('shard_*.mat')}==paths,f'Incomplete or extra shards: {split}')
        require({path.name for path in (root/split).glob('shard_*.mat.complete.json')}==
                {name+'.complete.json' for name in paths},f'Incomplete shard markers: {split}')
        require(not list((root/split).glob('*.partial.mat')),f'Uncommitted partial shard in {split}')
        counts=np.zeros(len(conditions),np.int64)
        for shard in range(1,shard_count+1):
            path=root/split/f'shard_{shard:04d}.mat'; done=read(str(path)+'.complete.json')
            lo=(shard-1)*p['shard_size']; hi=min(lo+p['shard_size'],count); n=hi-lo
            require(done['status']=='COMPLETE' and done['channels']==n and
                done['first_channel_id']==lo+1 and done['last_channel_id']==hi,'Shard identity/count mismatch')
            digest=sha(path)
            require(digest==done['sha256'],'Shard checksum mismatch')
            require(done['protocol_sha256']==frozen['protocol_sha256'] and
                    done['generator_sha256']==frozen['generator_sha256'],'Shard generation binding mismatch')
            m=sio.loadmat(path)
            require(string(m['protocol_sha256'])==frozen['protocol_sha256'] and string(m['split_name'])==split
                and string(m['dataset_version'])==p['dataset_version'],'Stored shard metadata mismatch')
            for key in ('channel_id','condition_index','snr_db','delay_spread_ns','max_doppler_hz','role','channel_seed'):
                require(np.array_equal(m[key].reshape(-1),schedule[key].reshape(-1)[lo:hi]),f'Schedule mismatch: {key}')
            for key in ('H_true','X','Y','Y_noiseless'):
                require(m[key].shape==(120,14,n) and m[key].dtype==np.complex64 and np.isfinite(m[key]).all(),f'Invalid field: {key}')
            require(m['tx_bits'].shape==(120,14,2,n) and m['tx_bits'].dtype==np.uint8 and
                    np.isin(m['tx_bits'],[0,1]).all(),'Invalid reference bits')
            require(np.array_equal(m['subcarrier_frequencies_hz'],geometry['subcarrier_frequencies_hz']) and
                    np.array_equal(m['symbol_times_s'],geometry['symbol_times_s']),'Coordinate mismatch')
            truth=m['H_true'].astype(np.complex128); energy=np.mean(abs(truth)**2,axis=(0,1))
            require(np.all(energy>0) and np.allclose(energy,m['target_energy'].reshape(-1),rtol=2e-14,atol=1e-15),'Stored target energy mismatch')
            maximum_energy_error=max(maximum_energy_error,float(abs(energy-m['target_energy'].reshape(-1)).max()))
            bits=m['tx_bits'].astype(np.float64)
            mapping=((1-2*bits[:,:,0,:])+1j*(1-2*bits[:,:,1,:]))/np.sqrt(2)
            error=float(abs(mapping-m['X']).max()); maximum_mapping_error=max(maximum_mapping_error,error)
            require(error<5e-8,'QPSK mapping mismatch')
            require(abs(m['X'][union,:]-geometry['pilot_symbols'][union,None]).max()<1e-7,'Wrong pilot symbols')
            variance=10.**(-m['snr_db'].reshape(-1).astype(np.float64)/10.)
            require(np.allclose(variance,m['noise_variance'].reshape(-1),rtol=2e-15),'Noise variance mismatch')
            normalized=np.sum(abs(m['Y'].astype(np.complex128)-m['Y_noiseless'].astype(np.complex128))**2,axis=(0,1))/variance
            total_noise+=float(normalized.sum()); noise_res+=1680*n
            for snr in p['conditions']['snr_db']:
                selected=m['snr_db'].reshape(-1)==snr
                noise_by_snr[str(snr)][0]+=float(normalized[selected].sum())
                noise_by_snr[str(snr)][1]+=int(selected.sum())*1680
            mismatch=np.mean(abs(m['Y_noiseless'].astype(np.complex128)-truth*m['X'].astype(np.complex128))**2,axis=(0,1))/energy
            require(np.allclose(mismatch,m['noiseless_mismatch_nmse'].reshape(-1),rtol=2e-13,atol=1e-15),'Stored waveform mismatch metric differs')
            for key,offset in (('channel_seed',0),('data_seed',10000000),('noise_seed',20000000)):
                assigned=m[key].reshape(-1).astype(np.uint64)
                require(np.array_equal(assigned,spec['seed_base']+np.arange(lo+1,hi+1,dtype=np.uint64)+offset),'Wrong seed assignment')
                require(seeds.isdisjoint(assigned.tolist()),'Random streams overlap')
                seeds.update(assigned.tolist())
            for index in range(n):
                identity=hashlib.sha256(np.ascontiguousarray(m['H_true'][:,:,index]).tobytes()).hexdigest()
                require(identity not in truths,'Duplicate channel truth across stored channels'); truths.add(identity)
            np.add.at(counts,m['condition_index'].reshape(-1).astype(int)-1,1)
            records.append(dict(path=path.relative_to(root).as_posix(),sha256=digest,bytes=path.stat().st_size,
                channels=n,views=2*n,first_channel_id=lo+1,last_channel_id=hi))
            total+=n
            if len(records)%10==0: print(json.dumps(dict(stage='audit',channels=total,shards=len(records))),flush=True)
        require(int(np.ptp(counts))<=1,'Unbalanced condition counts')
        summaries[split]=dict(channels=count,min_per_condition=int(counts.min()),max_per_condition=int(counts.max()))
    normalized_noise=total_noise/noise_res
    # Small smoke samples need sampling uncertainty; formal tolerances retain the original audit.
    tolerance=max(0.003,6/math.sqrt(noise_res))
    require(abs(normalized_noise-1)<tolerance,'AWGN power check failed')
    ratios={}
    for snr,(power,number) in noise_by_snr.items():
        if number:
            ratios[snr]=power/number
            require(abs(power/number-1)<max(0.005,6/math.sqrt(number)),f'AWGN power check failed at SNR {snr}')
    require(read(root/'generation/replay.json')['status']=='PASS','MATLAB replay must pass before the full audit')
    require(not (root/'test').exists(),'Source generation does not include an independent final test')
    result=dict(status='PASS',generation_mode=p['generation_mode'],independent_channels=total,
        observations=2*total,shards=len(records),unique_channel_data_noise_seeds=len(seeds),distinct_truth_hashes=len(truths),
        normalized_AWGN_power=normalized_noise,normalized_AWGN_power_by_snr=ratios,
        max_target_energy_absolute_error=maximum_energy_error,max_QPSK_mapping_error=maximum_mapping_error,
        splits=summaries,source_layouts=['A','C'],final_test_generated=False,
        matlab_smoke=read(root/'generation/smoke.json'),replay=read(root/'generation/replay.json'))
    write(root/'audit.json',result)
    manifest=dict(status='SMOKE_COMPLETE_AND_VERIFIED' if p['generation_mode']=='smoke' else 'COMPLETE_AND_VERIFIED',
        dataset_version=p['dataset_version'],generation_mode=p['generation_mode'],independent_channels=total,
        observations=2*total,shards=records,protocol_sha256=sha(root/'protocol.json'),
        freeze_sha256=sha(root/'generation/freeze.json'),audit_sha256=sha(root/'audit.json'),final_test_generated=False)
    write(root/'manifest.json',manifest)
    validate_generated_binding(root,allow_smoke=True)
    return result
