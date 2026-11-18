"""Independent K fitting and immutable shard feature caches; source layouts only."""
from collections import defaultdict
import gc
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from support import *

def observation(dataset,raw,within,layout):
    if layout not in 'ABCD':raise ValueError('Unknown geometry')
    mask=dataset.geometry['pilot_masks'][:,:,'ABCD'.index(layout)].astype(bool).reshape(-1)
    values=(raw['Y'][:,:,within].reshape(-1)[mask]/raw['X'][:,:,within].reshape(-1)[mask]).astype(np.complex64)
    return dict(pilot_layout_id=layout,pilot_values=torch.from_numpy(values),
          pilot_coordinates=dataset.coordinates[mask],target_coordinates=dataset.coordinates,
          noise_variance=torch.tensor(float(raw['noise_variance'].reshape(-1)[within])),
          rms_delay_spread_s=torch.tensor(float(raw['delay_spread_ns'].reshape(-1)[within])*1e-9),
          max_doppler_hz=torch.tensor(float(raw['max_doppler_hz'].reshape(-1)[within])))

def physical_grid(dataset,raw,layout,frontend,coefficients):
    n=raw['Y'].shape[-1];result=np.empty((n,120,14),np.complex64);groups=defaultdict(list)
    for i in range(n):
        sample=observation(dataset,raw,i,layout)
        groups[(float(sample['rms_delay_spread_s']),float(sample['max_doppler_hz']))].append((i,sample))
    for group in groups.values():
        indices=[i for i,_ in group];samples=[s for _,s in group]
        pilots=np.stack([s['pilot_values'].numpy() for s in samples],axis=1)
        result[indices]=frontend.predict(samples[0],pilots,[float(s['noise_variance']) for s in samples],coefficients).reshape(-1,120,14)
    return result

def calibrate(config,output,binding):
    from runner import calibration_groups,log
    identity=dict(binding=digest(binding),contract=config['calibration']);destination=output/'calibration.json'
    if destination.exists():
        result=read(destination)
        if result['identity']!=identity:raise ValueError('Changed calibration identity')
        return result
    dataset,groups=calibration_groups(config);report={}
    log('CALIBRATION_POOL_READY',channels=dataset.channels,observations=len(dataset),source_layouts=['A','C'])
    for arm in K_ARMS:
        progress=output/('calibration_progress_'+mode_for(arm)+'.json')
        previous=read(progress) if progress.exists() else dict(identity=identity,objectives={})
        if previous['identity']!=identity:raise ValueError('Changed calibration progress')
        objectives=previous['objectives'];spec=config['calibration']
        noise,delays,dopplers=(list(spec[k]) for k in ('noise_scales','delay_scales','doppler_scales'))
        frontend=KronFrontend(mode_for(arm));attempts=[]
        for expansion in range(spec['maximum_boundary_expansions']+1):
            for delay,doppler in itertools.product(delays,dopplers):
                key=lambda n:json.dumps([float(n),float(delay),float(doppler)])
                missing=[v for v in noise if key(v) not in objectives]
                if not missing:continue
                totals=np.zeros(len(missing),np.float64);started=time.perf_counter();last_progress=started
                for gi,group in enumerate(groups.values(),1):
                    f,z,mapped=frontend.modal(group['sample'],group['pilots'],[1,delay,doppler])
                    h=group['targets'];energy=np.sum(abs(h)**2,axis=0)
                    if np.any(energy<=0):raise ValueError('Zero calibration energy')
                    gram=np.kron(f['af'].conj().T@f['af'],f['at'].conj().T@f['at'])
                    cross=mapped.conj().T@h
                    for i,n in enumerate(missing):
                        weights=z/frontend.denominator(f,group['variances'],n)
                        error=(np.real(np.sum(weights.conj()*(gram@weights),axis=0))
                             -2*np.real(np.sum(weights.conj()*cross,axis=0))+energy)/energy
                        if not np.isfinite(error).all() or error.min() < -1e-7:raise ValueError('Unstable calibration objective')
                        totals[i]+=np.maximum(error,0).sum()
                    if time.perf_counter()-last_progress>40:
                        log('CALIBRATION_PROGRESS',method=arm,groups_done=gi,groups_total=len(groups));last_progress=time.perf_counter()
                for n,total in zip(missing,totals):objectives[key(n)]=float(total/len(dataset))
                write(progress,dict(identity=identity,objectives=objectives))
                log('CALIBRATION_KERNEL_COMPLETE',method=arm,delay_scale=delay,doppler_scale=doppler,candidates=len(objectives),seconds=time.perf_counter()-started)
            objective,selected=min((objectives[json.dumps([float(n),float(d),float(f)])],[float(n),float(d),float(f)])
                           for n,d,f in itertools.product(noise,delays,dopplers))
            positive=min(n for n in noise if n>0)
            boundary=dict(noise_upper=selected[0]==max(noise),noise_lower_positive=selected[0]==positive,
                          delay=selected[1] in (min(delays),max(delays)),doppler=selected[2] in (min(dopplers),max(dopplers)))
            attempts.append(dict(expansion=expansion,selected=selected,nmse=objective,boundary=boundary))
            log('CALIBRATION_SEARCH_RESULT',method=arm,**attempts[-1])
            if not any(boundary.values()) or config['smoke']:break
            if expansion==spec['maximum_boundary_expansions']:raise RuntimeError('Calibration still on search boundary; progress preserved, no neural training started')
            if boundary['noise_upper']:noise=sorted(set(noise+[max(noise)*2]))
            if boundary['noise_lower_positive']:noise=sorted(set(noise+[positive/2]))
            if boundary['delay']:delays=sorted(set(delays+[min(delays)/2,max(delays)*2]))
            if boundary['doppler']:dopplers=sorted(set(dopplers+[min(dopplers)/2,max(dopplers)*2]))
        max_reference=0.;dense=SpectralFrontend()
        for index in sorted(set((0,1,len(dataset)-2,len(dataset)-1))):
            sample=dataset[index];got=frontend.predict(sample,sample['pilot_values'].numpy(),[float(sample['noise_variance'])],selected)[0]
            independent=CartesianKroneckerFrontend(mode_for(arm),selected).predict(sample['pilot_values'].numpy(),
                 sample['pilot_coordinates'].numpy(),sample['target_coordinates'].numpy(),noise_variance=float(sample['noise_variance']),
                 rms_delay_spread_s=float(sample['rms_delay_spread_s']),max_doppler_hz=float(sample['max_doppler_hz']))
            np.testing.assert_allclose(got,independent,atol=2e-5,rtol=2e-5)
            if mode_for(arm)=='exact':
                reference=dense.predict(sample,sample['pilot_values'].numpy(),[float(sample['noise_variance'])],selected)[0]
                max_reference=max(max_reference,float(abs(got-reference).max()))
        if max_reference>2e-5:raise ValueError('Exact K differs from prior dense estimator')
        report[arm]=dict(coefficients=selected,fit_nmse=objective,attempts=attempts,maximum_dense_error=max_reference,
                         calibration_channels=dataset.channels,calibration_observations=len(dataset),source_layouts=['A','C'])
    result=dict(status='FROZEN_SOURCE_AC_CALIBRATION',identity=identity,arms=report,independently_calibrated=True,final_test_accessed=False)
    write(destination,result);return result

def cache_features(config,output,binding,calibration,alpha_only=False):
    from runner import log
    dataset=source_base('train',config);manifest=read(Path(config['data_root'])/'manifest.json')
    selection_channels=dataset.protocol.get('validation_partition',{}).get('selection_channels',8000)
    expected=dict(binding=digest(binding),calibration_sha256=sha(output/'calibration.json'))
    frontends={arm:KronFrontend(mode_for(arm)) for arm in K_ARMS};records=[]
    cache_root=output/'cache';write(cache_root/'binding.json',expected)
    for entry in manifest['shards']:
        relative=entry['path'].replace(chr(92),'/');split=Path(relative).parts[0]
        shard=int(Path(relative).stem.split('_')[-1]);first=(shard-1)*dataset.protocol['shard_size']
        role='validation_alpha' if split=='validation' and first>=selection_channels else 'validation_selection' if split=='validation' else split
        if (alpha_only and role!='validation_alpha') or (not alpha_only and role not in ('train','validation_selection')):continue
        count=entry['channels']
        if role in ACTIVE_LIMITS:
            start=selection_channels if role=='validation_alpha' else 0
            count=min(count,ACTIVE_LIMITS[role]-(first-start))
            if count<=0:continue
        raw=None
        for arm in K_ARMS:
            mode=mode_for(arm);dest=cache_root/split/f'shard_{shard:04d}_{mode}.npy';marker=Path(str(dest)+'.complete.json')
            identity=dict(**expected,source_sha256=entry['sha256'],feature=mode,shape=[count*2,120,14])
            if dest.exists() and marker.exists():
                old=read(marker)
                if old['identity']!=identity or sha(dest)!=old['sha256']:raise ValueError('Invalid feature cache')
            else:
                if raw is None:
                    raw=sio.loadmat(Path(config['data_root'])/relative,variable_names=['Y','X','noise_variance','delay_spread_ns','max_doppler_hz'])
                    raw=slice_raw(raw,count)
                values=np.empty(identity['shape'],np.complex64)
                for view,layout in enumerate(('A','C')):
                    values[view::2]=physical_grid(dataset,raw,layout,frontends[arm],calibration['arms'][arm]['coefficients'])
                if not np.isfinite(values).all():raise ValueError('Nonfinite cache')
                dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(dest)+'.tmp')
                with tmp.open('wb') as f:np.save(f,values,allow_pickle=False);f.flush();os.fsync(f.fileno())
                os.replace(tmp,dest);write(marker,dict(identity=identity,sha256=sha(dest)))
                log('FEATURE_CACHE_COMMITTED',split=split,shard=shard,feature=mode,observations=len(values))
            records.append(dict(path=dest.relative_to(output).as_posix(),sha256=read(marker)['sha256']))
    write(cache_root/('alpha_cache.json' if alpha_only else 'training_cache.json'),dict(identity=expected,files=records))

def slice_raw(raw,count):
    result={}
    for k,v in raw.items():
        if k.startswith('__'):continue
        if k in ('Y','X','H_true','Y_noiseless'):v=v[:,:,:count]
        elif k=='tx_bits':v=v[:,:,:,:count]
        elif k in ('snr_db','delay_spread_ns','max_doppler_hz','noise_variance','target_energy'):v=v.reshape(-1)[:count]
        result[k]=v
    return result

def verify_training_cache(config,output,binding):
    index=read(output/'cache/training_cache.json')
    if index['identity']!=dict(binding=digest(binding),calibration_sha256=sha(output/'calibration.json')):raise ValueError('Cache binding changed')
    expected=set()
    for role in ('train','validation_selection'):
        dataset=source_base(role,config)
        size=dataset.protocol['shard_size'];first=dataset.start//size;stop=(dataset.start+dataset.channels-1)//size+1
        expected.update(f'cache/{dataset.split}/shard_{shard+1:04d}_exact.npy' for shard in range(first,stop))
    if {e['path'] for e in index['files']}!=expected or len(index['files'])!=len(expected):raise ValueError('Incomplete training cache')
    for entry in index['files']:
        if sha(output/entry['path'])!=entry['sha256']:raise ValueError('Corrupt training cache')

def preflight(config,output,device):
    from runner import update,move_batch,optimizer_for,runtime,log
    from torch.utils.data._utils.collate import default_collate
    dataset=source_base('train',config);features={}
    for arm in K_ARMS:
        frontend=KronFrontend(mode_for(arm));features[arm]=[]
        for i in (0,1):
            sample=dataset[i];grid=frontend.predict(sample,sample['pilot_values'].numpy(),[float(sample['noise_variance'])],[1,1,1])[0]
            features[arm].append(dict(input=torch.from_numpy(grid.reshape(120,14)),target=sample['target_values'].reshape(120,14),
                   **{k:sample[k] for k in ('snr_db','rms_delay_spread_s','max_doppler_hz')}))
    size=2 if device.type=='cpu' else config['batch_size'];checks=[]
    for method in METHODS:
        pair=features[method]
        batch=default_collate([pair[i%2] for i in range(size)]);seed_all(config['training_seeds'][0])
        model=model_create(method,config,device);optimizer,scheduler=optimizer_for(model,config)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        log('PREFLIGHT_BEGIN',method=method,batch_size=size)
        try:
            for _ in range(2):loss,count=update(model,method,batch,optimizer,device)
            model.eval();obs,target=move_batch(batch,device)
            with torch.inference_mode():
                pred=inference(model,method,obs)
                if pred.shape!=target.shape or not torch.isfinite(pred).all():raise ValueError('Invalid preflight output')
        except torch.cuda.OutOfMemoryError as exc:raise RuntimeError(f'Actual batch {size} failed for {method}; batch was not reduced') from exc
        checks.append(dict(method=method,actual_batch=count,loss=loss,parameters=sum(p.numel() for p in model.parameters()),
                    train_batches=2,validation_batches=1,peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
        del model,optimizer,scheduler,batch,obs,target,pred;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    result=dict(status='PASS',checks=checks,runtime=runtime(device),samples_repeated_for_memory_check=True,formal_training_performed=False)
    write(output/'preflight'/f'{device.type}.json',result);log('PREFLIGHT_PASS',models=len(checks),actual_batch=size)
    return result

def verify_frozen(config,output,binding):
    freeze=read(output/'model_freeze.json')
    if freeze['binding']!=digest(binding) or freeze['calibration_sha256']!=sha(output/'calibration.json') or freeze['alpha_sha256']!=sha(output/'alpha.json'):raise ValueError('Frozen binding changed')
    for row in freeze['runs']:
        if sha(output/row['checkpoint'])!=row['sha256']:raise ValueError('Frozen checkpoint changed')
    if len(freeze['runs'])!=len(METHODS)*len(config['training_seeds']):raise ValueError('Not all models frozen')
    return freeze

def fit_alpha_and_freeze(config,output,binding,device):
    from runner import run_identity,move_batch,log
    runs=[]
    for seed,method in itertools.product(config['training_seeds'],METHODS):
        folder=output/'training'/str(seed)/method;completion=read(folder/'completion.json')
        if completion['identity']!=run_identity(method,seed,config,output,binding,device) or sha(folder/'best.pt')!=completion['best_checkpoint_sha256']:raise ValueError('Incomplete or changed run')
        runs.append(dict(method=method,seed=seed,checkpoint=(folder/'best.pt').relative_to(output).as_posix(),sha256=completion['best_checkpoint_sha256'],
                    epochs=completion['epochs'],updates=completion['updates'],complex_observation_exposures=completion['complex_observation_exposures']))
    identity=dict(binding=digest(binding),runs=runs,calibration_sha256=sha(output/'calibration.json'),contract=config['alpha'])
    path=output/'alpha.json'
    if path.exists():
        alpha_record=read(path)
        if alpha_record['identity']!=identity:raise ValueError('Changed alpha identity')
    else:
        cache_features(config,output,binding,read(output/'calibration.json'),alpha_only=True);alphas={};details={}
        for arm in K_ARMS:
            ds=SourceDataset('validation_alpha',config['data_root'],output/'cache',mode_for(arm));aa=bb=0.;per_seed=[]
            for seed in config['training_seeds']:
                model=model_create(arm,config,device);model.load_state_dict(torch.load(output/'training'/str(seed)/arm/'best.pt',map_location='cpu',weights_only=False)['model']);model.eval()
                a=b=0.;count=0
                with torch.inference_mode():
                    for batch in DataLoader(ds,batch_size=config['batch_size'],shuffle=False,generator=torch.Generator().manual_seed(1)):
                        obs,target=move_batch(batch,device);pred=inference(model,arm,obs).to(torch.complex128)
                        base=obs['input'].to(torch.complex128);delta=pred-base;error=target.to(torch.complex128)-base
                        energy=target.to(torch.complex128).abs().square().flatten(1).sum(1)
                        a+=float((delta.abs().square().flatten(1).sum(1)/energy).sum());b+=float(((delta.conj()*error).real.flatten(1).sum(1)/energy).sum());count+=len(target)
                if count!=len(ds):raise ValueError('Alpha pool incomplete')
                aa+=a;bb+=b;per_seed.append(dict(seed=seed,a=a,b=b,observations=count));del model
            alphas[arm]=float(np.clip(bb/aa,0,1)) if aa else 0.;details[arm]=per_seed
        alpha_record=dict(identity=identity,alpha=alphas,source_layouts=['A','C'],per_seed=details,split='validation_alpha',final_test_accessed=False)
        write(path,alpha_record)
    freeze=dict(status='SMOKE_ONLY' if config['smoke'] else 'FROZEN_SOURCE_MODELS',binding=digest(binding),runs=runs,
              calibration_sha256=sha(output/'calibration.json'),alpha_sha256=sha(path),alpha=alpha_record['alpha'],final_test_accessed=False,
              K_fit_layouts=['A','C'],specialist_layout_supervision=False,from_scratch=True)
    write(output/'model_freeze.json',freeze);verify_frozen(config,output,binding)
    log('ALL_TRAINING_COMPLETE',runs=len(runs),alpha=alpha_record['alpha'],final_test_accessed=False)
