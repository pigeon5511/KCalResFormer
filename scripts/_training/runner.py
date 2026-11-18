"""Separate training/freeze and source-development cross-layout evaluation."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import gc
import itertools
import math
import os
from pathlib import Path
import platform
import sys
import time
from support import *
from torch.utils.data import DataLoader

def dataset_for(method,split,config,output):
    if method in K_ARMS:return SourceDataset(split,config['data_root'],output/'cache',mode_for(method))
    raise ValueError('Unknown model')

def run_identity(method,seed,config,output,binding,device):
    identity=dict(method=method,seed=seed,binding=digest(binding),runtime=runtime(device))
    if method in K_ARMS:identity['calibration_sha256']=sha(output/'calibration.json')
    return identity
def log(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False, allow_nan=False), flush=True)


def runtime(device):
    import scipy
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                scipy=scipy.__version__, device=str(device), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)


def configure(config, device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select a GPU in the training task; CPU checks cannot start formal training.")


def source_identity(config):
    manifest = read(CONFIG_ROOT/'SOURCE_MANIFEST.json')
    for relative, expected in manifest['files'].items():
        if sha(REPO_ROOT/relative) != expected:
            raise ValueError(f'Source changed: {relative}. Regenerate the source manifest for a new experiment.')
    return dict(configuration=config, package=manifest['files'])


@contextmanager
def exclusive(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/"execution.lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if stream.read(1) == b"":
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def audit_data(config, output):
    root = Path(config["data_root"]).resolve()
    protocol = read(CONFIG_ROOT/"dataset_protocol.json")
    protocol_sha = sha(root/"protocol.json")
    if protocol["dataset_version"] != config["dataset_version"]:
        raise ValueError("Unexpected dataset version")
    for name in ("protocol.json", "manifest.json"):
        if not (root/name).is_file():
            raise FileNotFoundError(root/name)
    actual_protocol = read(root/"protocol.json")
    generated = config.get('dataset_origin') == 'generated'
    if generated:
        from scripts.dataset_generation import validate_generated_binding
        protocol, _ = validate_generated_binding(root,allow_smoke=config['smoke'])
    else:
        actual_protocol.pop("data_root", None)
        if actual_protocol != protocol:
            raise ValueError("Dataset protocol differs from frozen support protocol")
    manifest = read(root/"manifest.json")
    status = 'SMOKE_COMPLETE_AND_VERIFIED' if generated and protocol['generation_mode']=='smoke' else 'COMPLETE_AND_VERIFIED'
    if manifest["status"] != status or manifest["dataset_version"] != config["dataset_version"]:
        raise ValueError("Dataset completion manifest is invalid")
    if sha(root/"manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Different source dataset realizations; use the audited formal dataset")
    expected_count = sum(math.ceil(s["channels"]/protocol["shard_size"]) for s in protocol["splits"].values())
    if len(manifest["shards"]) != expected_count:
        raise ValueError("Incomplete source manifest")
    identity = source_identity(config)
    identity.update(data_root=str(root), protocol_sha256=protocol_sha, manifest_sha256=sha(root/"manifest.json"))
    old_path = output/"data_audit.json"
    old = read(old_path) if old_path.exists() else None
    records = []
    paths = set()
    old_stats = {v["path"]: v for v in old["shards"]} if old and old["identity"] == identity else {}
    for index, entry in enumerate(manifest["shards"]):
        relative = entry["path"].replace("\\", "/")
        path = (root/relative).resolve()
        if root not in path.parents or relative in paths or Path(relative).parts[0] not in ("calibration", "train", "validation"):
            raise ValueError(f"Invalid shard path: {relative}")
        paths.add(relative)
        marker = path.with_name(path.name+".complete.json")
        completion = read(marker)
        stat = path.stat()
        current = dict(path=relative, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                       sha256=entry["sha256"], marker_sha256=sha(marker))
        if stat.st_size != entry["bytes"] or completion["sha256"] != entry["sha256"] or completion["protocol_sha256"] != protocol_sha or completion["status"] != "COMPLETE":
            raise ValueError(f"Shard/marker mismatch: {path}")
        if old_stats.get(relative) != current and sha(path) != entry["sha256"]:
            raise ValueError(f"Corrupted shard: {path}")
        records.append(current)
        if index % 25 == 0:
            log("DATA_AUDIT", completed=index+1, total=expected_count)
    expected_paths = {f"{split}/shard_{i:04d}.mat" for split, spec in protocol["splits"].items()
                      for i in range(1, math.ceil(spec["channels"]/protocol["shard_size"])+1)}
    if paths != expected_paths:
        raise ValueError("Manifest shard set differs from declared split sizes")
    binding = dict(identity=identity, shard_hashes={v["path"]:v["sha256"] for v in records})
    if (output/"binding.json").exists() and read(output/"binding.json") != binding:
        raise ValueError("Experiment binding changed. Use a new output folder; never mix runs.")
    write(output/"binding.json", binding)
    write(old_path, dict(status="PASS", identity=identity, shards=records, final_test_accessed=False))
    log("SOURCE_DATA_PASS", shards=len(records), binding=digest(binding))
    return binding


def move_batch(batch, device):
    target = batch["target"].to(device)
    observation = {k:v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor) and k != "target"}
    return observation, target


def optimizer_for(model, config):
    spec = config["optimizer"]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=spec["lr"],
        betas=tuple(spec["betas"]), eps=spec["eps"], weight_decay=spec["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["scheduler"]["gamma"])
    return optimizer, scheduler


def update(model, method, batch, optimizer, device):
    model.train()
    observation, target = move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    prediction = inference(model, method, observation)
    losses = (prediction-target).abs().square().flatten(1).mean(1)
    loss = losses.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("Nonfinite gradient")
    optimizer.step()
    return float(loss.detach()), len(target)


def calibration_groups(config):
    dataset = source_base('calibration',config)
    groups = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        key = (sample["pilot_layout_id"], float(sample["rms_delay_spread_s"]), float(sample["max_doppler_hz"]))
        if key not in groups:
            groups[key] = dict(sample=sample, pilots=[], targets=[], variances=[])
        group = groups[key]
        group["pilots"].append(sample["pilot_values"].numpy())
        group["targets"].append(sample["target_values"].numpy())
        group["variances"].append(float(sample["noise_variance"]))
    for group in groups.values():
        group["pilots"] = np.stack(group["pilots"], axis=1).astype(np.complex128)
        group["targets"] = np.stack(group["targets"], axis=1).astype(np.complex128)
        group["variances"] = np.asarray(group["variances"])
    return dataset, groups


def raw_observation(dataset, raw, within, view):
    layout, layout_index = (("A",0),("C",2))[view]
    mask = dataset.geometry["pilot_masks"][:,:,layout_index].astype(bool).reshape(-1)
    values = (raw["Y"][:,:,within].reshape(-1)[mask]/raw["X"][:,:,within].reshape(-1)[mask]).astype(np.complex64)
    return dict(pilot_layout_id=layout,pilot_values=torch.from_numpy(values),pilot_grid_mask=torch.from_numpy(mask),
                pilot_coordinates=dataset.coordinates[mask],target_coordinates=dataset.coordinates,
                noise_variance=torch.tensor(float(raw["noise_variance"].reshape(-1)[within])),
                rms_delay_spread_s=torch.tensor(float(raw["delay_spread_ns"].reshape(-1)[within])*1e-9),
                max_doppler_hz=torch.tensor(float(raw["max_doppler_hz"].reshape(-1)[within])))


def evaluate_selection(model,method,dataset,config,device):
    model.eval()
    loader=DataLoader(dataset,batch_size=config["batch_size"],shuffle=False,num_workers=0,
                      generator=torch.Generator().manual_seed(1))
    total,count=0.,0
    with torch.inference_mode():
        for batch in loader:
            observation,target=move_batch(batch,device)
            prediction=inference(model,method,observation)
            losses=(prediction.to(torch.complex128)-target.to(torch.complex128)).abs().square().flatten(1).mean(1)
            if not bool(torch.isfinite(losses).all()):
                raise FloatingPointError("Nonfinite selection loss")
            total+=float(losses.sum());count+=len(losses)
    return total/count


def best_checkpoint(payload):
    return dict(identity=payload["identity"],model=payload["best_model"],epoch=payload["best_epoch"],
                validation_mse=payload["best_mse"],selection_metric="validation_selection_complex_MSE")


def train_one(method,seed,config,output,binding,device):
    folder=output/"training"/str(seed)/method
    folder.mkdir(parents=True,exist_ok=True)
    identity=run_identity(method,seed,config,output,binding,device)
    completion_path=folder/"completion.json"
    if completion_path.exists():
        completion=read(completion_path)
        if completion["identity"]!=identity or sha(folder/"best.pt")!=completion["best_checkpoint_sha256"]:
            raise ValueError(f"Changed completed run: {folder}")
        log("VERIFIED_COMPLETED_RUN",method=method,seed=seed)
        return completion
    seed_all(seed)
    model=model_create(method,config,device)
    optimizer,scheduler=optimizer_for(model,config)
    train=dataset_for(method,"train",config,output)
    selection=dataset_for(method,"validation_selection",config,output)
    sampler=OrderSampler(train,config["batch_size"],seed)
    key="adafortitran" if method.startswith("adafortitran_") else method
    cap=config["maximum_epochs"][key]
    state=dict(epoch=0,next_batch=0,updates=0,exposures=0,train_sum=0.,train_count=0,
               best_mse=float("inf"),best_epoch=None,bad_epochs=0,best_model=None,history=[],elapsed_seconds=0.)
    last=folder/"last.pt"
    if last.exists():
        payload=torch.load(last,map_location="cpu",weights_only=False)
        if payload["identity"]!=identity:
            raise ValueError("Resume identity/runtime changed; cannot continue this experiment")
        model.load_state_dict(payload["model"],strict=True)
        optimizer.load_state_dict(payload["optimizer"]);scheduler.load_state_dict(payload["scheduler"])
        state=payload["state"];restore_rng(payload["rng"])
        log("RESUMED",method=method,seed=seed,epoch=state["epoch"]+1,next_batch=state["next_batch"],updates=state["updates"])
    started=time.perf_counter();prior_elapsed=state["elapsed_seconds"]

    def checkpoint():
        state["elapsed_seconds"]=prior_elapsed+time.perf_counter()-started
        payload=dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),rng=rng_state(),state=state)
        save(last,payload)
        return payload

    while state["epoch"]<cap and state["bad_epochs"]<config["early_stopping_patience"]:
        epoch=state["epoch"]
        sampler.set_epoch(epoch,state["next_batch"])
        loader=DataLoader(train,batch_sampler=sampler,num_workers=0,
                          generator=torch.Generator().manual_seed(seed+epoch))
        log("TRAIN_EPOCH_BEGIN",method=method,seed=seed,epoch=epoch+1,total_batches=sampler.total_batches,start_batch=state["next_batch"])
        for batch in loader:
            loss,count=update(model,method,batch,optimizer,device)
            state["updates"]+=1;state["exposures"]+=count;state["next_batch"]+=1
            state["train_sum"]+=loss*count;state["train_count"]+=count
            if state["updates"]%config["checkpoint_every_updates"]==0:
                checkpoint()
                log("TRAIN_PROGRESS",method=method,seed=seed,epoch=epoch+1,batch=state["next_batch"],
                    total_batches=sampler.total_batches,loss=loss,updates=state["updates"])
        validation=evaluate_selection(model,method,selection,config,device)
        expected_updates=(epoch+1)*sampler.total_batches
        if state['train_count']!=len(train) or state['updates']!=expected_updates or state['exposures']!=(epoch+1)*len(train):
            raise ValueError('Incomplete epoch or optimizer/exposure budget mismatch')
        improved=validation<state["best_mse"]-config["minimum_improvement"]
        if improved:
            state["best_mse"]=validation;state["best_epoch"]=epoch+1;state["bad_epochs"]=0
            state["best_model"]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            state["bad_epochs"]+=1
        scheduler.step()
        record=dict(epoch=epoch+1,train_mse=state["train_sum"]/state["train_count"],validation_mse=validation,
                    best_epoch=state["best_epoch"],best_validation_mse=state["best_mse"],
                    bad_epochs=state["bad_epochs"],learning_rate_next=optimizer.param_groups[0]["lr"],
                    updates=state["updates"],complex_observation_exposures=state["exposures"])
        state["history"].append(record)
        state["epoch"]+=1;state["next_batch"]=0;state["train_sum"]=0.;state["train_count"]=0
        checkpoint()
        save(folder/"best.pt",best_checkpoint(dict(identity=identity,**state)))
        write(folder/"history.json",state["history"])
        log("TRAIN_EPOCH_COMPLETE",method=method,seed=seed,**record)
    save(folder/"best.pt",best_checkpoint(dict(identity=identity,**state)))
    completion=dict(status="TRAINING_COMPLETE",identity=identity,epochs=state["epoch"],maximum_epochs=cap,
                    early_stopping_patience=config["early_stopping_patience"],updates=state["updates"],
                    complex_observation_exposures=state["exposures"],best_epoch=state["best_epoch"],
                    best_validation_mse=state["best_mse"],elapsed_seconds=state["elapsed_seconds"],
                    stop_reason="patience" if state["bad_epochs"]>=config["early_stopping_patience"] else "maximum_epochs",
                    active_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    train_observations_per_epoch=len(train),selection_observations=len(selection),
                    best_checkpoint_sha256=sha(folder/"best.pt"),final_test_accessed=False)
    write(completion_path,completion)
    log("TRAINING_RUN_COMPLETE",method=method,seed=seed,epochs=state["epoch"],best_epoch=state["best_epoch"])
    del model,optimizer,scheduler,train,selection
    gc.collect()
    if device.type=="cuda":
        torch.cuda.empty_cache()
    return completion

def main():
    from contract import validate
    from pipeline import calibrate,cache_features,preflight,verify_training_cache,fit_alpha_and_freeze
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','check','prepare','train','status'))
    parser.add_argument('--data-root',type=Path);parser.add_argument('--output-root',type=Path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--generated-data',action='store_true')
    args=parser.parse_args();config=read(CONFIG_ROOT/'training.json');validate(config);source_identity(config)
    config['data_root']=str((args.data_root or Path(config['data_root'])).resolve())
    output=(args.output_root or HERE/config['output_root']).resolve()
    if output==Path(config['data_root']) or Path(config['data_root']) in output.parents:raise ValueError('Output must be outside dataset')
    if args.smoke:
        if 'smoke' not in output.name.lower():output=output.with_name(output.name+'_smoke')
        config=copy.deepcopy(config);config.update(smoke=True,batch_size=2,training_seeds=config['training_seeds'][:1],checkpoint_every_updates=1)
        config['maximum_epochs']={k:1 for k in config['maximum_epochs']}
        config['calibration'].update(noise_scales=[0.125,1.],delay_scales=[1.],doppler_scales=[1.],maximum_boundary_expansions=0)
        config['evaluation'].update(channels=6,bootstrap_repeats=20)
    if args.generated_data:
        from scripts.dataset_generation import validate_generated_binding
        protocol,manifest=validate_generated_binding(Path(config['data_root']),allow_smoke=args.smoke)
        config['dataset_origin']='generated'
        config['dataset_manifest_sha256']=sha(Path(config['data_root'])/'manifest.json')
        config['generated_protocol_sha256']=sha(Path(config['data_root'])/'protocol.json')
        config['alpha']['channels']=protocol['validation_partition']['alpha_channels']
    configure_limits(config);config['output_root']=str(output)
    if args.stage=='plan':
        log('PLAN',experiment=config['experiment_id'],training_runs=len(METHODS)*len(config['training_seeds']),methods=list(METHODS),
            source_shared_layouts=['A','C'],specialist_layouts=[],batch_size=config['batch_size'],maximum_epochs=config['maximum_epochs'],
            patience=config['early_stopping_patience'],training_seeds=config['training_seeds'],loss=config['loss'],
            train_channels=8 if args.smoke else 100000,calibration_channels=6 if args.smoke else 5000,
            selection_channels=6 if args.smoke else 8000,alpha_channels=4 if args.smoke else 2000,
            data_root=config['data_root'],data_exists=Path(config['data_root']).is_dir(),output=str(output),
            evaluation_scope=config['evaluation']['scope'],smoke=args.smoke)
        return
    if args.stage=='status':
        for seed,method in itertools.product(config['training_seeds'],METHODS):
            path=output/'training'/str(seed)/method/'completion.json'
            log('RUN_STATUS',seed=seed,method=method,complete=path.exists())
        return
    device=torch.device(args.device);configure(config,device)
    if args.stage in ('train','all') and device.type!='cuda' and not args.smoke:raise RuntimeError('Formal training requires GPU; --smoke --device cpu is available')
    with exclusive(output):
        binding=audit_data(config,output)
        if args.stage=='check':preflight(config,output,device);return
        if args.stage in ('train','all'):preflight(config,output,device)
        if args.stage in ('prepare','train','all'):
            calibration=calibrate(config,output,binding);cache_features(config,output,binding,calibration)
            verify_training_cache(config,output,binding);log('PREPARATION_COMPLETE',scope='SMOKE' if args.smoke else 'FORMAL')
        if args.stage in ('train','all'):
            for seed,method in itertools.product(config['training_seeds'],METHODS):train_one(method,seed,config,output,binding,device)
            fit_alpha_and_freeze(config,output,binding,device)

if __name__=='__main__':main()
