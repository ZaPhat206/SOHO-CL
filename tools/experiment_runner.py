"""Resumable cache-based T-SOHO evaluator; full feature extraction belongs on Kaggle."""
import argparse, csv, json, random, subprocess, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from methods.t_soho import create_learner as create_tsoho_learner
from methods.sft_cl import METHODS as SFT_METHODS, create_learner as create_sft_learner
from methods.crt_soho import METHODS as CRT_METHODS, create_learner as create_crt_learner
from methods.cached_replay_baselines import (
    CachedFlyCL,
    CachedFlyCLFidelity,
    CachedSOHOReplay,
    CachedSOHOReplayFidelity,
)
from models.backbone import load_model
from utils.data_utils import load_dataset
from utils.train_utils import feature_extract, random_initialization

TSOHO_METHODS={"raw_ridge","random_orthogonal_code","truncated_simplex_code","spectral_confusion_code"}
# `raw_ridge` remains the historical T-SOHO identity-code spelling so existing
# notebooks/results can resume unchanged.  `sft_raw_ridge` is the minimal
# sufficient-statistic raw-feature control used by the new Fisher ablation.
SFT_CACHE_METHODS={"sft_raw_ridge", *SFT_METHODS-{"raw_ridge"}}
CRT_CACHE_METHODS={f"crt_{method}" for method in CRT_METHODS}
CACHE_REFERENCE_METHODS={
 "cached_flycl", "cached_soho_replay",
 "cached_flycl_fidelity", "cached_soho_replay_fidelity",
}
METHODS=sorted(TSOHO_METHODS | SFT_CACHE_METHODS | CRT_CACHE_METHODS | CACHE_REFERENCE_METHODS)
SCHEMA_VERSION=1


def create_cached_learner(args, method, feature_dim, ridge_lambda, rank, kappa=None, delta=None, residual_ridge=None, complement_ridge=None, confusion_temperature=None):
 """Dispatch cache learners without changing legacy T-SOHO semantics."""
 if method in SFT_CACHE_METHODS:
  sft_method="raw_ridge" if method=="sft_raw_ridge" else method
  return create_sft_learner(method=sft_method,feature_dim=feature_dim,ridge_lambda=ridge_lambda,requested_rank=rank,
                            kappa=float(getattr(args,"fisher_kappa",1.0) if kappa is None else kappa),
                            delta=float(getattr(args,"fisher_delta",.1) if delta is None else delta),
                            scatter_epsilon=float(getattr(args,"fisher_scatter_epsilon",1e-4)),
                            seed=args.seed,device=args.device)
 if method=="cached_flycl":
  return CachedFlyCL(feature_dim=feature_dim,expand_dim=int(getattr(args,"fly_expand_dim",10000)),synaptic_degree=int(getattr(args,"fly_synaptic_degree",300)),
                     coding_level=float(getattr(args,"fly_coding_level",.3)),ridge_lambda=ridge_lambda,seed=args.seed,device=args.device)
 if method=="cached_soho_replay":
  return CachedSOHOReplay(feature_dim=feature_dim,expand_dim=int(getattr(args,"soho_expand_dim",10000)),density=float(getattr(args,"soho_density",.1)),
                           olda_dim=int(getattr(args,"soho_olda_dim",None) or feature_dim),use_etf=not bool(getattr(args,"soho_no_etf",False)),
                           coding_level=float(getattr(args,"soho_coding_level",.45)),ridge_lambda=ridge_lambda,seed=args.seed,device=args.device)
 if method=="cached_flycl_fidelity":
  return CachedFlyCLFidelity(feature_dim=feature_dim,expand_dim=int(getattr(args,"fly_expand_dim",10000)),
                            synaptic_degree=int(getattr(args,"fly_synaptic_degree",300)),coding_level=float(getattr(args,"fly_coding_level",.3)),
                            num_classes=args.num_classes,ridge_lower=float(getattr(args,"fly_ridge_lower",6)),
                            ridge_upper=float(getattr(args,"fly_ridge_upper",10)),seed=args.seed,device=args.device)
 if method=="cached_soho_replay_fidelity":
  return CachedSOHOReplayFidelity(feature_dim=feature_dim,expand_dim=int(getattr(args,"soho_expand_dim",10000)),
                                 density=float(getattr(args,"soho_density",.1)),olda_dim=int(getattr(args,"soho_olda_dim",None) or feature_dim),
                                 use_etf=not bool(getattr(args,"soho_no_etf",False)),coding_level=float(getattr(args,"soho_coding_level",.25)),
                                 num_classes=args.num_classes,ridge_lower=float(getattr(args,"soho_ridge_lower",-2)),
                                 ridge_upper=float(getattr(args,"soho_ridge_upper",10)),seed=args.seed,device=args.device,
                                 replay_chunk_size=int(getattr(args,"soho_replay_chunk_size",2000)),
                                 gcv_sample_size=int(getattr(args,"soho_gcv_sample_size",3000)))
 if method in CRT_CACHE_METHODS:
  dtype={"float32":torch.float32,"float64":torch.float64}[getattr(args,"crt_statistics_dtype","float64")]
  return create_crt_learner(method=method.removeprefix("crt_"),raw_dim=feature_dim,
                            anchor_dim=int(getattr(args,"crt_anchor_dim",2048)),synaptic_degree=int(getattr(args,"crt_synaptic_degree",300)),
                            coding_level=float(getattr(args,"crt_coding_level",.3)),anchor_ridge=ridge_lambda,
                            residual_ridge=float(getattr(args,"crt_residual_ridge",1.) if residual_ridge is None else residual_ridge),
                            complement_ridge=float(getattr(args,"crt_complement_ridge",1.) if complement_ridge is None else complement_ridge),
                            requested_rank=max(int(rank),1),
                            confusion_temperature=float(getattr(args,"crt_confusion_temperature",1.) if confusion_temperature is None else confusion_temperature),
                            scatter_epsilon=float(getattr(args,"crt_scatter_epsilon",1e-4)),seed=args.seed,device=args.device,dtype=dtype)
 return create_tsoho_learner(method=method,feature_dim=feature_dim,ridge_lambda=ridge_lambda,requested_rank=rank,seed=args.seed,device=args.device)
def dump(path,x): Path(path).write_text(json.dumps(x,indent=2,default=str),encoding="utf-8")
def paths(d):
 d=Path(d); return d/"metadata.json",d/"train.pt",d/"test.pt"
def forgetting_from_matrix(matrix):
 """Mean max-past-accuracy drop for every task except the final one."""
 if len(matrix)<=1: return 0.0
 final=matrix[-1]
 return sum(max(matrix[stage][class_task] for stage in range(class_task,len(matrix)))-final[class_task] for class_task in range(len(matrix)-1))/max(len(matrix)-1,1)
def save_cache(d,train_x,train_y,test_x,test_y,args,checkpoint_hash=None):
 d=Path(d);d.mkdir(parents=True,exist_ok=True);m,t,v=paths(d);torch.save({"features":train_x.cpu(),"labels":train_y.cpu()},t);torch.save({"features":test_x.cpu(),"labels":test_y.cpu()},v)
 try: commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 except Exception: commit=None
 dataset_version={"CIFAR-100":"torchvision-cifar100","CUB-200-2011":"processed-imagefolder"}.get(args.dataset,"unspecified")
 dump(m,{"schema_version":SCHEMA_VERSION,"dataset":args.dataset,"dataset_version":dataset_version,"backbone_model":args.model_name,"checkpoint_sha256":checkpoint_hash,"preprocessing":args.data_augmentation,"resolved_data_config":{"input_size":[3,224,224],"normalization":args.data_augmentation},"feature_dim":int(train_x.shape[1]),"dtype":str(train_x.dtype),"split_sizes":{"train":int(train_x.shape[0]),"test":int(test_x.shape[0])},"train_shape":list(train_x.shape),"test_shape":list(test_x.shape),"train_labels_shape":list(train_y.shape),"test_labels_shape":list(test_y.shape),"finite":bool(torch.isfinite(train_x).all() and torch.isfinite(test_x).all()),"git_commit":commit})
def save_train_cache(d,train_x,train_y,args,expected_test_samples,checkpoint_hash=None):
 """Write a selection cache without materializing or opening held-out images."""
 d=Path(d);d.mkdir(parents=True,exist_ok=True);m,t,_=paths(d);torch.save({"features":train_x.cpu(),"labels":train_y.cpu()},t)
 try: commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 except Exception: commit=None
 dataset_version={"CIFAR-100":"torchvision-cifar100","CUB-200-2011":"processed-imagefolder"}.get(args.dataset,"unspecified")
 dump(m,{"schema_version":SCHEMA_VERSION,"dataset":args.dataset,"dataset_version":dataset_version,"backbone_model":args.model_name,"checkpoint_sha256":checkpoint_hash,"preprocessing":args.data_augmentation,"resolved_data_config":{"input_size":[3,224,224],"normalization":args.data_augmentation},"feature_dim":int(train_x.shape[1]),"dtype":str(train_x.dtype),"split_sizes":{"train":int(train_x.shape[0]),"test":int(expected_test_samples)},"train_shape":list(train_x.shape),"test_shape":None,"train_labels_shape":list(train_y.shape),"test_labels_shape":None,"finite":bool(torch.isfinite(train_x).all()),"test_features_materialized":False,"git_commit":commit})
def validate_cache(d,args,load_test=True):
 """Validate a feature cache, optionally without opening held-out test data."""
 m,t,v=paths(d)
 required=(m,t,v) if load_test else (m,t)
 if not all(x.is_file() for x in required): raise FileNotFoundError("cache requires metadata.json, train.pt" + (", test.pt" if load_test else ""))
 meta=json.loads(m.read_text()); train=torch.load(t,weights_only=True);test=torch.load(v,weights_only=True) if load_test else None
 if meta.get("schema_version")!=SCHEMA_VERSION or meta.get("dataset")!=args.dataset or meta.get("backbone_model")!=args.model_name: raise ValueError("cache metadata mismatch")
 for s in (train,) if test is None else (train,test):
  if set(s)!={"features","labels"} or s["features"].ndim!=2 or s["labels"].ndim!=1 or s["features"].shape[0]!=s["labels"].shape[0] or not bool(torch.isfinite(s["features"]).all()): raise ValueError("invalid cache")
 return train,test,meta
def split(labels,order,tasks):
 n=len(order)//tasks; return [torch.isin(labels,torch.tensor(order[i*n:(i+1)*n])).nonzero().flatten() for i in range(tasks)]
def run(args):
 train,test,meta=validate_cache(args.feature_cache_dir,args); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
 progress=out/"progress.pt"
 if args.resume and (out/"run.log").is_file() and (out/"run.log").read_text().strip()=="completed": print("resume: completed run already present; skipping"); return
 rng=random.Random(args.seed); order=rng.sample(list(range(args.num_classes)),args.num_classes); dump(out/"config.json",vars(args)); dump(out/"class_order.json",order); dump(out/"environment.json",{"torch":torch.__version__,"device":args.device,"cache_metadata":meta})
 tr,te=split(train["labels"],order,args.num_tasks),split(test["labels"],order,args.num_tasks); l=create_cached_learner(args,args.method,train["features"].shape[1],args.ridge_lambda,args.rank); matrix=[]; states=[]; timing=[]; diag=[]; start_task=0
 if args.resume and progress.is_file():
  saved=torch.load(progress,weights_only=False); l.load_state_dict(saved["learner"]); matrix,states,timing,diag=saved["matrix"],saved["states"],saved["timing"],saved["diagnostics"]; start_task=saved["next_task"]
 for task in range(start_task,args.num_tasks):
  ix=tr[task]
  start=time.perf_counter(); l.update(train["features"][ix],train["labels"][ix]); timing.append({"task":task,"update_seconds":time.perf_counter()-start}); row=[]
  for old in range(task+1):
   start=time.perf_counter(); logits=l.predict_logits(test["features"][te[old]]); pred=torch.tensor([l.class_ids[i] for i in logits.argmax(1).tolist()]); row.append(float((pred==test["labels"][te[old]]).float().mean()*100)); timing.append({"task":task,"eval_task":old,"inference_seconds":time.perf_counter()-start})
  matrix.append(row); states.append({"task":task,"persistent_state_bytes":l.persistent_state_bytes()}); diag.append({"task":task,"requested_rank":l.diagnostics.get("requested_rank"),"effective_rank":l.diagnostics.get("effective_rank"),"selected_ridge":l.diagnostics.get("selected_ridge"),"ridge_policy":l.diagnostics.get("ridge_policy"),"replay_required":l.diagnostics.get("replay_required"),"retained_sample_count":l.diagnostics.get("retained_sample_count"),"tau":l.diagnostics.get("tau"),"eigenvalues":l.diagnostics.get("eigenvalues").tolist() if isinstance(l.diagnostics.get("eigenvalues"),torch.Tensor) else None,"gains":l.diagnostics.get("gains").tolist() if isinstance(l.diagnostics.get("gains"),torch.Tensor) else None,"singular_values":l.diagnostics.get("singular_values").tolist() if isinstance(l.diagnostics.get("singular_values"),torch.Tensor) else None,"retained_correction_energy":l.diagnostics.get("retained_correction_energy"),"solver_residual_max":l.diagnostics.get("solver_residual_max"),"solver_relative_residual_max":l.diagnostics.get("solver_relative_residual_max"),"geometry":l.diagnostics.get("geometry"),"residualized":l.diagnostics.get("residualized"),"affinity_min":l.diagnostics.get("affinity_min"),"affinity_median":l.diagnostics.get("affinity_median"),"affinity_max":l.diagnostics.get("affinity_max"),"affinity_edge_cv":l.diagnostics.get("affinity_edge_cv"),"affinity_normalized_entropy":l.diagnostics.get("affinity_normalized_entropy")})
  torch.save({"next_task":task+1,"learner":l.state_dict(),"matrix":matrix,"states":states,"timing":timing,"diagnostics":diag},progress)
 def writecsv(name,rows):
  fields=sorted({k for r in rows for k in r});
  with (out/name).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 with (out/"accuracy_matrix.csv").open("w",newline="") as f: csv.writer(f).writerows(matrix)
 final=[matrix[-1][i] for i in range(args.num_tasks)]; means=[sum(r)/len(r) for r in matrix]; forgetting=forgetting_from_matrix(matrix)
 writecsv("task_accuracies.csv",[{"task":i,"accuracy":v} for i,v in enumerate(final)]);writecsv("state_bytes.csv",states);writecsv("timing.csv",timing);dump(out/"code_diagnostics.json",diag);dump(out/"metrics.json",{"final_accuracy":sum(final)/len(final),"average_incremental_accuracy":sum(means)/len(means),"forgetting":forgetting,"persistent_state_bytes":l.persistent_state_bytes(),"feature_cache_disk_bytes":sum(x.stat().st_size for x in paths(args.feature_cache_dir)),"exemplar_free":bool(getattr(l,"is_exemplar_free",True))});(out/"run.log").write_text("completed\n"); progress.unlink(missing_ok=True)
def extract(args):
 """Kaggle path: write only disk cache, never learner state."""
 random_initialization(args.seed); device=torch.device(args.device); train_loaders,test_loaders=load_dataset(args)
 backbone=load_model(args.model_name,checkpoint_path=args.backbone_checkpoint,expected_checkpoint_size=args.backbone_checkpoint_size,expected_checkpoint_sha256=args.backbone_checkpoint_sha256).eval().to(device)
 train_x=[];train_y=[];test_x=[];test_y=[]
 for task in range(args.num_tasks):
  x,y=feature_extract(backbone,train_loaders[task],device);train_x.append(x.cpu());train_y.append(y.cpu())
  print(f"feature extraction train task {task + 1}/{args.num_tasks}: {len(y)} samples",flush=True)
  if not getattr(args,"extract_train_only",False):
   x,y=feature_extract(backbone,test_loaders[task],device);test_x.append(x.cpu());test_y.append(y.cpu())
   print(f"feature extraction test task {task + 1}/{args.num_tasks}: {len(y)} samples",flush=True)
 checkpoint_hash=backbone.checkpoint_verification["sha256"] if args.backbone_checkpoint else None
 class A: pass
 meta=A();meta.dataset=args.dataset;meta.model_name=args.model_name;meta.data_augmentation=args.data_augmentation
 if getattr(args,"extract_train_only",False):
  expected_test_samples=sum(len(loader.dataset) for loader in test_loaders.values())
  save_train_cache(args.feature_cache_dir,torch.cat(train_x),torch.cat(train_y),meta,expected_test_samples,checkpoint_hash)
 else:
  save_cache(args.feature_cache_dir,torch.cat(train_x),torch.cat(train_y),torch.cat(test_x),torch.cat(test_y),meta,checkpoint_hash)
def train_validation_indices(labels,task_indices,seed,fraction):
 """Deterministic stratified train/validation split; never reads test features."""
 if not 0 < fraction < 1: raise ValueError("validation_fraction must be in (0,1)")
 generator=torch.Generator().manual_seed(seed); training=[]; validation=[]
 for indices in task_indices:
  task_labels=labels[indices]; train_parts=[]; val_parts=[]
  for cls in torch.unique(task_labels):
   cls_indices=indices[task_labels==cls]; perm=cls_indices[torch.randperm(len(cls_indices),generator=generator)]; n_val=max(1,int(len(perm)*fraction)); val_parts.append(perm[:n_val]);train_parts.append(perm[n_val:])
  training.append(torch.cat(train_parts));validation.append(torch.cat(val_parts))
 return training,validation
def select_config(args):
 """Rank/lambda selection from cached *training* features only; no test access."""
 train,_,meta=validate_cache(args.feature_cache_dir,args,load_test=False); order=random.Random(args.seed).sample(list(range(args.num_classes)),args.num_classes); tasks=split(train["labels"],order,args.num_tasks); train_parts,val_parts=train_validation_indices(train["labels"],tasks,args.seed,args.validation_fraction)
 ranks=[int(x) for x in args.search_ranks.split(",")];lambdas=[float(x) for x in args.search_lambdas.split(",")];kappas=[float(x) for x in getattr(args,"search_kappas",str(getattr(args,"fisher_kappa",1.0))).split(",")];deltas=[float(x) for x in getattr(args,"search_deltas",str(getattr(args,"fisher_delta",.1))).split(",")];residual_ridges=[float(x) for x in getattr(args,"search_crt_residual_ridges","0.01,0.1,1.0").split(",")];complement_ridges=[float(x) for x in getattr(args,"search_crt_complement_ridges","0.01,0.1,1.0").split(",")];temperatures=[float(x) for x in getattr(args,"search_crt_temperatures","0.25,0.5,1.0").split(",")];results=[]
 for method in args.search_methods.split(","):
  if method not in METHODS: raise ValueError(f"unknown search method {method!r}; choices: {METHODS}")
  if method in {"cached_flycl_fidelity","cached_soho_replay_fidelity"}: raise ValueError(f"{method} uses its locked internal GCV policy and cannot enter an external search grid")
  for rank in ([0] if method in {"raw_ridge","sft_raw_ridge","cached_flycl","cached_soho_replay","crt_anchor_only","crt_full_raw_residual"} or method.endswith("soft") else ranks):
   for ridge_lambda in lambdas:
    for kappa in (kappas if method in SFT_CACHE_METHODS and "fisher" in method else [None]):
     for delta in (deltas if method in SFT_CACHE_METHODS and method.endswith("soft") else [None]):
      for residual_ridge in (residual_ridges if method in CRT_CACHE_METHODS and method!="crt_anchor_only" else [None]):
       for complement_ridge in (complement_ridges if method in CRT_CACHE_METHODS and method not in {"crt_anchor_only","crt_confusion_no_residualization"} else [None]):
        for temperature in (temperatures if method in {"crt_confusion_residual","crt_shuffled_confusion_residual","crt_confusion_no_residualization"} else [None]):
         learner=create_cached_learner(args,method,train["features"].shape[1],ridge_lambda,rank,kappa,delta,residual_ridge,complement_ridge,temperature);scores=[]
         for task in range(args.num_tasks):
          learner.update(train["features"][train_parts[task]],train["labels"][train_parts[task]])
          for previous in range(task+1):
           logits=learner.predict_logits(train["features"][val_parts[previous]]);pred=torch.tensor([learner.class_ids[i] for i in logits.argmax(1).tolist()]);scores.append(float((pred==train["labels"][val_parts[previous]]).float().mean()*100))
         results.append({"method":method,"rank":rank,"ridge_lambda":ridge_lambda,"fisher_kappa":kappa,"fisher_delta":delta,"crt_residual_ridge":residual_ridge,"crt_complement_ridge":complement_ridge,"crt_confusion_temperature":temperature,"validation_average_accuracy":sum(scores)/len(scores),"uses_test_set":False})
 best=max(results,key=lambda x:x["validation_average_accuracy"]);out=Path(args.selection_output or Path(args.output_dir)/"selection.json");out.parent.mkdir(parents=True,exist_ok=True);dump(out,{"selection_protocol":"stratified held-out subset of cached training features only","validation_fraction":args.validation_fraction,"cache_metadata":meta,"best":best,"candidates":results});print(json.dumps(best))
def tiny(args):
 torch.manual_seed(args.seed); x=torch.randn(30,8); y=torch.tensor([0,1,2]*10);args.dataset="tiny";args.model_name="synthetic";args.data_augmentation="none";args.num_classes=3;args.num_tasks=3;save_cache(args.feature_cache_dir,x[:21],y[:21],x[21:],y[21:],args);run(args)
def main():
 config_probe=argparse.ArgumentParser(add_help=False);config_probe.add_argument("--config")
 configured,_=config_probe.parse_known_args()
 p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--method",choices=METHODS,default="spectral_confusion_code");p.add_argument("--rank",type=int,default=8);p.add_argument("--ridge-lambda",type=float,default=1.);p.add_argument("--fisher-kappa",type=float,default=1.0);p.add_argument("--fisher-delta",type=float,default=.1);p.add_argument("--fisher-scatter-epsilon",type=float,default=1e-4);p.add_argument("--fly-expand-dim",type=int,default=10000);p.add_argument("--fly-synaptic-degree",type=int,default=300);p.add_argument("--fly-coding-level",type=float,default=.3);p.add_argument("--fly-ridge-lower",type=float,default=6);p.add_argument("--fly-ridge-upper",type=float,default=10);p.add_argument("--soho-expand-dim",type=int,default=10000);p.add_argument("--soho-density",type=float,default=.1);p.add_argument("--soho-olda-dim",type=int);p.add_argument("--soho-coding-level",type=float,default=.45);p.add_argument("--soho-ridge-lower",type=float,default=-2);p.add_argument("--soho-ridge-upper",type=float,default=10);p.add_argument("--soho-replay-chunk-size",type=int,default=2000);p.add_argument("--soho-gcv-sample-size",type=int,default=3000);p.add_argument("--soho-no-etf",action="store_true");p.add_argument("--crt-anchor-dim",type=int,default=2048);p.add_argument("--crt-synaptic-degree",type=int,default=300);p.add_argument("--crt-coding-level",type=float,default=.3);p.add_argument("--crt-residual-ridge",type=float,default=1.);p.add_argument("--crt-complement-ridge",type=float,default=1.);p.add_argument("--crt-confusion-temperature",type=float,default=1.);p.add_argument("--crt-scatter-epsilon",type=float,default=1e-4);p.add_argument("--crt-statistics-dtype",choices=("float32","float64"),default="float64");p.add_argument("--seed",type=int,default=1993);p.add_argument("--feature-cache-dir",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--dataset",default="CIFAR-100");p.add_argument("--model-name",default="vit_base_patch16_224");p.add_argument("--root");p.add_argument("--backbone-checkpoint");p.add_argument("--backbone-checkpoint-size",type=int);p.add_argument("--backbone-checkpoint-sha256");p.add_argument("--data-augmentation",default="vit");p.add_argument("--num-classes",type=int,default=100);p.add_argument("--num-tasks",type=int,default=10);p.add_argument("--device",default="cpu");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--num-workers",type=int,default=8);p.add_argument("--resume",action="store_true");p.add_argument("--tiny-synthetic",action="store_true");p.add_argument("--extract-features-only",action="store_true");p.add_argument("--extract-train-only",action="store_true");p.add_argument("--select-config",action="store_true");p.add_argument("--search-methods",default="spectral_confusion_code");p.add_argument("--search-ranks",default="8,16,32,64");p.add_argument("--search-lambdas",default="0.01,0.1,1.0,10.0");p.add_argument("--search-kappas",default="0.01,0.1,1.0");p.add_argument("--search-deltas",default="0.01,0.1,0.5");p.add_argument("--search-crt-residual-ridges",default="0.01,0.1,1.0");p.add_argument("--search-crt-complement-ridges",default="0.01,0.1,1.0");p.add_argument("--search-crt-temperatures",default="0.25,0.5,1.0");p.add_argument("--validation-fraction",type=float,default=.1);p.add_argument("--selection-output")
 if configured.config:
  payload=json.loads(Path(configured.config).read_text(encoding="utf-8"))
  if not isinstance(payload,dict): raise ValueError("config must be a JSON object")
  unknown=set(payload)-{action.dest for action in p._actions}
  if unknown: raise ValueError(f"unknown config keys: {sorted(unknown)}")
  p.set_defaults(**payload)
 a=p.parse_args();tiny(a) if a.tiny_synthetic else extract(a) if a.extract_features_only else select_config(a) if a.select_config else run(a)
if __name__=="__main__":main()
