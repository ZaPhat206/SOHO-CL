"""Resumable cache-based T-SOHO evaluator; full feature extraction belongs on Kaggle."""
import argparse, csv, json, random, subprocess, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from methods.t_soho import create_learner
from models.backbone import load_model
from utils.data_utils import load_dataset
from utils.train_utils import feature_extract, random_initialization

METHODS=["raw_ridge","random_orthogonal_code","truncated_simplex_code","spectral_confusion_code"]
SCHEMA_VERSION=1
def dump(path,x): Path(path).write_text(json.dumps(x,indent=2,default=str),encoding="utf-8")
def paths(d):
 d=Path(d); return d/"metadata.json",d/"train.pt",d/"test.pt"
def save_cache(d,train_x,train_y,test_x,test_y,args,checkpoint_hash=None):
 d=Path(d);d.mkdir(parents=True,exist_ok=True);m,t,v=paths(d);torch.save({"features":train_x.cpu(),"labels":train_y.cpu()},t);torch.save({"features":test_x.cpu(),"labels":test_y.cpu()},v)
 try: commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 except Exception: commit=None
 dump(m,{"schema_version":SCHEMA_VERSION,"dataset":args.dataset,"dataset_version":"torchvision-cifar100","backbone_model":args.model_name,"checkpoint_sha256":checkpoint_hash,"preprocessing":args.data_augmentation,"resolved_data_config":{"input_size":[3,224,224],"normalization":"vit"},"feature_dim":int(train_x.shape[1]),"dtype":str(train_x.dtype),"split_sizes":{"train":int(train_x.shape[0]),"test":int(test_x.shape[0])},"train_shape":list(train_x.shape),"test_shape":list(test_x.shape),"train_labels_shape":list(train_y.shape),"test_labels_shape":list(test_y.shape),"finite":bool(torch.isfinite(train_x).all() and torch.isfinite(test_x).all()),"git_commit":commit})
def validate_cache(d,args):
 m,t,v=paths(d)
 if not all(x.is_file() for x in (m,t,v)): raise FileNotFoundError("cache requires metadata.json, train.pt, test.pt")
 meta=json.loads(m.read_text()); train,test=torch.load(t,weights_only=True),torch.load(v,weights_only=True)
 if meta.get("schema_version")!=SCHEMA_VERSION or meta.get("dataset")!=args.dataset or meta.get("backbone_model")!=args.model_name: raise ValueError("cache metadata mismatch")
 for s in (train,test):
  if set(s)!={"features","labels"} or s["features"].ndim!=2 or s["labels"].ndim!=1 or s["features"].shape[0]!=s["labels"].shape[0] or not bool(torch.isfinite(s["features"]).all()): raise ValueError("invalid cache")
 return train,test,meta
def split(labels,order,tasks):
 n=len(order)//tasks; return [torch.isin(labels,torch.tensor(order[i*n:(i+1)*n])).nonzero().flatten() for i in range(tasks)]
def run(args):
 train,test,meta=validate_cache(args.feature_cache_dir,args); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
 progress=out/"progress.pt"
 if args.resume and (out/"run.log").is_file() and (out/"run.log").read_text().strip()=="completed": print("resume: completed run already present; skipping"); return
 rng=random.Random(args.seed); order=rng.sample(list(range(args.num_classes)),args.num_classes); dump(out/"config.json",vars(args)); dump(out/"class_order.json",order); dump(out/"environment.json",{"torch":torch.__version__,"device":args.device,"cache_metadata":meta})
 tr,te=split(train["labels"],order,args.num_tasks),split(test["labels"],order,args.num_tasks); l=create_learner(method=args.method,feature_dim=train["features"].shape[1],ridge_lambda=args.ridge_lambda,requested_rank=args.rank,seed=args.seed,device=args.device); matrix=[]; states=[]; timing=[]; diag=[]; start_task=0
 if args.resume and progress.is_file():
  saved=torch.load(progress,weights_only=False); l.load_state_dict(saved["learner"]); matrix,states,timing,diag=saved["matrix"],saved["states"],saved["timing"],saved["diagnostics"]; start_task=saved["next_task"]
 for task in range(start_task,args.num_tasks):
  ix=tr[task]
  start=time.perf_counter(); l.update(train["features"][ix],train["labels"][ix]); timing.append({"task":task,"update_seconds":time.perf_counter()-start}); row=[]
  for old in range(task+1):
   start=time.perf_counter(); logits=l.predict_logits(test["features"][te[old]]); pred=torch.tensor([l.class_ids[i] for i in logits.argmax(1).tolist()]); row.append(float((pred==test["labels"][te[old]]).float().mean()*100)); timing.append({"task":task,"eval_task":old,"inference_seconds":time.perf_counter()-start})
  matrix.append(row); states.append({"task":task,"persistent_state_bytes":l.persistent_state_bytes()}); diag.append({"task":task,"effective_rank":l.diagnostics.get("effective_rank"),"tau":l.diagnostics.get("tau"),"eigenvalues":l.diagnostics.get("eigenvalues").tolist() if isinstance(l.diagnostics.get("eigenvalues"),torch.Tensor) else None})
  torch.save({"next_task":task+1,"learner":l.state_dict(),"matrix":matrix,"states":states,"timing":timing,"diagnostics":diag},progress)
 def writecsv(name,rows):
  fields=sorted({k for r in rows for k in r});
  with (out/name).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 with (out/"accuracy_matrix.csv").open("w",newline="") as f: csv.writer(f).writerows(matrix)
 final=[matrix[-1][i] for i in range(args.num_tasks)]; means=[sum(r)/len(r) for r in matrix]; forgetting=sum(max(matrix[i][i:])-matrix[-1][i] for i in range(args.num_tasks-1))/max(args.num_tasks-1,1)
 writecsv("task_accuracies.csv",[{"task":i,"accuracy":v} for i,v in enumerate(final)]);writecsv("state_bytes.csv",states);writecsv("timing.csv",timing);dump(out/"code_diagnostics.json",diag);dump(out/"metrics.json",{"final_accuracy":sum(final)/len(final),"average_incremental_accuracy":sum(means)/len(means),"forgetting":forgetting,"persistent_state_bytes":l.persistent_state_bytes(),"feature_cache_disk_bytes":sum(x.stat().st_size for x in paths(args.feature_cache_dir))});(out/"run.log").write_text("completed\n"); progress.unlink(missing_ok=True)
def extract(args):
 """Kaggle path: write only disk cache, never learner state."""
 random_initialization(args.seed); device=torch.device(args.device); train_loaders,test_loaders=load_dataset(args)
 backbone=load_model(args.model_name,checkpoint_path=args.backbone_checkpoint,expected_checkpoint_size=args.backbone_checkpoint_size,expected_checkpoint_sha256=args.backbone_checkpoint_sha256).eval().to(device)
 train_x=[];train_y=[];test_x=[];test_y=[]
 for task in range(args.num_tasks):
  x,y=feature_extract(backbone,train_loaders[task],device);train_x.append(x.cpu());train_y.append(y.cpu())
  x,y=feature_extract(backbone,test_loaders[task],device);test_x.append(x.cpu());test_y.append(y.cpu())
 checkpoint_hash=backbone.checkpoint_verification["sha256"] if args.backbone_checkpoint else None
 class A: pass
 meta=A();meta.dataset=args.dataset;meta.model_name=args.model_name;meta.data_augmentation=args.data_augmentation
 save_cache(args.feature_cache_dir,torch.cat(train_x),torch.cat(train_y),torch.cat(test_x),torch.cat(test_y),meta,checkpoint_hash)
def tiny(args):
 torch.manual_seed(args.seed); x=torch.randn(30,8); y=torch.tensor([0,1,2]*10);args.dataset="tiny";args.model_name="synthetic";args.data_augmentation="none";args.num_classes=3;args.num_tasks=3;save_cache(args.feature_cache_dir,x[:21],y[:21],x[21:],y[21:],args);run(args)
def main():
 p=argparse.ArgumentParser();p.add_argument("--method",choices=METHODS,default="spectral_confusion_code");p.add_argument("--rank",type=int,default=8);p.add_argument("--ridge-lambda",type=float,default=1.);p.add_argument("--seed",type=int,default=1993);p.add_argument("--feature-cache-dir",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--dataset",default="CIFAR-100");p.add_argument("--model-name",default="vit_base_patch16_224");p.add_argument("--root");p.add_argument("--backbone-checkpoint");p.add_argument("--backbone-checkpoint-size",type=int);p.add_argument("--backbone-checkpoint-sha256");p.add_argument("--data-augmentation",default="vit");p.add_argument("--num-classes",type=int,default=100);p.add_argument("--num-tasks",type=int,default=10);p.add_argument("--device",default="cpu");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--num-workers",type=int,default=8);p.add_argument("--resume",action="store_true");p.add_argument("--tiny-synthetic",action="store_true");p.add_argument("--extract-features-only",action="store_true");a=p.parse_args();tiny(a) if a.tiny_synthetic else extract(a) if a.extract_features_only else run(a)
if __name__=="__main__":main()
