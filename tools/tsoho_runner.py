"""Feature-cache and experiment runner; full datasets are intended for Kaggle."""
import argparse, csv, json, os, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from methods.t_soho import create_learner

SCHEMA_VERSION = 1

def _json(path, value): Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
def tiny(args):
    torch.manual_seed(args.seed); x=torch.randn(24,8); y=torch.tensor([0,1,2]*8); learner=create_learner(method=args.method, feature_dim=8, ridge_lambda=args.ridge_lambda, requested_rank=args.rank, seed=args.seed)
    learner.update(x[:12],y[:12]); learner.update(x[12:],y[12:]); print(json.dumps({"logits_shape":list(learner.predict_logits(x[:3]).shape),"state_bytes":learner.persistent_state_bytes(),"classes":learner.class_ids}))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--method", choices=["raw_ridge","random_orthogonal_code","truncated_simplex_code","spectral_confusion_code"], default="spectral_confusion_code"); parser.add_argument("--rank",type=int,default=8); parser.add_argument("--ridge-lambda",type=float,default=1.0); parser.add_argument("--seed",type=int,default=1993); parser.add_argument("--feature-cache-dir"); parser.add_argument("--output-dir"); parser.add_argument("--root"); parser.add_argument("--backbone-checkpoint"); parser.add_argument("--num-tasks",type=int,default=10); parser.add_argument("--device",default="cpu"); parser.add_argument("--batch-size",type=int,default=128); parser.add_argument("--num-workers",type=int,default=8); parser.add_argument("--resume",action="store_true"); parser.add_argument("--tiny-synthetic",action="store_true"); parser.add_argument("--extract-features-only",action="store_true")
    args=parser.parse_args()
    if args.tiny_synthetic: return tiny(args)
    raise SystemExit("Full cache extraction/experiment execution is intentionally Kaggle-only; use --tiny-synthetic locally.")
if __name__=="__main__": main()
