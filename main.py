import argparse
import json
import time
import torch
import numpy as np

from utils.data_utils import load_dataset
from utils.train_utils import random_initialization
from utils.metrics import print_accuracy_matrix
from models.backbone import load_model
from models.soho import SOHO
from methods.sohocl import SOHOCL
from methods.flycl import FlyCL
from methods.streaming_raw_ridge import StreamingRawRidge

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SOHO-CL Experiment Pipeline")

    # Continual Learning Task Setting
    parser.add_argument('--method', default='flycl', choices=['flycl', 'sohocl', 'streaming_raw_ridge'], help='CL Method')
    parser.add_argument('--dataset', default='CIFAR-100', help='Choose dataset')
    parser.add_argument('--root', default='../data', help='Dataset path')
    parser.add_argument('--num_classes', type=int, default=100, help='Total number of classes')
    parser.add_argument('--num_tasks', type=int, default=20, help='Number of tasks')

    # model Architecture
    parser.add_argument('--model_name', type=str, default="vit_base_patch16_224", help='model name')
    parser.add_argument('--backbone_checkpoint', type=str, default=None, help='Verified local safetensors backbone checkpoint')
    parser.add_argument('--backbone_checkpoint_size', type=int, default=None, help='Expected local checkpoint size in bytes')
    parser.add_argument('--backbone_checkpoint_sha256', type=str, default=None, help='Expected local checkpoint SHA-256')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension of pre-trained model')
    parser.add_argument('--expand_dim', type=int, default=10000, help='Expansion dimension of FlyHash')
    parser.add_argument('--synaptic_degree', type=int, default=100, help='Number of connections')
    parser.add_argument('--coding_level', type=float, default=0.01, help='Top-k sparsity ratio')
    parser.add_argument('--density', type=float, default=0.3, help='Density of Sparse Rademacher Matrix')
    parser.add_argument('--olda_dim', type=int, default=768, help='Output dimension of OLDA')
    parser.add_argument('--no_etf', action='store_true', help='Disable ETF Procrustes Alignment')

    # Training Configuration
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--ridge_lower', type=float, default=4, help='lower bound for ridge coefficient (log10)')
    parser.add_argument('--ridge_upper', type=float, default=10, help='upper bound for ridge coefficient (log10)')
    parser.add_argument('--auto_ridge', action='store_true',
                        help='FIX Hướng 2: Tự động chọn ridge_lower tối ưu theo dataset thay vì dùng giá trị mặc định.')
    parser.add_argument('--data_augmentation', default=None, help='choose which normalization or not')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='Choose gpu')
    parser.add_argument('--config', type=str, default=None, help='Optional JSON experiment configuration')
    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str)
    config_args, _ = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, encoding='utf-8') as config_file:
            config = json.load(config_file)
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object of CLI argument names and values.")
        parser.set_defaults(**config)
    args = parser.parse_args()

    # FIX Hướng 2: Tự động điều chỉnh ridge_lower theo từng dataset
    # GCV cần tìm lambda trong vùng phù hợp với scale của Gram matrix.
    # CIFAR-100 và CUB có nhiều mẫu/task → Gram matrix lớn → lambda tối ưu cao hơn.
    if args.auto_ridge and args.method == 'sohocl':
        dataset_ridge_map = {
            "CIFAR-100":   2,   # 50k samples, Gram scale cao → cần lambda cao
            "CUB-200-2011": 1,  # 9k samples, moderate
            "ImageNet-R":  -1,  # 30k samples, varied distribution
        }
        auto_lower = dataset_ridge_map.get(args.dataset, args.ridge_lower)
        print(f"[auto_ridge] Override ridge_lower: {args.ridge_lower} → {auto_lower} (dataset={args.dataset})")
        args.ridge_lower = auto_lower
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    random_initialization(args.seed)

    print(f"Loading {args.dataset}...")
    train_loader_dict, test_loader_dict = load_dataset(args)
    print("Dataset loaded and split successfully.")

    print(f"Initializing Backbone ({args.model_name})...")
    backbone = load_model(
        args.model_name,
        checkpoint_path=args.backbone_checkpoint,
        expected_checkpoint_size=args.backbone_checkpoint_size,
        expected_checkpoint_sha256=args.backbone_checkpoint_sha256,
    )
    if args.backbone_checkpoint:
        verification = backbone.checkpoint_verification
        print(f"Local backbone checkpoint: {verification}")
        print(f"Backbone load keys | missing={backbone.checkpoint_load_info['missing_keys']} | unexpected={backbone.checkpoint_load_info['unexpected_keys']}")
    backbone.eval()
    backbone.to(device)

    if args.method == 'flycl':
        print("Initializing FlyHash & FlyCL Agent...")
        from models.flyhash import FlyHash
        flyhash = FlyHash(args.embedding_dim, args.expand_dim, args.synaptic_degree).to(device)
        agent = FlyCL(backbone, flyhash, args.num_classes, args.coding_level, 
                      args.ridge_lower, args.ridge_upper, device)
    elif args.method == 'sohocl':
        print("Initializing SOHO & SOHOCL Agent...")
        # SOHO can output at most in_dim non-zero components from OLDA
        use_etf = not args.no_etf
        soho = SOHO(args.embedding_dim, output_dim=args.expand_dim, device=device, density=args.density, olda_dim=args.olda_dim, use_etf=use_etf)
        agent = SOHOCL(backbone, soho, args.num_classes, args.coding_level, 
                       args.ridge_lower, args.ridge_upper, device)
    else:
        print("Initializing streaming raw-feature Ridge baseline...")
        agent = StreamingRawRidge(backbone, args.embedding_dim, args.ridge_lower, args.ridge_upper, device)

    from utils.metrics import print_accuracy_matrix, print_timing_metrics, compute_memory_footprint
    
    acc = {}
    training_time = []
    feature_extract_time = []
    update_time = []
    inference_time = []
    inference_post_extract_time = []
    persistent_state_bytes = []
    
    print("\n" + "="*50)
    print("🚀 Starting Continual Learning")
    print("="*50)

    for task in range(args.num_tasks):
        acc[task] = []
        
        # Train Task
        print(f"\n[Task {task:02d}] Training...")
        best_lam, ext_time, t_time = agent.train_task(task, train_loader_dict[task])
        training_time.append(t_time)
        feature_extract_time.append(ext_time)
        update_time.append(t_time - ext_time)
        print(f"[Task {task:02d}] Done | train_time={t_time:.2f}s | best_lam={best_lam}")
        if hasattr(agent, 'persistent_state_summary'):
            state = agent.persistent_state_summary()
            persistent_state_bytes.append(state['tensor_bytes'])
            print(f"[Task {task:02d}] Persistent learner state: {state['tensor_bytes']} bytes | {state['tensors']}")

        # Eval Task up to current task
        for sub_task in range(task + 1):
            inference_start = time.time()
            test_acc = agent.eval_task(sub_task, test_loader_dict[sub_task])
            inference_time.append(time.time() - inference_start)
            inference_post_extract_time.append(getattr(agent, 'last_eval_classifier_time', inference_time[-1]))
            acc[sub_task].append(test_acc)
            
    print("\n" + "="*50)
    print("📊 Evaluation Summary")
    print("="*50)
    
    aa = print_accuracy_matrix(acc, args.num_tasks)
    print_timing_metrics(training_time, feature_extract_time)
    print("Update Time (excluding feature extraction)")
    print(", ".join(f"{value:.2f}" for value in update_time))
    print("Inference Time (end-to-end, one evaluated task each)")
    print(", ".join(f"{value:.2f}" for value in inference_time))
    print("Inference Time (post-feature-extraction, one evaluated task each)")
    print(", ".join(f"{value:.2f}" for value in inference_post_extract_time))
    if persistent_state_bytes:
        print("Persistent Learner State Bytes (after each training task)")
        print(", ".join(str(value) for value in persistent_state_bytes))
    compute_memory_footprint(agent)
