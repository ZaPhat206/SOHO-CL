"""Load one real CIFAR batch and a verified local frozen backbone; no training."""

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import timm
import torch
import torchvision

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.backbone import load_model
from utils.data_utils import build_transform, load_dataset
from utils.train_utils import random_initialization


def main():
    parser = argparse.ArgumentParser(description="Local-checkpoint backbone preflight")
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-size", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    random_initialization(args.seed)
    class_order = random.sample(list(range(100)), 100)
    dataset_args = SimpleNamespace(
        dataset="CIFAR-100",
        root=args.root,
        num_classes=100,
        num_tasks=10,
        batch_size=args.batch_size,
        data_augmentation="vit",
    )
    train_loaders, test_loaders = load_dataset(dataset_args)
    images, labels = next(iter(train_loaders[0]))
    model = load_model(
        "vit_base_patch16_224",
        checkpoint_path=args.checkpoint,
        expected_checkpoint_size=args.checkpoint_size,
        expected_checkpoint_sha256=args.checkpoint_sha256,
    ).eval()
    with torch.no_grad():
        features = model(images[:2])

    train_dataset = train_loaders[0].dataset.dataset
    print("checkpoint_verification=", model.checkpoint_verification)
    print("model_architecture=", model.__class__.__name__)
    print("timm_model_name=vit_base_patch16_224")
    print("parameter_count_before_head_removal=", sum(parameter.numel() for parameter in model.parameters()))
    print("missing_keys=", model.checkpoint_load_info["missing_keys"])
    print("unexpected_keys=", model.checkpoint_load_info["unexpected_keys"])
    print("classifier_handling=reset_classifier(0) after strict checkpoint load")
    print("pretrained_cfg=", model.pretrained_cfg)
    print("preprocessing=", build_transform(is_cifar=True, data_augmentation="vit"))
    print("train_samples=", len(train_dataset), "test_samples=", len(test_loaders[0].dataset.dataset))
    print("class_mapping=", train_dataset.class_to_idx)
    print("class_order=", class_order)
    print("task0_classes=", class_order[:10])
    print("batch_shape=", tuple(images.shape), "batch_dtype=", images.dtype, "label_dtype=", labels.dtype)
    print("feature_shape=", tuple(features.shape), "feature_dtype=", features.dtype)
    print("feature_min=", features.min().item(), "feature_max=", features.max().item())
    print("feature_mean=", features.mean().item(), "feature_std=", features.std().item())
    print("feature_finite_ratio=", torch.isfinite(features).float().mean().item())
    print("environment=", {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__, "device": "cpu"})


if __name__ == "__main__":
    main()
