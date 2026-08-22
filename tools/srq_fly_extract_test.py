"""Extract a held-out test feature cache only after SRQ-FLY authorization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbone import load_model
from tools.srq_fly_heldout import _read_manifest, _sha256_file, _validate_authorization
from utils.data_utils import load_dataset
from utils.train_utils import feature_extract, random_initialization


def _atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def extract_test(args) -> dict:
    manifest_path = Path(args.manifest).resolve()
    manifest = _read_manifest(manifest_path)
    authorization = _validate_authorization(Path(args.authorization).resolve(), manifest_path)
    dataset = manifest["datasets"][args.dataset_key]
    backbone_contract = manifest["backbone"]
    cache_dir = Path(args.feature_cache_dir).resolve()
    metadata_path, train_path, test_path = cache_dir / "metadata.json", cache_dir / "train.pt", cache_dir / "test.pt"
    if not metadata_path.is_file() or not train_path.is_file():
        raise FileNotFoundError("train-only cache must exist before held-out extraction")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("dataset") != dataset["dataset"]
        or metadata.get("backbone_model") != backbone_contract["model_name"]
        or metadata.get("checkpoint_sha256") != backbone_contract["checkpoint_sha256"]
        or metadata.get("preprocessing") != backbone_contract["preprocessing"]
        or metadata.get("test_features_materialized") not in {False, True, None}
    ):
        raise ValueError("train-only cache metadata does not match the held-out manifest")
    if test_path.is_file():
        test = torch.load(test_path, weights_only=True, map_location="cpu")
        if (
            tuple(test.get("features", torch.empty(0)).shape) != (dataset["test_samples"], backbone_contract["feature_dim"])
            or tuple(test.get("labels", torch.empty(0)).shape) != (dataset["test_samples"],)
            or not bool(torch.isfinite(test["features"]).all())
        ):
            raise ValueError("existing held-out test cache is invalid")
        print(f"TEST CACHE restored dataset={args.dataset_key} samples={len(test['labels'])}", flush=True)
        return {"status": "restored", "test_sha256": _sha256_file(test_path)}

    random_initialization(2025)
    namespace = argparse.Namespace(
        dataset=dataset["dataset"], root=args.root, num_classes=dataset["num_classes"],
        num_tasks=dataset["num_tasks"], batch_size=args.batch_size,
        data_augmentation=backbone_contract["preprocessing"], num_workers=args.num_workers,
    )
    _, test_loaders = load_dataset(namespace)
    device = torch.device(args.device)
    backbone = load_model(
        backbone_contract["model_name"], checkpoint_path=args.backbone_checkpoint,
        expected_checkpoint_size=backbone_contract["checkpoint_size"],
        expected_checkpoint_sha256=backbone_contract["checkpoint_sha256"],
    ).eval().to(device)
    features, labels = [], []
    for task in range(dataset["num_tasks"]):
        values, targets = feature_extract(backbone, test_loaders[task], device)
        features.append(values.cpu()); labels.append(targets.cpu())
        print(
            f"HELDOUT EXTRACT dataset={args.dataset_key} task={task+1}/{dataset['num_tasks']} "
            f"samples={len(targets)}",
            flush=True,
        )
    packed = {"features": torch.cat(features), "labels": torch.cat(labels)}
    if (
        tuple(packed["features"].shape) != (dataset["test_samples"], backbone_contract["feature_dim"])
        or tuple(packed["labels"].shape) != (dataset["test_samples"],)
        or not bool(torch.isfinite(packed["features"]).all())
        or sorted(map(int, torch.unique(packed["labels"]).tolist())) != list(range(dataset["num_classes"]))
    ):
        raise ValueError("extracted held-out tensor contract mismatch")
    _atomic_torch(test_path, packed)
    metadata.update({
        "test_shape": list(packed["features"].shape),
        "test_labels_shape": list(packed["labels"].shape),
        "test_features_materialized": True,
        "finite": bool(metadata.get("finite") and torch.isfinite(packed["features"]).all()),
        "heldout_authorization_id": authorization["authorization_id"],
        "test_sha256": _sha256_file(test_path),
    })
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)
    print(f"TEST CACHE complete dataset={args.dataset_key} shape={tuple(packed['features'].shape)}", flush=True)
    return {"status": "complete", "test_sha256": metadata["test_sha256"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-key", choices=("cifar100", "cub200", "imagenetr"), required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None):
    extract_test(parse_args(argv))


if __name__ == "__main__":
    main()
