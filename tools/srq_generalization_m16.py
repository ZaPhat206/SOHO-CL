"""Locked Stanford Cars / RanPAC Phase-2 confirmation for the SRQ backend."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import (  # noqa: E402
    ExactGramBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from tools import srq_generalization_m5 as m5  # noqa: E402


METHODS = ("exact", "p2b_int8", "adaptive_int8_fp16")
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "uses_test_set",
    "test_tuning_allowed", "accuracy_based_method_selection", "num_classes",
    "class_increments", "expected_train_samples", "expected_test_samples",
    "replicates", "methods", "ranpac_source", "backbone", "ranpac",
    "ridge_selection", "p2b", "adaptive", "evaluation", "integrity_gates",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_git() -> None:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("M16 requires a clean committed checkout")


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M16 config keys/schema mismatch")
    if not (
        config["dataset"] == "Stanford Cars"
        and config["uses_test_set"] is True
        and config["test_tuning_allowed"] is False
        and config["accuracy_based_method_selection"] is False
        and config["num_classes"] == 196
        and config["class_increments"] == [16] + [20] * 9
        and sum(config["class_increments"]) == config["num_classes"]
        and config["expected_train_samples"] == 8144
        and config["expected_test_samples"] == 8041
        and tuple(config["methods"]) == METHODS
    ):
        raise ValueError("M16 frozen scope changed")
    expected_replicates = [
        {
            "class_order_seed": seed,
            "projection_seed": seed,
            "ridge_split_seed": seed,
        }
        for seed in range(1993, 1999)
    ]
    if config["replicates"] != expected_replicates:
        raise ValueError("M16 replicate seeds changed")
    source = config["ranpac_source"]
    if not (
        source["commit"] == "cf4b301d18b0c27db030f4371b72b768005ae58a"
        and source["cars_publish_row"] == 10
        and source["path"] == "phase2_no_petl_random_relu"
        and all(
            len(source[key]) == 64
            for key in (
                "ranpac_py_sha256", "resnet_py_sha256", "data_py_sha256",
                "cars_publish_csv_sha256",
            )
        )
    ):
        raise ValueError("M16 RanPAC source lock changed")
    backbone = config["backbone"]
    if not (
        backbone["architecture"] == "resnet50"
        and backbone["weights"] == "IMAGENET1K_V2"
        and backbone["feature_dimension"] == 2048
        and backbone["checkpoint_size_bytes"] == 102540417
        and backbone["checkpoint_sha256"]
        == "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
    ):
        raise ValueError("M16 backbone identity changed")
    ranpac = config["ranpac"]
    if not (
        ranpac["expand_dimension"] == 10000
        and ranpac["projection_distribution"] == "standard_normal"
        and ranpac["activation"] == "relu"
        and min(ranpac["encode_batch_size"], ranpac["evaluation_batch_size"]) > 0
    ):
        raise ValueError("M16 RanPAC representation changed")
    ridge = config["ridge_selection"]
    candidates = [10.0**power for power in range(-8, 9)]
    if not (
        ridge["policy"] == "first_task_train_only_80_20_mse_then_freeze"
        and ridge["candidate_lambdas"] == candidates
        and ridge["fit_fraction"] == 0.8
        and ridge["metric"] == "mean_squared_error"
        and ridge["solver"] == "mathematically_equivalent_dual_ridge"
        and ridge["tie_break"] == "smallest_lambda"
        and ridge["common_across_backends_within_replicate"] is True
    ):
        raise ValueError("M16 Ridge-selection policy changed")
    p2b = config["p2b"]
    if not (
        p2b["block_size"] == 256
        and p2b["group_size"] == 64
        and p2b["update_panel_size"] == 128
        and p2b["first_update_backend"] == "gram_cholesky"
        and p2b["quantization_backend"] == "streaming"
        and p2b["quantization_batch_blocks"] == 64
    ):
        raise ValueError("M16 P2B identity changed")
    adaptive = config["adaptive"]
    if not (
        adaptive["budget_fraction_between_int8_and_fp16"] == 0.25
        and adaptive["selection_signal"]
        == "current_factor_values_only_no_labels_or_accuracy"
    ):
        raise ValueError("M16 adaptive identity changed")
    evaluation = config["evaluation"]
    gates = config["integrity_gates"]
    if not (
        evaluation["training_split"] == "complete_official_train_split"
        and evaluation["test_split"] == "official_test_split"
        and evaluation["allow_post_test_method_selection"] is False
        and evaluation["allow_post_test_retry"] is False
        and evaluation["confidence_intervals_in_main_table"] is False
        and gates["accuracy_gate"] is None
        and gates["require_all_units_complete"] is True
        and gates["require_compressed_state_smaller_than_exact"] is True
        and float(gates["maximum_solver_relative_residual"]) > 0
    ):
        raise ValueError("M16 evaluation/integrity policy changed")
    return config


def _split_by_increments(
    labels: torch.Tensor, order: list[int], increments: list[int]
) -> list[torch.Tensor]:
    if labels.ndim != 1 or sum(increments) != len(order) or len(set(order)) != len(order):
        raise ValueError("invalid M16 class schedule")
    parts, offset = [], 0
    for increment in increments:
        if increment <= 0:
            raise ValueError("class increments must be positive")
        classes = torch.tensor(order[offset : offset + increment], dtype=labels.dtype)
        parts.append(torch.nonzero(torch.isin(labels.cpu(), classes), as_tuple=False).squeeze(1))
        offset += increment
    if sum(len(part) for part in parts) != len(labels):
        raise ValueError("labels do not match the complete class schedule")
    return parts


def _projection(config: dict, replicate: dict, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(replicate["projection_seed"])
    return torch.randn(
        config["backbone"]["feature_dimension"],
        config["ranpac"]["expand_dimension"],
        generator=generator,
        dtype=torch.float32,
    ).to(device)


def _encode(
    features: torch.Tensor, projection: torch.Tensor, indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    return m5._encode_indices(
        lambda values: torch.relu(
            values.to(device=projection.device, dtype=torch.float32) @ projection
        ),
        features,
        indices,
        batch_size,
    )


def _first_task_fit_validation(
    first_task_indices: torch.Tensor, split_seed: int, fit_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    values = first_task_indices.cpu().tolist()
    random.Random(split_seed).shuffle(values)
    boundary = int(len(values) * fit_fraction)
    if boundary <= 0 or boundary >= len(values):
        raise ValueError("M16 first-task Ridge split is empty")
    return torch.tensor(values[:boundary]), torch.tensor(values[boundary:])


def _select_ridge_dual(
    fit_codes: torch.Tensor,
    fit_labels: torch.Tensor,
    validation_codes: torch.Tensor,
    validation_labels: torch.Tensor,
    candidate_lambdas: list[float],
) -> tuple[float, list[dict]]:
    classes = sorted(set(map(int, fit_labels.cpu().tolist())))
    validation_classes = set(map(int, validation_labels.cpu().tolist()))
    if not validation_classes.issubset(classes):
        raise ValueError("Ridge validation contains a class absent from fit")
    columns = {class_id: index for index, class_id in enumerate(classes)}
    fit_targets = torch.nn.functional.one_hot(
        torch.tensor([columns[int(value)] for value in fit_labels.cpu().tolist()]),
        num_classes=len(classes),
    ).to(device=fit_codes.device, dtype=torch.float64)
    validation_targets = torch.nn.functional.one_hot(
        torch.tensor(
            [columns[int(value)] for value in validation_labels.cpu().tolist()]
        ),
        num_classes=len(classes),
    ).to(device=fit_codes.device, dtype=torch.float64)
    fit = fit_codes.to(torch.float64)
    validation = validation_codes.to(torch.float64)
    gram_dual = fit @ fit.T
    cross_kernel = validation @ fit.T
    identity = torch.eye(len(fit), device=fit.device, dtype=torch.float64)
    losses = []
    for ridge in candidate_lambdas:
        alpha = torch.linalg.solve(gram_dual + float(ridge) * identity, fit_targets)
        prediction = cross_kernel @ alpha
        loss = float(torch.mean((prediction - validation_targets) ** 2).item())
        if not math.isfinite(loss):
            raise RuntimeError("non-finite M16 Ridge-selection loss")
        losses.append({"ridge_lambda": float(ridge), "validation_mse": loss})
    selected = min(losses, key=lambda item: (item["validation_mse"], item["ridge_lambda"]))
    return float(selected["ridge_lambda"]), losses


def _cache_file(cache_dir: str | Path, split: str) -> Path:
    return Path(cache_dir) / f"{split}.pt"


def _load_cache(config: dict, cache_dir: str | Path, split: str) -> tuple[dict, dict]:
    cache_path = _cache_file(cache_dir, split)
    metadata_path = Path(cache_dir) / "metadata.json"
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"M16 {split} feature cache is missing")
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = config[f"expected_{split}_samples"]
    if not (
        set(payload) == {"features", "labels"}
        and payload["features"].shape
        == (expected, config["backbone"]["feature_dimension"])
        and payload["labels"].shape == (expected,)
        and sorted(torch.unique(payload["labels"]).tolist())
        == list(range(config["num_classes"]))
        and bool(torch.isfinite(payload["features"]).all())
        and metadata.get("checkpoint_sha256")
        == config["backbone"]["checkpoint_sha256"]
        and metadata.get(f"{split}_features_sha256")
        == _tensor_sha256(payload["features"])
        and metadata.get(f"{split}_labels_sha256")
        == _tensor_sha256(payload["labels"])
    ):
        raise ValueError(f"M16 {split} cache identity/inventory mismatch")
    return payload, metadata


def _find_imagefolder_split(root: str | Path, split: str, expected: int) -> Path:
    root_path = Path(root).resolve()
    # Deliberately avoid recursive discovery: before authorization, a recursive
    # walk from the dataset root would also enumerate entries under raw test.
    candidates = [
        root_path / split,
        root_path / "car_data" / split,
        root_path / "car_data" / "car_data" / split,
        root_path / "stanford-cars-dataset" / split,
        root_path / "stanford-cars-dataset" / "car_data" / split,
        root_path / "stanford-cars-dataset" / "car_data" / "car_data" / split,
    ]
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    valid = []
    for candidate in dict.fromkeys(
        path.resolve() for path in candidates if path.is_dir()
    ):
        class_dirs = [path for path in candidate.iterdir() if path.is_dir()]
        if len(class_dirs) != 196:
            continue
        count = sum(
            1 for path in candidate.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        if count == expected:
            valid.append(candidate)
    if len(valid) != 1:
        raise FileNotFoundError(
            f"expected one Stanford Cars {split} ImageFolder with {expected} images; "
            f"found {[str(path) for path in valid]}"
        )
    return valid[0]


def _cars_dataset(config: dict, root: str | Path, split: str):
    from torchvision import datasets, transforms

    expected = config[f"expected_{split}_samples"]
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    folder = _find_imagefolder_split(root, split, expected)
    dataset = datasets.ImageFolder(folder, transform=transform)
    if len(dataset) != expected or len(dataset.classes) != config["num_classes"]:
        raise ValueError("Stanford Cars ImageFolder inventory mismatch")
    manifest = [
        [str(Path(path).relative_to(folder)).replace("\\", "/"), int(label)]
        for path, label in dataset.samples
    ]
    return dataset, {
        "layout": "imagefolder",
        "split_root": str(folder),
        "classes": list(dataset.classes),
        "class_to_idx": dataset.class_to_idx,
        "sample_manifest_sha256": _canonical_sha256(manifest),
    }


def _load_resnet50(config: dict, checkpoint: str | Path, device: torch.device):
    from torchvision.models import resnet50

    checkpoint_path = Path(checkpoint)
    backbone = config["backbone"]
    if not (
        checkpoint_path.stat().st_size == backbone["checkpoint_size_bytes"]
        and _sha256_file(checkpoint_path) == backbone["checkpoint_sha256"]
    ):
        raise ValueError("M16 ResNet-50 checkpoint identity mismatch")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Identity()
    return model.eval().to(device)


def _extract_split(args, split: str) -> None:
    config = _read_config(args.config)
    if args.require_clean_git or config["integrity_gates"]["require_clean_git"]:
        _require_clean_git()
    cache_dir = Path(args.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _cache_file(cache_dir, split).exists():
        raise FileExistsError(f"M16 {split} cache already exists")
    dataset, dataset_identity = _cars_dataset(config, args.root, split)
    metadata_path = cache_dir / "metadata.json"
    if split == "test" and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        train_identity = existing.get("train_dataset_identity", {})
        if (
            dataset_identity["classes"] != train_identity.get("classes")
            or dataset_identity["class_to_idx"] != train_identity.get("class_to_idx")
        ):
            raise ValueError("M16 train/test class mapping mismatch")
    device = torch.device(args.device)
    model = _load_resnet50(config, args.backbone_checkpoint, device)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    features, labels = [], []
    with torch.no_grad():
        for index, (images, target) in enumerate(loader):
            value = model(images.to(device, non_blocking=True)).detach().cpu()
            features.append(value)
            labels.append(target.cpu())
            print(f"M16 {split.upper()} FEATURES {index + 1}/{len(loader)}", flush=True)
    payload = {"features": torch.cat(features), "labels": torch.cat(labels).long()}
    expected = config[f"expected_{split}_samples"]
    if not (
        payload["features"].shape
        == (expected, config["backbone"]["feature_dimension"])
        and payload["labels"].shape == (expected,)
        and sorted(torch.unique(payload["labels"]).tolist())
        == list(range(config["num_classes"]))
        and bool(torch.isfinite(payload["features"]).all())
    ):
        raise RuntimeError(f"invalid extracted M16 {split} features")
    torch.save(payload, _cache_file(cache_dir, split))
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {
        "schema_version": 1,
        "study_id": config["study_id"],
        "git_commit_at_first_extraction": _git_commit(),
        "checkpoint_sha256": config["backbone"]["checkpoint_sha256"],
        "backbone": config["backbone"],
    }
    metadata.update({
        f"{split}_dataset_identity": dataset_identity,
        f"{split}_features_sha256": _tensor_sha256(payload["features"]),
        f"{split}_labels_sha256": _tensor_sha256(payload["labels"]),
        f"{split}_shape": list(payload["features"].shape),
        f"{split}_features_materialized": True,
    })
    _atomic_json(metadata_path, metadata)


def select_ridge(args) -> dict:
    config = _read_config(args.config)
    if args.require_clean_git or config["integrity_gates"]["require_clean_git"]:
        _require_clean_git()
    cache_dir = Path(args.feature_cache_dir)
    if _cache_file(cache_dir, "test").exists():
        raise RuntimeError("M16 test cache exists before Ridge selection")
    train, metadata = _load_cache(config, cache_dir, "train")
    device = torch.device(args.device)
    records = []
    for replicate in config["replicates"]:
        order = random.Random(replicate["class_order_seed"]).sample(
            list(range(config["num_classes"])), config["num_classes"]
        )
        first_task = _split_by_increments(
            train["labels"], order, config["class_increments"]
        )[0]
        fit_indices, validation_indices = _first_task_fit_validation(
            first_task,
            replicate["ridge_split_seed"],
            config["ridge_selection"]["fit_fraction"],
        )
        projection = _projection(config, replicate, device)
        fit_codes = _encode(
            train["features"], projection, fit_indices,
            config["ranpac"]["encode_batch_size"],
        )
        validation_codes = _encode(
            train["features"], projection, validation_indices,
            config["ranpac"]["encode_batch_size"],
        )
        selected, losses = _select_ridge_dual(
            fit_codes, train["labels"][fit_indices], validation_codes,
            train["labels"][validation_indices],
            config["ridge_selection"]["candidate_lambdas"],
        )
        records.append({
            "replicate": replicate,
            "class_order": order,
            "first_task_classes": order[: config["class_increments"][0]],
            "fit_indices_sha256": _tensor_sha256(fit_indices),
            "validation_indices_sha256": _tensor_sha256(validation_indices),
            "selected_ridge_lambda": selected,
            "candidate_losses": losses,
        })
        print(
            f"M16 RIDGE seed={replicate['class_order_seed']} lambda={selected:g}",
            flush=True,
        )
        del projection, fit_codes, validation_codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    identity = {
        "study_id": config["study_id"],
        "config_sha256": _sha256_file(args.config),
        "train_features_sha256": metadata["train_features_sha256"],
        "train_labels_sha256": metadata["train_labels_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "records": records,
        "uses_test_set": False,
    }
    result = {
        "schema_version": 1,
        "status": "LOCKED_M16_TRAIN_ONLY_RIDGE_SELECTION",
        "selection_id": _canonical_sha256(identity),
        "identity": identity,
    }
    _atomic_json(args.selection, result)
    return result


def _validate_selection(config: dict, config_path: str | Path, train: dict, selection_path: str | Path) -> dict:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    identity = selection.get("identity", {})
    records = identity.get("records", [])
    if not (
        selection.get("status") == "LOCKED_M16_TRAIN_ONLY_RIDGE_SELECTION"
        and selection.get("selection_id") == _canonical_sha256(identity)
        and identity.get("study_id") == config["study_id"]
        and identity.get("config_sha256") == _sha256_file(config_path)
        and identity.get("train_features_sha256") == _tensor_sha256(train["features"])
        and identity.get("train_labels_sha256") == _tensor_sha256(train["labels"])
        and identity.get("uses_test_set") is False
        and len(records) == len(config["replicates"])
    ):
        raise ValueError("M16 Ridge selection identity mismatch")
    for expected, record in zip(config["replicates"], records):
        if (
            record.get("replicate") != expected
            or record.get("class_order")
            != random.Random(expected["class_order_seed"]).sample(
                list(range(config["num_classes"])), config["num_classes"]
            )
            or float(record.get("selected_ridge_lambda", -1))
            not in config["ridge_selection"]["candidate_lambdas"]
        ):
            raise ValueError("M16 Ridge record mismatch")
    return selection


def authorize(args) -> dict:
    config = _read_config(args.config)
    if args.require_clean_git or config["integrity_gates"]["require_clean_git"]:
        _require_clean_git()
    cache_dir = Path(args.feature_cache_dir)
    if _cache_file(cache_dir, "test").exists():
        raise RuntimeError("M16 test cache exists before authorization")
    train, metadata = _load_cache(config, cache_dir, "train")
    selection = _validate_selection(config, args.config, train, args.selection)
    identity = {
        "study_id": config["study_id"],
        "git_commit": _git_commit(),
        "config_sha256": _sha256_file(args.config),
        "selection_sha256": _sha256_file(args.selection),
        "selection_id": selection["selection_id"],
        "train_features_sha256": metadata["train_features_sha256"],
        "train_labels_sha256": metadata["train_labels_sha256"],
        "train_dataset_identity": metadata["train_dataset_identity"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "test_features_materialized_at_authorization": False,
    }
    result = {
        "schema_version": 1,
        "authorized": True,
        "uses_test_set": False,
        "authorization_id": _canonical_sha256(identity),
        "identity": identity,
    }
    _atomic_json(args.authorization, result)
    print(f"M16 AUTHORIZED {result['authorization_id']}", flush=True)
    return result


def _read_authorization(args, config: dict, train: dict) -> dict:
    authorization = json.loads(Path(args.authorization).read_text(encoding="utf-8"))
    selection = _validate_selection(config, args.config, train, args.selection)
    metadata = json.loads((Path(args.feature_cache_dir) / "metadata.json").read_text())
    identity = {
        "study_id": config["study_id"],
        "git_commit": _git_commit(),
        "config_sha256": _sha256_file(args.config),
        "selection_sha256": _sha256_file(args.selection),
        "selection_id": selection["selection_id"],
        "train_features_sha256": metadata["train_features_sha256"],
        "train_labels_sha256": metadata["train_labels_sha256"],
        "train_dataset_identity": metadata["train_dataset_identity"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "test_features_materialized_at_authorization": False,
    }
    if not (
        authorization.get("authorized") is True
        and authorization.get("uses_test_set") is False
        and authorization.get("identity") == identity
        and authorization.get("authorization_id") == _canonical_sha256(identity)
    ):
        raise RuntimeError("M16 authorization identity mismatch")
    return authorization


def extract_test(args) -> None:
    config = _read_config(args.config)
    train, _ = _load_cache(config, args.feature_cache_dir, "train")
    if _cache_file(args.feature_cache_dir, "test").exists():
        raise FileExistsError("M16 test cache already exists")
    _read_authorization(args, config, train)
    _extract_split(args, "test")
    metadata_path = Path(args.feature_cache_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["m16_authorization_id"] = json.loads(
        Path(args.authorization).read_text()
    )["authorization_id"]
    _atomic_json(metadata_path, metadata)


def _make_backend(config: dict, ridge_lambda: float, method: str, device: torch.device):
    common = m5._common_backend(
        config["ranpac"]["expand_dimension"], float(ridge_lambda), device
    )
    if method == "exact":
        return ExactGramBackend(**common)
    p2b = config["p2b"]
    shared = {
        "block_size": p2b["block_size"],
        "group_size": p2b["group_size"],
        "update_panel_size": p2b["update_panel_size"],
        "update_trailing_chunk_size": p2b["update_trailing_chunk_size"],
        "first_update_backend": p2b["first_update_backend"],
        "quantization_backend": p2b["quantization_backend"],
        "quantization_batch_blocks": p2b["quantization_batch_blocks"],
    }
    if method == "p2b_int8":
        return SquareRootBackend(storage_mode="int8", **shared, **common)
    if method == "adaptive_int8_fp16":
        return SquareRootBackend(
            storage_mode="adaptive_int8_fp16",
            adaptive_budget_fraction=config["adaptive"][
                "budget_fraction_between_int8_and_fp16"
            ],
            **shared,
            **common,
        )
    raise ValueError(f"unsupported M16 method: {method}")


def _run_unit(
    config: dict, train: dict, test: dict, replicate: dict,
    ridge_lambda: float, method: str, device: torch.device,
) -> dict:
    order = random.Random(replicate["class_order_seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    train_parts = _split_by_increments(
        train["labels"], order, config["class_increments"]
    )
    test_parts = _split_by_increments(
        test["labels"], order, config["class_increments"]
    )
    projection = _projection(config, replicate, device)
    backend = _make_backend(config, ridge_lambda, method, device)
    records, update_seconds, encoding_seconds = [], 0.0, 0.0
    for task_index, train_indices in enumerate(train_parts):
        m5._sync(device)
        started = time.perf_counter()
        codes = _encode(
            train["features"], projection, train_indices,
            config["ranpac"]["encode_batch_size"],
        )
        m5._sync(device)
        encoding_seconds += time.perf_counter() - started
        m5._sync(device)
        started = time.perf_counter()
        backend.update(codes, train["labels"][train_indices])
        m5._sync(device)
        update_seconds += time.perf_counter() - started
        seen = torch.cat(test_parts[: task_index + 1])
        accuracy = m5._evaluate(
            encoder=lambda values: torch.relu(
                values.to(device=device, dtype=torch.float32) @ projection
            ),
            backends={method: backend},
            features=test["features"], labels=test["labels"], indices=seen,
            batch_size=config["ranpac"]["evaluation_batch_size"],
        )[method]
        state = persistent_tensor_bytes(
            {"projection": projection, **backend.persistent_tensors()}
        )
        record = {
            "task": task_index + 1,
            "classes_seen": sum(config["class_increments"][: task_index + 1]),
            "test_accuracy_percent": float(accuracy),
            "total_persistent_bytes": int(state),
            "solver_relative_residual": float(
                backend.diagnostics["solver_relative_residual"]
            ),
        }
        if method == "adaptive_int8_fp16":
            record.update({
                "selected_fp16_blocks": int(
                    backend.diagnostics["selected_fp16_blocks"]
                ),
                "total_blocks": int(backend.diagnostics["total_blocks"]),
                "factor_budget_ceiling_bytes": int(
                    backend.diagnostics["factor_budget_ceiling_bytes"]
                ),
            })
        records.append(record)
        print(
            f"M16 seed={replicate['class_order_seed']} {method} "
            f"task={task_index + 1}/10 test={accuracy:.4f}", flush=True,
        )
        del codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    accuracies = [record["test_accuracy_percent"] for record in records]
    return {
        "identity": {
            "replicate": replicate,
            "method": method,
            "ridge_lambda": float(ridge_lambda),
        },
        "class_order": order,
        "records": records,
        "uses_test_set": True,
        "test_tuning_allowed": False,
        "average_incremental_accuracy_percent": sum(accuracies) / len(accuracies),
        "final_accuracy_percent": accuracies[-1],
        "final_total_persistent_bytes": records[-1]["total_persistent_bytes"],
        "maximum_solver_relative_residual": max(
            record["solver_relative_residual"] for record in records
        ),
        "representation_encoding_seconds": encoding_seconds,
        "analytic_update_seconds": update_seconds,
    }


def _mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _aggregate(units: list[dict]) -> dict:
    exact = {
        unit["identity"]["replicate"]["class_order_seed"]: unit
        for unit in units if unit["identity"]["method"] == "exact"
    }
    result = {}
    for method in METHODS:
        selected = [unit for unit in units if unit["identity"]["method"] == method]
        selected.sort(key=lambda unit: unit["identity"]["replicate"]["class_order_seed"])
        aia = [unit["average_incremental_accuracy_percent"] for unit in selected]
        final = [unit["final_accuracy_percent"] for unit in selected]
        result[method] = {
            "average_incremental_accuracy_percent": _mean_sd(aia),
            "final_accuracy_percent": _mean_sd(final),
            "paired_aia_difference_from_exact_pp": _mean_sd([
                unit["average_incremental_accuracy_percent"]
                - exact[unit["identity"]["replicate"]["class_order_seed"]][
                    "average_incremental_accuracy_percent"
                ]
                for unit in selected
            ]),
            "paired_final_difference_from_exact_pp": _mean_sd([
                unit["final_accuracy_percent"]
                - exact[unit["identity"]["replicate"]["class_order_seed"]][
                    "final_accuracy_percent"
                ]
                for unit in selected
            ]),
            "final_total_persistent_bytes": _mean_sd([
                float(unit["final_total_persistent_bytes"]) for unit in selected
            ]),
            "analytic_update_seconds": _mean_sd([
                unit["analytic_update_seconds"] for unit in selected
            ]),
            "maximum_solver_relative_residual": max(
                unit["maximum_solver_relative_residual"] for unit in selected
            ),
        }
    return result


def _validate_unit(
    unit: dict, config: dict, replicate: dict, ridge_lambda: float, method: str
) -> None:
    expected_identity = {
        "replicate": replicate,
        "method": method,
        "ridge_lambda": float(ridge_lambda),
    }
    expected_order = random.Random(replicate["class_order_seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    records = unit.get("records", [])
    if not (
        unit.get("identity") == expected_identity
        and unit.get("class_order") == expected_order
        and unit.get("uses_test_set") is True
        and unit.get("test_tuning_allowed") is False
        and len(records) == len(config["class_increments"])
    ):
        raise RuntimeError("M16 resumable unit identity mismatch")
    for index, record in enumerate(records):
        values = (
            record.get("test_accuracy_percent"),
            record.get("solver_relative_residual"),
        )
        if not (
            record.get("task") == index + 1
            and record.get("classes_seen")
            == sum(config["class_increments"][: index + 1])
            and isinstance(record.get("total_persistent_bytes"), int)
            and record["total_persistent_bytes"] > 0
            and all(value is not None and math.isfinite(float(value)) for value in values)
            and 0 <= float(record["test_accuracy_percent"]) <= 100
            and float(record["solver_relative_residual"]) >= 0
        ):
            raise RuntimeError(f"invalid M16 unit task record {index + 1}")
    accuracies = [float(record["test_accuracy_percent"]) for record in records]
    residual = max(float(record["solver_relative_residual"]) for record in records)
    if not (
        math.isclose(
            float(unit.get("average_incremental_accuracy_percent", math.nan)),
            sum(accuracies) / len(accuracies), rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            float(unit.get("final_accuracy_percent", math.nan)),
            accuracies[-1], rel_tol=0.0, abs_tol=1e-12,
        )
        and unit.get("final_total_persistent_bytes")
        == records[-1]["total_persistent_bytes"]
        and math.isclose(
            float(unit.get("maximum_solver_relative_residual", math.nan)),
            residual, rel_tol=0.0, abs_tol=1e-12,
        )
        and all(
            math.isfinite(float(unit.get(name, math.nan)))
            and float(unit[name]) >= 0
            for name in ("representation_encoding_seconds", "analytic_update_seconds")
        )
    ):
        raise RuntimeError("invalid M16 resumable unit summary")


def _unit_path(output_dir: Path, replicate: dict, method: str) -> Path:
    return output_dir / "units" / f"seed_{replicate['class_order_seed']}_{method}.json"


def _write_csv(path: Path, units: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "class_order_seed", "projection_seed", "method", "ridge_lambda",
            "aia_percent", "final_accuracy_percent", "final_state_bytes",
            "max_solver_relative_residual", "analytic_update_seconds",
        ])
        for unit in units:
            identity = unit["identity"]
            replicate = identity["replicate"]
            writer.writerow([
                replicate["class_order_seed"], replicate["projection_seed"],
                identity["method"], identity["ridge_lambda"],
                unit["average_incremental_accuracy_percent"],
                unit["final_accuracy_percent"], unit["final_total_persistent_bytes"],
                unit["maximum_solver_relative_residual"],
                unit["analytic_update_seconds"],
            ])


def _export_handoff(output_dir: Path, args, completed_units: list[Path]) -> Path:
    export = Path(args.handoff_export)
    export.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "study_id": _read_config(args.config)["study_id"],
        "source_commit": _git_commit(),
        "config_sha256": _sha256_file(args.config),
        "selection_sha256": _sha256_file(args.selection),
        "authorization_sha256": _sha256_file(args.authorization),
        "units": [
            {
                "name": f"units/{path.name}",
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(completed_units)
        ],
    }
    payload = json.dumps(identity, indent=2).encode() + b"\n"
    with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("handoff_manifest.json", payload)
        for path in sorted(completed_units):
            archive.write(path, f"units/{path.name}")
    return export


def _export(output_dir: Path, args, result: dict) -> Path:
    manifest = []
    members = [
        Path(args.config), Path(args.selection), Path(args.authorization),
        output_dir / "m16_results.json", output_dir / "m16_units.csv",
        *sorted((output_dir / "units").glob("*.json")),
    ]
    for path in members:
        manifest.append({
            "name": path.name if path.parent == output_dir else str(path.relative_to(output_dir))
            if output_dir in path.parents else path.name,
            "sha256": _sha256_file(path), "size_bytes": path.stat().st_size,
        })
    manifest_path = output_dir / "m16_manifest.json"
    _atomic_json(manifest_path, {"status": result["status"], "files": manifest})
    export = Path(args.export)
    export.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            if path.parent == output_dir:
                arcname = path.name
            elif output_dir in path.parents:
                arcname = str(path.relative_to(output_dir)).replace("\\", "/")
            else:
                arcname = path.name
            archive.write(path, arcname)
        archive.write(manifest_path, manifest_path.name)
    return export


def run(args) -> dict:
    config = _read_config(args.config)
    if args.require_clean_git or config["integrity_gates"]["require_clean_git"]:
        _require_clean_git()
    train, _ = _load_cache(config, args.feature_cache_dir, "train")
    test, metadata = _load_cache(config, args.feature_cache_dir, "test")
    authorization = _read_authorization(args, config, train)
    if metadata.get("m16_authorization_id") != authorization["authorization_id"]:
        raise RuntimeError("M16 test cache was not created under this authorization")
    selection = _validate_selection(config, args.config, train, args.selection)
    ridge_by_seed = {
        record["replicate"]["class_order_seed"]: record["selected_ridge_lambda"]
        for record in selection["identity"]["records"]
    }
    output_dir = Path(args.output_dir)
    (output_dir / "units").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    units = []
    new_units = 0
    stopped_early = False
    for replicate in config["replicates"]:
        for method in METHODS:
            path = _unit_path(output_dir, replicate, method)
            if path.exists():
                unit = json.loads(path.read_text(encoding="utf-8"))
            else:
                unit = _run_unit(
                    config, train, test, replicate,
                    ridge_by_seed[replicate["class_order_seed"]], method, device,
                )
                _atomic_json(path, unit)
                new_units += 1
            _validate_unit(
                unit, config, replicate,
                ridge_by_seed[replicate["class_order_seed"]], method,
            )
            units.append(unit)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.max_new_units is not None and new_units >= args.max_new_units:
                stopped_early = len(units) < len(config["replicates"]) * len(METHODS)
                break
        if stopped_early:
            break
    completed_paths = sorted((output_dir / "units").glob("*.json"))
    if stopped_early:
        handoff = _export_handoff(output_dir, args, completed_paths)
        result = {
            "schema_version": 1,
            "status": "INCOMPLETE_M16_CARS_RANPAC_PHASE2_LOCKED",
            "completed_units": len(completed_paths),
            "total_units": len(config["replicates"]) * len(METHODS),
            "handoff": str(handoff),
            "handoff_sha256": _sha256_file(handoff),
        }
        print(
            f"M16 CHECKPOINT: {len(completed_paths)}/18 {handoff} "
            f"SHA256={result['handoff_sha256']}", flush=True,
        )
        return result
    residual_gate = config["integrity_gates"]["maximum_solver_relative_residual"]
    residual_pass = all(
        unit["maximum_solver_relative_residual"] <= residual_gate for unit in units
    )
    compressed_smaller = True
    for replicate in config["replicates"]:
        seed = replicate["class_order_seed"]
        selected = {
            unit["identity"]["method"]: unit for unit in units
            if unit["identity"]["replicate"]["class_order_seed"] == seed
        }
        compressed_smaller &= all(
            selected[method]["final_total_persistent_bytes"]
            < selected["exact"]["final_total_persistent_bytes"]
            for method in METHODS[1:]
        )
    gates = {
        "all_units_complete": len(units) == 18,
        "solver_residual": residual_pass,
        "compressed_state_smaller_than_exact": bool(compressed_smaller),
        "accuracy_gate": None,
    }
    status = (
        "PASS_M16_CARS_RANPAC_PHASE2_LOCKED"
        if all(value for key, value in gates.items() if key != "accuracy_gate")
        else "FAIL_M16_CARS_RANPAC_PHASE2_LOCKED"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "study_id": config["study_id"],
        "git_commit": _git_commit(),
        "authorization_id": authorization["authorization_id"],
        "selection_id": selection["selection_id"],
        "uses_test_set": True,
        "test_tuning_allowed": False,
        "scope_disclosure": (
            "Official-source-pinned RanPAC Phase-2 ResNet-50/random-ReLU frontend "
            "with a first-task-selected then frozen Ridge; not a full PETL RanPAC reproduction."
        ),
        "selected_ridge_by_class_order_seed": ridge_by_seed,
        "summary": _aggregate(units),
        "gates": gates,
        "units": units,
    }
    _atomic_json(output_dir / "m16_results.json", result)
    _write_csv(output_dir / "m16_units.csv", units)
    export = _export(output_dir, args, result)
    print(f"M16 STATUS: {status}", flush=True)
    print(f"M16 EXPORT: {export} SHA256={_sha256_file(export)}", flush=True)
    if not status.startswith("PASS"):
        raise AssertionError("M16 integrity gates failed; preserve the artifact")
    return result


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default="configs/srq_generalization_m16_cars_phase2_locked.json"
    )
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--require-clean-git", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract_train_parser = commands.add_parser("extract-train")
    _add_common(extract_train_parser)
    extract_train_parser.add_argument("--root", required=True)
    extract_train_parser.add_argument("--backbone-checkpoint", required=True)
    extract_train_parser.add_argument("--device", default="cuda")
    extract_train_parser.add_argument("--batch-size", type=int, default=128)
    extract_train_parser.add_argument("--num-workers", type=int, default=2)
    select_parser = commands.add_parser("select-ridge")
    _add_common(select_parser)
    select_parser.add_argument("--selection", required=True)
    select_parser.add_argument("--device", default="cuda")
    authorize_parser = commands.add_parser("authorize")
    _add_common(authorize_parser)
    authorize_parser.add_argument("--selection", required=True)
    authorize_parser.add_argument("--authorization", required=True)
    extract_test_parser = commands.add_parser("extract-test")
    _add_common(extract_test_parser)
    extract_test_parser.add_argument("--selection", required=True)
    extract_test_parser.add_argument("--authorization", required=True)
    extract_test_parser.add_argument("--root", required=True)
    extract_test_parser.add_argument("--backbone-checkpoint", required=True)
    extract_test_parser.add_argument("--device", default="cuda")
    extract_test_parser.add_argument("--batch-size", type=int, default=128)
    extract_test_parser.add_argument("--num-workers", type=int, default=2)
    run_parser = commands.add_parser("run")
    _add_common(run_parser)
    run_parser.add_argument("--selection", required=True)
    run_parser.add_argument("--authorization", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--export", required=True)
    run_parser.add_argument("--handoff-export", required=True)
    run_parser.add_argument("--max-new-units", type=int)
    run_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "extract-train":
        _extract_split(args, "train")
    elif args.command == "select-ridge":
        select_ridge(args)
    elif args.command == "authorize":
        authorize(args)
    elif args.command == "extract-test":
        extract_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
