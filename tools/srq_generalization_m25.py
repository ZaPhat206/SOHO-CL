"""Cross-backbone train-only confirmation for SRQ on RanPAC/CIFAR-100.

M25 is intentionally narrower than the equal-memory studies.  It asks one
question: does the same frozen-state SRQ backend behave similarly when the
RanPAC input is a frozen ImageNet ResNet-50 feature rather than a ViT feature?
The runner never opens or creates ``test.pt``.  Ridge selection, class order,
and the validation stream are locked per seed before any reported update.
"""

from __future__ import annotations

import argparse
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
import traceback
from types import SimpleNamespace
from typing import Any
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
from tools import srq_generalization_m4 as m4  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools import srq_generalization_m11 as m11  # noqa: E402
from tools.experiment_runner import (  # noqa: E402
    save_train_cache,
    split,
    train_validation_indices,
)


METHODS = ("exact", "p2b_int8", "adaptive_int8_fp16")


def _stream_complete(payload: dict, stream: dict) -> bool:
    return (
        payload.get("schema_version") == 1
        and payload.get("study_id") == "srq-generalization-m25-ranpac-resnet-cifar-train-only-v1"
        and payload.get("stream_id") == stream["stream_id"]
        and payload.get("status") == "PASS_M25_RANPAC_RESNET_CIFAR_TRAIN_ONLY"
        and payload.get("uses_test_set") is False
        and all(payload.get("gates", {}).values())
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _sequence_sha256(parts: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        value = part.detach().cpu().to(torch.int64).contiguous()
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _clean_git() -> None:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("M25 requires a clean source checkout")


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "study_id", "dataset", "model_name",
        "checkpoint_sha256", "uses_test_set", "accuracy_based_selection",
        "num_classes", "num_tasks", "expected_train_samples",
        "expected_test_samples", "outer_validation_fraction", "statistics_dtype",
        "solver_dtype", "streams", "methods", "backbone", "ranpac",
        "ridge_selection", "p2b", "adaptive", "evaluation", "integrity_gates",
    }
    if set(config) != required or config.get("schema_version") != 1:
        raise ValueError("M25 config keys/schema mismatch")
    if config["dataset"] != "CIFAR-100" or config["model_name"] != "resnet50":
        raise ValueError("M25 is locked to CIFAR-100/ResNet-50")
    if config["study_id"] != "srq-generalization-m25-ranpac-resnet-cifar-train-only-v1":
        raise ValueError("M25 study identity changed")
    if config["uses_test_set"] is not False or config["accuracy_based_selection"] is not False:
        raise ValueError("M25 must remain train-only and accuracy-blind")
    if (
        config["num_classes"] != 100
        or config["num_tasks"] != 10
        or config["expected_train_samples"] != 50000
        or config["expected_test_samples"] != 10000
        or config["statistics_dtype"] != "float32"
        or config["solver_dtype"] != "float32"
        or not 0 < config["outer_validation_fraction"] < 1
        or tuple(config["methods"]) != METHODS
    ):
        raise ValueError("M25 dataset/task/method contract changed")
    streams = config["streams"]
    if not isinstance(streams, list) or len(streams) != 6:
        raise ValueError("M25 requires six preregistered streams")
    if [item.get("stream_id") for item in streams] != [f"s{seed}" for seed in range(2025, 2031)]:
        raise ValueError("M25 stream ids/seeds changed")
    if any(set(item) != {"stream_id", "seed"} or item["seed"] != int(item["stream_id"][1:]) for item in streams):
        raise ValueError("M25 stream identity mismatch")
    backbone = config["backbone"]
    if not (
        backbone["architecture"] == "resnet50"
        and backbone["feature_dimension"] == 2048
        and backbone["weights"] == "IMAGENET1K_V2"
        and backbone["checkpoint_sha256"] == config["checkpoint_sha256"]
        and backbone["checkpoint_size_bytes"] == 102540417
    ):
        raise ValueError("M25 ResNet-50 checkpoint lock changed")
    ranpac = config["ranpac"]
    if not (
        ranpac["feature_dimension"] == 2048
        and ranpac["full_expand_dimension"] == 10000
        and ranpac["projection_distribution"] == "standard_normal"
        and ranpac["activation"] == "relu"
        and ranpac["path"] == "phase2_no_petl_random_relu"
        and min(ranpac["encode_batch_size"], ranpac["evaluation_batch_size"]) > 0
    ):
        raise ValueError("M25 RanPAC contract changed")
    selection = config["ridge_selection"]
    candidates = list(map(float, selection["candidate_lambdas"]))
    if not (
        selection["policy"] == "per_stream_train_only_calibration"
        and selection["metric"] == "mean_squared_error"
        and selection["tie_break"] == "smallest_lambda"
        and candidates == sorted(set(candidates))
        and min(candidates) > 0
        and selection["fit_samples_per_class"] > 0
        and selection["validation_samples_per_class"] > 0
    ):
        raise ValueError("M25 Ridge-selection policy changed")
    p2b = config["p2b"]
    if p2b["first_update_backend"] != "gram_cholesky" or p2b["quantization_backend"] != "streaming":
        raise ValueError("M25 P2B backend changed")
    adaptive = config["adaptive"]
    if not (
        adaptive["budget_fraction_between_int8_and_fp16"] == 0.25
        and adaptive["selection_signal"] == "current_factor_values_only_no_labels_or_accuracy"
    ):
        raise ValueError("M25 adaptive policy changed")
    gates = config["integrity_gates"]
    if not (
        gates["require_clean_git"] is True
        and gates["require_train_only_cache"] is True
        and gates["require_all_streams_complete"] is True
        and gates["require_compressed_state_smaller_than_exact"] is True
        and float(gates["maximum_solver_relative_residual"]) > 0
        and gates["accuracy_gate"] is None
    ):
        raise ValueError("M25 integrity policy changed")
    return config


def _load_train_cache(config: dict, cache_dir: str | Path) -> tuple[dict, dict]:
    root = Path(cache_dir)
    if (root / "test.pt").exists():
        raise RuntimeError("M25 refuses a visible test.pt")
    metadata_path, train_path = root / "metadata.json", root / "train.pt"
    if not metadata_path.is_file() or not train_path.is_file():
        raise FileNotFoundError("M25 requires metadata.json and train.pt")
    payload = torch.load(train_path, map_location="cpu", weights_only=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not (
        set(payload) == {"features", "labels"}
        and payload["features"].shape == (config["expected_train_samples"], config["backbone"]["feature_dimension"])
        and payload["labels"].shape == (config["expected_train_samples"],)
        and sorted(map(int, torch.unique(payload["labels"]).tolist())) == list(range(config["num_classes"]))
        and bool(torch.isfinite(payload["features"]).all())
        and metadata.get("dataset") == config["dataset"]
        and metadata.get("backbone_model") == config["model_name"]
        and metadata.get("checkpoint_sha256") == config["checkpoint_sha256"]
    ):
        raise ValueError("M25 train cache identity/inventory mismatch")
    return payload, metadata


def _load_resnet50(config: dict, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    from torchvision.models import resnet50

    path = Path(checkpoint)
    if path.stat().st_size != config["backbone"]["checkpoint_size_bytes"] or _sha256_file(path) != config["checkpoint_sha256"]:
        raise ValueError("M25 ResNet-50 checkpoint identity mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Identity()
    return model.eval().to(device)


def extract_train(args) -> dict:
    """Extract only CIFAR-100 train features; never opens the test split."""
    config = _read_config(args.config)
    if args.require_clean_git:
        _clean_git()
    cache_dir = Path(args.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if (cache_dir / "test.pt").exists() or (cache_dir / "train.pt").exists():
        raise FileExistsError("M25 feature cache already exists")
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.CIFAR100(
        root=args.root, train=True, transform=transform, download=bool(args.download)
    )
    if len(dataset) != config["expected_train_samples"]:
        raise ValueError("M25 CIFAR-100 train inventory mismatch")
    device = torch.device(args.device)
    model = _load_resnet50(config, args.backbone_checkpoint, device)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    features, labels = [], []
    with torch.no_grad():
        for batch, (images, target) in enumerate(loader, 1):
            features.append(model(images.to(device, non_blocking=True)).cpu())
            labels.append(target.cpu())
            print(f"M25 TRAIN FEATURES {batch}/{len(loader)}", flush=True)
    train_features = torch.cat(features)
    train_labels = torch.cat(labels).long()
    if train_features.shape != (config["expected_train_samples"], config["backbone"]["feature_dimension"]):
        raise RuntimeError("M25 extracted feature shape mismatch")
    dummy = SimpleNamespace(dataset=config["dataset"], model_name=config["model_name"], data_augmentation=config["backbone"]["preprocessing"])
    save_train_cache(
        cache_dir, train_features, train_labels, dummy,
        config["expected_test_samples"], config["checkpoint_sha256"],
    )
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "m25_study_id": config["study_id"],
        "train_features_sha256": _tensor_sha256(train_features),
        "train_labels_sha256": _tensor_sha256(train_labels),
        "dataset_identity": {
            "dataset": "CIFAR-100",
            "split": "torchvision train=True",
            "sample_count": len(dataset),
            "class_names_sha256": _canonical_sha256(dataset.classes),
        },
    })
    _atomic_json(metadata_path, metadata)
    return metadata


def _projection(config: dict, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        config["ranpac"]["feature_dimension"],
        config["ranpac"]["full_expand_dimension"],
        generator=generator, dtype=torch.float32,
    ).to(device)


def _encoder(projection: torch.Tensor, device: torch.device):
    return lambda values: torch.relu(values.to(device=device, dtype=torch.float32) @ projection)


def _run_stream(config: dict, config_path: Path, stream: dict, cache_dir: Path, output_dir: Path, device_name: str) -> dict:
    train, metadata = _load_train_cache(config, cache_dir)
    seed = int(stream["seed"])
    device = torch.device(device_name)
    projection = _projection(config, seed, device)
    order = random.Random(seed).sample(list(range(config["num_classes"])), config["num_classes"])
    task_indices = split(train["labels"], order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"], task_indices, seed, config["outer_validation_fraction"]
    )
    selection = config["ridge_selection"]
    fit_indices, calibration_validation = m4._calibration_indices(
        train["labels"], training_parts, seed=seed,
        fit_per_class=int(selection["fit_samples_per_class"]),
        validation_per_class=int(selection["validation_samples_per_class"]),
    )
    encoder = _encoder(projection, device)
    ridge = m5._select_ridge(
        name="resnet50_random_relu", encoder=encoder, features=train["features"],
        labels=train["labels"], fit_indices=fit_indices,
        validation_indices=calibration_validation,
        candidate_lambdas=list(map(float, selection["candidate_lambdas"])),
        num_classes=config["num_classes"], batch_size=config["ranpac"]["encode_batch_size"],
    )
    ridge_lambda = float(ridge["selected_ridge_lambda"])
    p2b = config["p2b"]
    common = m5._common_backend(config["ranpac"]["full_expand_dimension"], ridge_lambda, device)
    backends = {
        "exact": ExactGramBackend(**common),
        "p2b_int8": SquareRootBackend(
            storage_mode="int8", block_size=p2b["block_size"], group_size=p2b["group_size"],
            update_panel_size=p2b["update_panel_size"], update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
            first_update_backend=p2b["first_update_backend"], quantization_backend=p2b["quantization_backend"],
            quantization_batch_blocks=p2b["quantization_batch_blocks"], **common,
        ),
    }
    projection_bytes = 4 * config["ranpac"]["feature_dimension"] * config["ranpac"]["full_expand_dimension"]
    expected = {
        "exact": projection_bytes + m5._exact_backend_bytes(config["ranpac"]["full_expand_dimension"], config["num_classes"]),
        "p2b_int8": projection_bytes + m5._square_root_backend_bytes(
            config["ranpac"]["full_expand_dimension"], config["num_classes"],
            block_size=p2b["block_size"], group_size=p2b["group_size"], mode="int8",
        ),
    }
    group = m5._run_group(
        encoder=encoder, backends=backends, extra_tensors={"projection": projection},
        expected_total_bytes=expected, features=train["features"], labels=train["labels"],
        training_parts=training_parts, validation_parts=validation_parts,
        encode_batch_size=config["ranpac"]["encode_batch_size"],
        evaluation_batch_size=config["ranpac"]["evaluation_batch_size"], device=device,
    )
    adaptive = m11._run_adaptive_width(
        config=config, width=config["ranpac"]["full_expand_dimension"], ridge=ridge_lambda,
        projection=projection, train=train, training_parts=training_parts,
        validation_parts=validation_parts, device=device,
    )
    summary = {
        "validation_aia_percent": {
            name: sum(row["accuracy_percent"][name] for row in group["records"]) / len(group["records"])
            for name in METHODS[:2]
        },
        "final_validation_accuracy_percent": {
            name: group["records"][-1]["accuracy_percent"][name] for name in METHODS[:2]
        },
        "final_total_persistent_bytes": {
            name: group["records"][-1]["state"][name]["total_persistent_bytes"] for name in METHODS[:2]
        },
        "analytic_update_seconds": {name: group["analytic_update_seconds"][name] for name in METHODS[:2]},
        "representation_encoding_seconds": group["encoding_seconds"],
    }
    summary["validation_aia_percent"]["adaptive_int8_fp16"] = adaptive["validation_aia_percent"]
    summary["final_validation_accuracy_percent"]["adaptive_int8_fp16"] = adaptive["final_validation_accuracy_percent"]
    summary["final_total_persistent_bytes"]["adaptive_int8_fp16"] = adaptive["final_total_persistent_bytes"]
    summary["analytic_update_seconds"]["adaptive_int8_fp16"] = adaptive["analytic_update_seconds"]
    summary["representation_encoding_seconds"] = {
        "exact": group["encoding_seconds"],
        "p2b_int8": group["encoding_seconds"],
        "adaptive_int8_fp16": adaptive["representation_encoding_seconds"],
    }
    residuals = [
        row["state"][name]["solver_relative_residual"]
        for row in group["records"] for name in ("exact", "p2b_int8")
    ] + [row["solver_relative_residual"] for row in adaptive["records"]]
    summary["maximum_solver_relative_residual"] = max(residuals)
    summary["paired_aia_difference_from_exact_pp"] = {
        name: summary["validation_aia_percent"][name] - summary["validation_aia_percent"]["exact"]
        for name in METHODS
    }
    gates = {
        "train_only_cache": not (cache_dir / "test.pt").exists(),
        "solver_residual": summary["maximum_solver_relative_residual"] <= config["integrity_gates"]["maximum_solver_relative_residual"],
        "compressed_state_smaller_than_exact": all(
            summary["final_total_persistent_bytes"][name] < summary["final_total_persistent_bytes"]["exact"]
            for name in METHODS[1:]
        ),
        "accuracy_not_used_as_gate": True,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "stream_id": stream["stream_id"],
        "status": "PASS_M25_RANPAC_RESNET_CIFAR_TRAIN_ONLY" if passed else "FAIL_M25_RANPAC_RESNET_CIFAR_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "selected_ridge_lambda": ridge_lambda,
        "ridge_selection": ridge,
        "summary": summary,
        "gates": gates,
        "records": {
            "exact_and_p2b": group["records"],
            "adaptive_int8_fp16": adaptive["records"],
        },
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "stream_seed": seed,
            "config_sha256": _sha256_file(config_path),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "backend_sha256": _sha256_file(ROOT / "methods/analytic_ridge/backends.py"),
            "ranpac_frontend_sha256": _sha256_file(ROOT / "methods/frontends/ranpac.py"),
            "train_sha256": _sha256_file(cache_dir / "train.pt"),
            "train_features_sha256": metadata.get("train_features_sha256"),
            "train_labels_sha256": metadata.get("train_labels_sha256"),
            "projection_sha256": m4._tensor_content_sha256(projection),
            "class_order": order,
            "training_indices_sha256": _sequence_sha256(training_parts),
            "validation_indices_sha256": _sequence_sha256(validation_parts),
            "calibration_fit_indices_sha256": _tensor_sha256(fit_indices),
            "calibration_validation_indices_sha256": _tensor_sha256(calibration_validation),
        },
    }
    destination = output_dir / f"stream_{seed}_results.json"
    _atomic_json(destination, payload)
    del projection, backends, group, adaptive
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError(f"M25 stream {stream['stream_id']} failed structural gates")
    return payload


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _aggregate(config: dict, payloads: dict[str, dict]) -> dict:
    ordered = [payloads[item["stream_id"]] for item in config["streams"]]
    complete = all(
        _stream_complete(payload, stream)
        for payload, stream in zip(ordered, config["streams"])
    )
    aggregate = {}
    for method in METHODS:
        aggregate[method] = {
            "validation_aia_percent": _mean_std([p["summary"]["validation_aia_percent"][method] for p in ordered]),
            "final_validation_accuracy_percent": _mean_std([p["summary"]["final_validation_accuracy_percent"][method] for p in ordered]),
            "final_total_persistent_bytes": _mean_std([float(p["summary"]["final_total_persistent_bytes"][method]) for p in ordered]),
            "analytic_update_seconds": _mean_std([p["summary"]["analytic_update_seconds"][method] for p in ordered]),
            "paired_aia_difference_from_exact_pp": _mean_std([p["summary"]["paired_aia_difference_from_exact_pp"][method] for p in ordered]),
        }
    gates = {
        "all_six_streams_complete": complete,
        "train_only": all(
            p.get("uses_test_set") is False
            and p.get("gates", {}).get("train_only_cache") is True
            for p in ordered
        ),
        "accuracy_not_used_as_gate": True,
    }
    return {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M25_RANPAC_RESNET_CIFAR_TRAIN_ONLY" if all(gates.values()) else "FAIL_M25_RANPAC_RESNET_CIFAR_TRAIN_ONLY",
        "uses_test_set": False,
        "streams": [item["stream_id"] for item in config["streams"]],
        "aggregate": aggregate,
        "per_stream": payloads,
        "gates": gates,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    if args.require_clean_git:
        _clean_git()
    cache_dir = Path(args.feature_cache_dir).resolve()
    _load_train_cache(config, cache_dir)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("M25 requested CUDA but CUDA is unavailable")
    output_dir = Path(args.output_dir).resolve()
    stream_dir = output_dir / "streams"
    payloads = {}
    for stream in config["streams"]:
        destination = stream_dir / f"stream_{stream['seed']}_results.json"
        if destination.is_file():
            candidate = json.loads(destination.read_text(encoding="utf-8"))
            if _stream_complete(candidate, stream):
                print(f"M25 RESUME SKIP {stream['stream_id']}", flush=True)
                payloads[stream["stream_id"]] = candidate
                continue
        print(f"M25 START {stream['stream_id']}", flush=True)
        payloads[stream["stream_id"]] = _run_stream(config, config_path, stream, cache_dir, stream_dir, args.device)
    result = _aggregate(config, payloads)
    _atomic_json(output_dir / "m25_results.json", result)
    if result["status"] != "PASS_M25_RANPAC_RESNET_CIFAR_TRAIN_ONLY":
        raise RuntimeError("M25 aggregate structural gate failed")
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2), flush=True)
    return result


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-train")
    extract.add_argument("--config", required=True)
    extract.add_argument("--feature-cache-dir", required=True)
    extract.add_argument("--root", required=True)
    extract.add_argument("--backbone-checkpoint", required=True)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--batch-size", type=int, default=128)
    extract.add_argument("--num-workers", type=int, default=2)
    extract.add_argument("--download", action="store_true")
    extract.add_argument("--require-clean-git", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--feature-cache-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "extract-train":
        extract_train(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
