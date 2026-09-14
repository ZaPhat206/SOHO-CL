"""M18: locked FLY-CL width-20k adaptive-precision test closure.

The experiment compares three analytic Ridge backends under one shared FLY
projection and one shared set of WTA codes in every replicate.  The six paired
replicate identities and Ridge coefficient are inherited from the primary
train-only SRQ-FLY protocol.  Test accuracy is reported but never gates,
selects, retries, or excludes a method or seed.
"""

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
from models.backbone import load_model  # noqa: E402
from tools.srq_fly_heldout import (  # noqa: E402
    _assert_sample_free_inventory,
    _inventory,
    _mean_std_ci,
    _result_metrics,
    _task_predictions,
)
from tools.tail_fly_phasea import _load_unit, _save_unit, _unit_path  # noqa: E402
from tools.twa_fly_pilot import (  # noqa: E402
    _prepare_code_cache,
    _sequence_sha256,
)
from tools.experiment_runner import validate_cache  # noqa: E402
from utils.data_utils import load_dataset  # noqa: E402
from utils.train_utils import feature_extract, random_initialization  # noqa: E402


METHODS = (
    "exact_fly_20000",
    "srq_fly_p2b_20000",
    "srq_fly_adaptive_20000",
)
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "backbone", "representation",
    "methods", "replicates", "ridge", "selection_evidence",
    "square_root_backend", "adaptive", "evaluation", "source_identity",
    "integrity_gates",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_source_file(path: str | Path) -> str:
    """Hash committed text identically under Windows CRLF and Linux LF."""
    return _sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def _canonical_sha256(value) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_git() -> None:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"M18 requires a clean committed checkout:\n{dirty}")


def _verify_source_identity(config: dict) -> dict[str, str]:
    observed = {
        relative: _sha256_source_file(ROOT / relative)
        for relative in config["source_identity"]
    }
    if observed != config["source_identity"]:
        mismatches = {
            relative: {"expected": config["source_identity"][relative], "actual": digest}
            for relative, digest in observed.items()
            if digest != config["source_identity"][relative]
        }
        raise ValueError(f"M18 source identity mismatch: {mismatches}")
    return observed


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M18 config keys/schema mismatch")
    dataset = config["dataset"]
    backbone = config["backbone"]
    representation = config["representation"]
    if not (
        dataset == {
            "name": "CIFAR-100",
            "version": "torchvision-cifar100",
            "num_classes": 100,
            "num_tasks": 10,
            "expected_train_samples": 50000,
            "expected_test_samples": 10000,
        }
        and backbone["model_name"] == "vit_base_patch16_224"
        and backbone["feature_dim"] == 768
        and backbone["preprocessing"] == "vit"
        and representation["frontend"] == "fly_wta"
        and representation["expand_dim"] == 20000
        and representation["synaptic_degree"] == 300
        and representation["coding_level"] == 0.3
        and representation["absolute_wta"] is False
        and representation["statistics_dtype"] == "float32"
        and representation["solver_dtype"] == "float32"
        and min(
            representation["encode_batch_size"],
            representation["evaluation_batch_size"],
        ) > 0
        and tuple(config["methods"]) == METHODS
    ):
        raise ValueError("M18 locked dataset/frontend identity changed")
    expected_replicates = [
        {"class_order_seed": 3031 + index, "projection_seed": 5031 + index}
        for index in range(6)
    ]
    if config["replicates"] != expected_replicates:
        raise ValueError("M18 must preserve all six primary paired replicates")
    ridge = config["ridge"]
    if not (
        ridge["lambda"] == 1_000_000.0
        and ridge["policy"]
        == "inherit_locked_fly_family_lambda_from_primary_train_only_selection"
        and ridge["width_20000_retuned"] is False
        and ridge["interpretation"]
        == "controlled_backend_comparison_not_an_optimized_width_20000_frontend_claim"
    ):
        raise ValueError("M18 inherited Ridge contract changed")
    source = config["selection_evidence"]
    if not (
        source["artifact_filename"]
        == "srq_fly_selfcontained_three_dataset_results.zip"
        and source["required_status"] == "SELECTION_COMPLETE"
        and source["required_selected_fly_family_lambda"] == ridge["lambda"]
        and all(
            len(source[field]) == 64
            for field in (
                "artifact_sha256", "result_sha256", "required_protocol_sha256",
                "required_runner_sha256",
            )
        )
    ):
        raise ValueError("M18 selection-evidence lock changed")
    square_root = config["square_root_backend"]
    if square_root != {
        "block_size": 256,
        "group_size": 64,
        "update_backend": "blocked_qr",
        "update_panel_size": 128,
        "update_trailing_chunk_size": None,
        "first_update_backend": "gram_cholesky",
        "quantization_backend": "streaming",
        "quantization_batch_blocks": 64,
    }:
        raise ValueError("M18 square-root backend contract changed")
    adaptive = config["adaptive"]
    if not (
        adaptive["budget_fraction_between_int8_and_fp16"] == 0.25
        and adaptive["selection_rule"]
        == "largest_factor_mse_reduction_per_added_byte"
        and adaptive["selection_signal"]
        == "current_factor_values_only_no_labels_logits_predictions_or_accuracy"
        and adaptive["precision_mask_dtype"] == "uint8"
        and adaptive["tie_break"] == "ascending_upper_block_index"
    ):
        raise ValueError("M18 adaptive policy changed")
    evaluation = config["evaluation"]
    gates = config["integrity_gates"]
    if not (
        evaluation["uses_test_set"] is True
        and evaluation["test_tuning_allowed"] is False
        and evaluation["accuracy_based_selection"] is False
        and evaluation["accuracy_based_early_stop"] is False
        and evaluation["seed_exclusion_allowed"] is False
        and evaluation.get("accuracy_gate", "missing") is None
        and gates.get("accuracy_gate", "missing") is None
        and gates["require_all_six_paired_replicates"] is True
        and gates["require_no_seed_exclusion"] is True
        and gates["require_identical_codes_within_replicate"] is True
        and gates["require_adaptive_budget_conformance"] is True
        and float(gates["maximum_solver_relative_residual"]) > 0
    ):
        raise ValueError("M18 evaluation/integrity contract changed")
    _verify_source_identity(config)
    return config


def _load_selection(config: dict, artifact_path: str | Path) -> dict:
    source = config["selection_evidence"]
    path = Path(artifact_path)
    if path.name != source["artifact_filename"]:
        raise ValueError("M18 selection artifact filename mismatch")
    if _sha256_file(path) != source["artifact_sha256"]:
        raise ValueError("M18 selection artifact SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(source["result_member"])
    if _sha256_bytes(raw) != source["result_sha256"]:
        raise ValueError("M18 embedded selection SHA-256 mismatch")
    payload = json.loads(raw)
    if not (
        payload.get("status") == source["required_status"]
        and payload.get("uses_test_set") is False
        and payload.get("held_out_test_authorized") is False
        and payload.get("protocol_sha256") == source["required_protocol_sha256"]
        and payload.get("runner_sha256") == source["required_runner_sha256"]
        and float(payload.get("selected_fly_family_lambda"))
        == float(source["required_selected_fly_family_lambda"])
    ):
        raise ValueError("M18 embedded selection contract mismatch")
    return payload


def _cache_args(config: dict):
    class Args:
        pass

    args = Args()
    args.dataset = config["dataset"]["name"]
    args.model_name = config["backbone"]["model_name"]
    return args


def _validate_cache(
    config: dict, cache_dir: Path, *, require_test: bool
) -> tuple[dict, dict | None, dict]:
    test_path = cache_dir / "test.pt"
    if not require_test and test_path.exists():
        raise RuntimeError("M18 authorization requires test.pt to be absent")
    train, test, metadata = validate_cache(
        cache_dir, _cache_args(config), load_test=require_test
    )
    dataset = config["dataset"]
    backbone = config["backbone"]
    expected = (("train", train, dataset["expected_train_samples"]),)
    if require_test:
        expected += (("test", test, dataset["expected_test_samples"]),)
    for name, packed, rows in expected:
        if not (
            set(packed) == {"features", "labels"}
            and tuple(packed["features"].shape) == (rows, backbone["feature_dim"])
            and tuple(packed["labels"].shape) == (rows,)
            and bool(torch.isfinite(packed["features"]).all())
            and sorted(map(int, torch.unique(packed["labels"]).tolist()))
            == list(range(dataset["num_classes"]))
        ):
            raise ValueError(f"M18 invalid {name} feature cache")
    if not (
        metadata.get("checkpoint_sha256") == backbone["checkpoint_sha256"]
        and metadata.get("preprocessing") == backbone["preprocessing"]
    ):
        raise ValueError("M18 feature-cache metadata mismatch")
    if not require_test and metadata.get("test_features_materialized") is not False:
        raise ValueError("M18 train cache does not assert test absence")
    return train, test, metadata


def _authorization_identity(
    *, config_path: Path, config: dict, train: dict, selection: dict
) -> dict:
    return {
        "schema_version": 1,
        "study_id": config["study_id"],
        "git_commit": _git_commit(),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "config_sha256": _sha256_file(config_path),
        "source_identity": config["source_identity"],
        "selection_result_sha256": config["selection_evidence"]["result_sha256"],
        "selection_status": selection["status"],
        "selected_fly_family_lambda": float(
            selection["selected_fly_family_lambda"]
        ),
        "train_features_sha256": _tensor_sha256(train["features"]),
        "train_labels_sha256": _tensor_sha256(train["labels"]),
        "frozen_choices_sha256": _canonical_sha256({
            key: config[key]
            for key in (
                "dataset", "backbone", "representation", "methods", "replicates",
                "ridge", "square_root_backend", "adaptive", "evaluation",
                "integrity_gates",
            )
        }),
        "test_features_materialized_at_authorization": False,
    }


def authorize(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    if config["integrity_gates"]["require_clean_committed_checkout"]:
        _require_clean_git()
    selection = _load_selection(config, args.selection_artifact)
    train, _, _ = _validate_cache(
        config, Path(args.feature_cache_dir).resolve(), require_test=False
    )
    identity = _authorization_identity(
        config_path=config_path, config=config, train=train, selection=selection
    )
    payload = {
        "authorization_id": _canonical_sha256(identity),
        "authorized": True,
        "uses_test_set": False,
        "test_tuning_allowed": False,
        "accuracy_based_selection": False,
        "identity": identity,
    }
    destination = Path(args.authorization).resolve()
    if destination.exists() and json.loads(
        destination.read_text(encoding="utf-8")
    ) != payload:
        raise RuntimeError("existing M18 authorization has a different identity")
    _atomic_json(destination, payload)
    print(f"M18 AUTHORIZED id={payload['authorization_id']}", flush=True)
    return payload


def _read_authorization(args, config: dict, *, require_test: bool) -> tuple[dict, dict, dict | None, dict]:
    path = Path(args.authorization).resolve()
    if not path.is_file():
        raise FileNotFoundError("M18 authorization is required")
    authorization = json.loads(path.read_text(encoding="utf-8"))
    train, test, metadata = _validate_cache(
        config, Path(args.feature_cache_dir).resolve(), require_test=require_test
    )
    selection = _load_selection(config, args.selection_artifact)
    expected = _authorization_identity(
        config_path=Path(args.config).resolve(),
        config=config,
        train=train,
        selection=selection,
    )
    if not (
        authorization.get("authorized") is True
        and authorization.get("uses_test_set") is False
        and authorization.get("test_tuning_allowed") is False
        and authorization.get("identity") == expected
        and authorization.get("authorization_id") == _canonical_sha256(expected)
    ):
        raise RuntimeError("M18 authorization identity mismatch")
    if require_test and metadata.get("m18_authorization_id") != authorization[
        "authorization_id"
    ]:
        raise RuntimeError("M18 test cache is not bound to this authorization")
    return authorization, train, test, metadata


def extract_test(args) -> dict:
    config = _read_config(args.config)
    if config["integrity_gates"]["require_clean_committed_checkout"]:
        _require_clean_git()
    cache_dir = Path(args.feature_cache_dir).resolve()
    if (cache_dir / "test.pt").exists():
        authorization, _, test, _ = _read_authorization(
            args, config, require_test=True
        )
        print(
            f"M18 TEST CACHE RESTORED samples={len(test['labels'])} "
            f"authorization={authorization['authorization_id']}", flush=True,
        )
        return {"status": "restored"}
    authorization, _, _, _ = _read_authorization(
        args, config, require_test=False
    )
    random_initialization(2025)
    device = torch.device(args.device)
    dataset = config["dataset"]
    backbone_config = config["backbone"]
    loader_args = argparse.Namespace(
        dataset=dataset["name"], root=args.root,
        num_classes=dataset["num_classes"], num_tasks=dataset["num_tasks"],
        batch_size=args.batch_size,
        data_augmentation=backbone_config["preprocessing"],
        num_workers=args.num_workers,
    )
    _, test_loaders = load_dataset(loader_args)
    backbone = load_model(
        backbone_config["model_name"],
        checkpoint_path=args.backbone_checkpoint,
        expected_checkpoint_size=backbone_config["checkpoint_size"],
        expected_checkpoint_sha256=backbone_config["checkpoint_sha256"],
    ).eval().to(device)
    feature_parts, label_parts = [], []
    for task in range(dataset["num_tasks"]):
        values, labels = feature_extract(backbone, test_loaders[task], device)
        feature_parts.append(values.cpu())
        label_parts.append(labels.cpu())
        print(
            f"M18 TEST FEATURE TASK {task + 1}/{dataset['num_tasks']} "
            f"samples={len(labels)}", flush=True,
        )
    test = {"features": torch.cat(feature_parts), "labels": torch.cat(label_parts)}
    if not (
        tuple(test["features"].shape)
        == (dataset["expected_test_samples"], backbone_config["feature_dim"])
        and tuple(test["labels"].shape) == (dataset["expected_test_samples"],)
        and bool(torch.isfinite(test["features"]).all())
        and sorted(map(int, torch.unique(test["labels"]).tolist()))
        == list(range(dataset["num_classes"]))
    ):
        raise RuntimeError("M18 extracted test feature inventory is invalid")
    test_path = cache_dir / "test.pt"
    _atomic_torch(test_path, test)
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "test_shape": list(test["features"].shape),
        "test_labels_shape": list(test["labels"].shape),
        "test_features_materialized": True,
        "m18_authorization_id": authorization["authorization_id"],
        "test_features_sha256": _tensor_sha256(test["features"]),
        "test_labels_sha256": _tensor_sha256(test["labels"]),
    })
    _atomic_json(metadata_path, metadata)
    print("M18 AUTHORIZED TEST FEATURE CACHE READY", flush=True)
    return {"status": "complete"}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _backend(config: dict, method: str, device: torch.device):
    common = {
        "dimension": int(config["representation"]["expand_dim"]),
        "ridge_lambda": float(config["ridge"]["lambda"]),
        "device": device,
        "statistics_dtype": torch.float32,
        "solver_dtype": torch.float32,
    }
    if method == "exact_fly_20000":
        return ExactGramBackend(**common)
    backend = config["square_root_backend"]
    storage_mode = {
        "srq_fly_p2b_20000": "int8",
        "srq_fly_adaptive_20000": "adaptive_int8_fp16",
    }[method]
    kwargs = {
        **common,
        "storage_mode": storage_mode,
        "block_size": int(backend["block_size"]),
        "group_size": int(backend["group_size"]),
        "update_panel_size": int(backend["update_panel_size"]),
        "update_trailing_chunk_size": backend["update_trailing_chunk_size"],
        "first_update_backend": backend["first_update_backend"],
        "quantization_backend": backend["quantization_backend"],
        "quantization_batch_blocks": int(backend["quantization_batch_blocks"]),
    }
    if storage_mode == "adaptive_int8_fp16":
        kwargs["adaptive_budget_fraction"] = float(
            config["adaptive"]["budget_fraction_between_int8_and_fp16"]
        )
    return SquareRootBackend(**kwargs)


def _factor_bytes(backend: SquareRootBackend) -> int:
    return persistent_tensor_bytes({
        name: tensor
        for name, tensor in backend.persistent_tensors().items()
        if name.startswith("factor.")
    })


def _cache_config(config: dict, projection_seed: int) -> dict:
    representation = config["representation"]
    return {
        "seed": int(projection_seed),
        "num_classes": int(config["dataset"]["num_classes"]),
        "representation": {
            "expand_dim": int(representation["expand_dim"]),
            "synaptic_degree": int(representation["synaptic_degree"]),
            "coding_level": float(representation["coding_level"]),
            "encode_batch_size": int(representation["encode_batch_size"]),
            "evaluation_batch_size": int(representation["evaluation_batch_size"]),
        },
        "statistics_dtype": representation["statistics_dtype"],
        "raw_ridge_lambda": 1.0,
        "solver_tolerance": float(
            config["integrity_gates"]["maximum_solver_relative_residual"]
        ),
        "solver_max_iterations": 100,
    }


def _expected_code_identity(
    config: dict, projection_seed: int, source_tensor_sha256: str, sample_count: int
) -> tuple[dict, str]:
    representation = config["representation"]
    identity = {
        "raw_dim": int(config["backbone"]["feature_dim"]),
        "expand_dim": int(representation["expand_dim"]),
        "synaptic_degree": int(representation["synaptic_degree"]),
        "coding_level": float(representation["coding_level"]),
        "seed": int(projection_seed),
        "statistics_dtype": representation["statistics_dtype"],
        "source_train_sha256": source_tensor_sha256,
        "sample_count": int(sample_count),
    }
    identity_sha256 = _sha256_bytes(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    )
    return identity, identity_sha256


def _relative_logit_error(reference: list[torch.Tensor], value: list[torch.Tensor]) -> float:
    numerator = sum(
        float(((right - left) ** 2).sum())
        for left, right in zip(reference, value)
    )
    denominator = max(sum(float((left ** 2).sum()) for left in reference), 1.0)
    return math.sqrt(numerator / denominator)


def _prediction_agreement(reference: list[torch.Tensor], value: list[torch.Tensor]) -> float:
    return statistics.fmean([
        float((left == right).float().mean())
        for left, right in zip(reference, value)
    ])


def _evaluate_replicate(
    *, config: dict, stream: dict, code_indices: torch.Tensor,
    code_values: torch.Tensor, projection: torch.Tensor,
    training_parts: list[torch.Tensor], test_parts: list[torch.Tensor],
    device: torch.device, replicate_index: int,
) -> dict:
    dimension = int(config["representation"]["expand_dim"])
    evaluation_batch_size = int(
        config["representation"]["evaluation_batch_size"]
    )
    backends = {method: _backend(config, method, device) for method in METHODS}
    matrices = {method: [] for method in METHODS}
    states = {method: [] for method in METHODS}
    residuals = {method: [] for method in METHODS}
    timings = {method: [] for method in METHODS}
    diagnostics = {method: [] for method in METHODS}
    comparisons = {
        method: {"prediction_agreement_by_task": [], "relative_logit_error_by_task": []}
        for method in METHODS[1:]
    }
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        code_started = time.perf_counter()
        codes = torch.zeros(
            (len(indices), dimension), device=device, dtype=torch.float32
        )
        selected_indices = code_indices[indices].to(device=device, dtype=torch.long)
        selected_values = code_values[indices].to(device=device, dtype=torch.float32)
        codes.scatter_(1, selected_indices, selected_values)
        del selected_indices, selected_values
        labels = stream["labels"][indices]
        _sync(device)
        code_seconds = time.perf_counter() - code_started
        for method, backend in backends.items():
            _sync(device)
            update_started = time.perf_counter()
            backend.update(codes, labels)
            _sync(device)
            timings[method].append({
                "task": task + 1,
                "shared_code_materialization_seconds": code_seconds,
                "update_seconds": time.perf_counter() - update_started,
            })
            backend.assert_exemplar_free_state()
            residuals[method].append(
                float(backend.diagnostics["solver_relative_residual"])
            )
        class_ids = backends[METHODS[0]].class_ids
        if any(backend.class_ids != class_ids for backend in backends.values()):
            raise AssertionError("M18 backend class-column order diverged")
        del codes
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        predictions, logits = {}, {}
        for method, backend in backends.items():
            _sync(device)
            inference_started = time.perf_counter()
            row, method_predictions, method_logits = _task_predictions(
                weights=backend.weights,
                class_ids=backend.class_ids,
                parts=test_parts,
                task=task,
                code_indices=code_indices,
                code_values=code_values,
                labels=stream["labels"],
                dimension=dimension,
                batch_size=evaluation_batch_size,
            )
            _sync(device)
            timings[method][-1]["inference_seconds"] = (
                time.perf_counter() - inference_started
            )
            matrices[method].append(row)
            predictions[method] = method_predictions
            logits[method] = method_logits
            inventory = _inventory({"projection": projection, **backend.persistent_tensors()})
            _assert_sample_free_inventory(
                inventory,
                backend.total_rows,
                {dimension, config["backbone"]["feature_dim"], len(class_ids)},
            )
            states[method].append(sum(item["bytes"] for item in inventory))
            record = {
                "task": task + 1,
                "solver_relative_residual": residuals[method][-1],
                "persistent_state_bytes": states[method][-1],
            }
            if method == "srq_fly_adaptive_20000":
                factor_bytes = _factor_bytes(backend)
                total_blocks = int(backend.diagnostics["total_blocks"])
                all_int8_factor_bytes = (
                    4 * dimension
                    + total_blocks
                    + int(backend.diagnostics["int8_strict_upper_bytes"])
                )
                all_fp16_factor_bytes = (
                    4 * dimension
                    + total_blocks
                    + int(backend.diagnostics["fp16_strict_upper_bytes"])
                )
                record.update({
                    "factor_persistent_bytes": factor_bytes,
                    "factor_budget_ceiling_bytes": int(
                        backend.diagnostics["factor_budget_ceiling_bytes"]
                    ),
                    "factor_all_int8_bytes": all_int8_factor_bytes,
                    "factor_all_fp16_bytes": all_fp16_factor_bytes,
                    "selected_fp16_blocks": int(
                        backend.diagnostics["selected_fp16_blocks"]
                    ),
                    "selected_fp16_values": int(
                        backend.diagnostics["selected_fp16_values"]
                    ),
                    "total_blocks": total_blocks,
                    "used_extra_bytes": int(
                        backend.diagnostics["used_extra_bytes"]
                    ),
                    "extra_budget_bytes": int(
                        backend.diagnostics["extra_budget_bytes"]
                    ),
                    "relative_local_factor_error": float(
                        backend.diagnostics["relative_local_factor_error"]
                    ),
                })
                if factor_bytes != int(
                    backend.diagnostics["factor_persistent_bytes"]
                ):
                    raise AssertionError("M18 adaptive factor accounting mismatch")
            diagnostics[method].append(record)
        for method in METHODS[1:]:
            comparisons[method]["prediction_agreement_by_task"].append(
                _prediction_agreement(predictions[METHODS[0]], predictions[method])
            )
            comparisons[method]["relative_logit_error_by_task"].append(
                _relative_logit_error(logits[METHODS[0]], logits[method])
            )
        print(
            f"M18 replicate={replicate_index + 1}/6 task={task + 1}/10 "
            f"exact={statistics.fmean(matrices[METHODS[0]][-1]):.4f} "
            f"int8={statistics.fmean(matrices[METHODS[1]][-1]):.4f} "
            f"adaptive={statistics.fmean(matrices[METHODS[2]][-1]):.4f}",
            flush=True,
        )
        del predictions, logits
        gc.collect()

    method_results = {}
    for method, backend in backends.items():
        metrics = _result_metrics(matrices[method])
        method_results[method] = {
            "status": "complete",
            "method": method,
            "uses_test_set": True,
            "test_tuning_allowed": False,
            "ridge_lambda": float(config["ridge"]["lambda"]),
            **metrics,
            "accuracy_matrix": matrices[method],
            "persistent_state_bytes": states[method][-1],
            "persistent_state_bytes_by_task": states[method],
            "persistent_tensor_inventory": _inventory({
                "projection": projection, **backend.persistent_tensors()
            }),
            "maximum_solver_relative_residual": max(residuals[method]),
            "total_update_seconds": sum(x["update_seconds"] for x in timings[method]),
            "total_inference_seconds": sum(
                x["inference_seconds"] for x in timings[method]
            ),
            "timing": timings[method],
            "diagnostics_by_task": diagnostics[method],
        }
        if method != METHODS[0]:
            method_results[method].update({
                "minimum_prediction_agreement": min(
                    comparisons[method]["prediction_agreement_by_task"]
                ),
                "maximum_relative_logit_frobenius_error": max(
                    comparisons[method]["relative_logit_error_by_task"]
                ),
            })
    return {
        "status": "complete",
        "uses_test_set": True,
        "methods": method_results,
        "paired_diagnostics": comparisons,
        "wall_seconds": time.perf_counter() - started,
    }


def _parts(config: dict, train: dict, test: dict, replicate: dict):
    dataset = config["dataset"]
    class_order = random.Random(replicate["class_order_seed"]).sample(
        range(dataset["num_classes"]), dataset["num_classes"]
    )
    classes_per_task = dataset["num_classes"] // dataset["num_tasks"]
    training_parts, test_parts = [], []
    offset = len(train["labels"])
    for task in range(dataset["num_tasks"]):
        class_ids = torch.tensor(
            class_order[task * classes_per_task:(task + 1) * classes_per_task]
        )
        training_parts.append(
            torch.nonzero(torch.isin(train["labels"], class_ids)).flatten()
        )
        test_parts.append(
            torch.nonzero(torch.isin(test["labels"], class_ids)).flatten() + offset
        )
    return class_order, training_parts, test_parts


def _run_unit(path: Path, context_sha256: str, label: str, evaluator) -> dict:
    restored = _load_unit(path, context_sha256)
    if restored is not None:
        return restored
    print(f"M18 START {label}", flush=True)
    try:
        result = evaluator()
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        result = {
            "status": "numerical_failure",
            "uses_test_set": True,
            "failure": f"{type(error).__name__}: {error}",
        }
    saved = _save_unit(path, context_sha256, result)
    print(f"M18 DONE {label} status={saved['status']}", flush=True)
    return saved


def _write_progress(output_dir: Path, config: dict, seed_results: list[dict]) -> None:
    _atomic_json(output_dir / "m18_progress.json", {
        "schema_version": 1,
        "study_id": config["study_id"],
        "completed_replicates": len(seed_results),
        "required_replicates": len(config["replicates"]),
        "completed_replicate_indices": [
            item["replicate_index"] for item in seed_results
        ],
        "seed_results": seed_results,
    })


def _summarize_results(config: dict, seed_results: list[dict]) -> tuple[dict, dict]:
    method_summaries = {}
    for method in METHODS:
        results = [item["methods"][method] for item in seed_results]
        method_summaries[method] = {
            metric: _mean_std_ci([float(result[metric]) for result in results])
            for metric in (
                "final_accuracy", "average_incremental_accuracy", "forgetting",
                "persistent_state_bytes", "total_update_seconds",
                "total_inference_seconds", "maximum_solver_relative_residual",
            )
        }
    paired = {}
    for method in METHODS[1:]:
        paired[method] = {
            metric: _mean_std_ci([
                float(item["methods"][method][metric])
                - float(item["methods"][METHODS[0]][metric])
                for item in seed_results
            ])
            for metric in ("final_accuracy", "average_incremental_accuracy")
        }
    all_complete = (
        len(seed_results) == 6
        and [item["replicate_index"] for item in seed_results] == list(range(6))
        and all(
            item["methods"][method].get("status") == "complete"
            for item in seed_results for method in METHODS
        )
    )
    state_ordering = all(
        exact > adaptive > int8
        for item in seed_results
        for exact, adaptive, int8 in zip(
            item["methods"][METHODS[0]]["persistent_state_bytes_by_task"],
            item["methods"][METHODS[2]]["persistent_state_bytes_by_task"],
            item["methods"][METHODS[1]]["persistent_state_bytes_by_task"],
        )
    ) if all_complete else False
    budget_conformance = all(
        record["factor_all_int8_bytes"]
        <= record["factor_persistent_bytes"]
        <= record["factor_budget_ceiling_bytes"]
        and record["used_extra_bytes"] <= record["extra_budget_bytes"]
        for item in seed_results
        for record in item["methods"][METHODS[2]]["diagnostics_by_task"]
    ) if all_complete else False
    maximum_residual = max(
        item["methods"][method]["maximum_solver_relative_residual"]
        for item in seed_results for method in METHODS
    ) if all_complete else math.inf
    gates = {
        "all_six_paired_replicates_complete": all_complete,
        "no_seed_excluded": all_complete,
        "identical_wta_codes_within_each_replicate": all_complete,
        "exact_state_larger_than_adaptive_larger_than_int8": state_ordering,
        "adaptive_budget_conformance": budget_conformance,
        "solver_relative_residual": maximum_residual
        <= float(config["integrity_gates"]["maximum_solver_relative_residual"]),
        "accuracy_gate": None,
    }
    return {
        "methods": method_summaries,
        "paired_minus_exact": paired,
        "maximum_solver_relative_residual": maximum_residual,
    }, gates


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    if args.max_new_replicates is not None and args.max_new_replicates <= 0:
        raise ValueError("--max-new-replicates must be positive")
    if config["integrity_gates"]["require_clean_committed_checkout"]:
        _require_clean_git()
    authorization, train, test, metadata = _read_authorization(
        args, config, require_test=True
    )
    selection = _load_selection(config, args.selection_artifact)
    stream = {
        "features": torch.cat((train["features"], test["features"])),
        "labels": torch.cat((train["labels"], test["labels"])),
    }
    source_tensor_sha = _canonical_sha256({
        "train_features": _tensor_sha256(train["features"]),
        "train_labels": _tensor_sha256(train["labels"]),
        "test_features": _tensor_sha256(test["features"]),
        "test_labels": _tensor_sha256(test["labels"]),
    })
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    code_cache_root = Path(args.code_cache_root).resolve()
    seed_results = []
    new_replicates = 0
    for replicate_index, replicate in enumerate(config["replicates"]):
        class_order, training_parts, test_parts = _parts(
            config, train, test, replicate
        )
        code_identity, code_identity_sha256 = _expected_code_identity(
            config,
            replicate["projection_seed"],
            source_tensor_sha,
            len(stream["features"]),
        )
        context = {
            "config_sha256": _sha256_file(config_path),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "authorization_id": authorization["authorization_id"],
            "selection_result_sha256": config["selection_evidence"]["result_sha256"],
            "selected_fly_family_lambda": selection["selected_fly_family_lambda"],
            "replicate_index": replicate_index,
            "replicate": replicate,
            "class_order": class_order,
            "training_parts_sha256": _sequence_sha256(training_parts),
            "test_parts_sha256": _sequence_sha256(test_parts),
            "source_tensor_sha256": source_tensor_sha,
            "code_identity": code_identity,
            "code_identity_sha256": code_identity_sha256,
            "methods": list(METHODS),
        }
        context_sha256 = _canonical_sha256(context)
        unit_path = _unit_path(output_dir, f"replicate_{replicate_index}_paired")
        unit = _load_unit(unit_path, context_sha256)
        cache = None
        if unit is None:
            if (
                args.max_new_replicates is not None
                and new_replicates >= args.max_new_replicates
            ):
                break
            cache = _prepare_code_cache(
                train=stream,
                train_sha256=source_tensor_sha,
                cache_dir=code_cache_root / f"replicate_{replicate_index}",
                config=_cache_config(config, replicate["projection_seed"]),
                device=device,
            )
            if (
                cache[2].get("identity") != code_identity
                or cache[2].get("identity_sha256") != code_identity_sha256
            ):
                raise RuntimeError("M18 WTA cache identity mismatch")
            unit = _run_unit(
                unit_path,
                context_sha256,
                f"replicate={replicate_index + 1}/6 exact+int8+adaptive",
                lambda cache=cache, training_parts=training_parts,
                test_parts=test_parts, replicate_index=replicate_index: _evaluate_replicate(
                    config=config,
                    stream=stream,
                    code_indices=cache[0],
                    code_values=cache[1],
                    projection=cache[3],
                    training_parts=training_parts,
                    test_parts=test_parts,
                    device=device,
                    replicate_index=replicate_index,
                ),
            )
            new_replicates += 1
        if unit.get("status") != "complete":
            seed_results.append({
                "replicate_index": replicate_index,
                "class_order_seed": replicate["class_order_seed"],
                "projection_seed": replicate["projection_seed"],
                "class_order": class_order,
                "status": unit.get("status"),
                "failure": unit.get("failure"),
            })
            _write_progress(output_dir, config, seed_results)
            raise RuntimeError(
                f"M18 replicate {replicate_index} failed: {unit.get('failure')}"
            )
        seed_results.append({
            "replicate_index": replicate_index,
            "class_order_seed": replicate["class_order_seed"],
            "projection_seed": replicate["projection_seed"],
            "class_order": class_order,
            "status": "complete",
            "methods": unit["methods"],
            "paired_diagnostics": unit["paired_diagnostics"],
            "wall_seconds": unit["wall_seconds"],
        })
        _write_progress(output_dir, config, seed_results)
        print(f"M18 REPLICATE CHECKPOINT {replicate_index + 1}/6", flush=True)
        del cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(seed_results) < len(config["replicates"]):
        partial = {
            "schema_version": 1,
            "study_id": config["study_id"],
            "status": "M18_PARTIAL_CHECKPOINT",
            "completed_replicates": len(seed_results),
            "required_replicates": len(config["replicates"]),
            "new_replicates_this_invocation": new_replicates,
            "accuracy_gate": None,
        }
        _atomic_json(output_dir / "m18_partial_status.json", partial)
        print(
            f"M18 PARTIAL CHECKPOINT {len(seed_results)}/"
            f"{len(config['replicates'])}", flush=True,
        )
        return partial

    summary, gates = _summarize_results(config, seed_results)
    status = (
        "PASS_M18_FLY20K_ADAPTIVE_LOCKED_TEST"
        if all(value is True for key, value in gates.items() if key != "accuracy_gate")
        else "FAIL_M18_FLY20K_ADAPTIVE_LOCKED_TEST"
    )
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": status,
        "uses_test_set": True,
        "test_tuning_allowed": False,
        "accuracy_based_selection": False,
        "accuracy_based_early_stop": False,
        "seed_exclusion_allowed": False,
        "accuracy_gate": None,
        "test_use_disclosure": config["evaluation"]["test_use_disclosure"],
        "authorization_id": authorization["authorization_id"],
        "config_sha256": _sha256_file(config_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "source_identity": config["source_identity"],
        "source_feature_metadata": metadata,
        "selection_evidence": config["selection_evidence"],
        "ridge": config["ridge"],
        "representation": config["representation"],
        "methods": list(METHODS),
        "summary": summary,
        "gates": gates,
        "seed_results": seed_results,
    }
    _atomic_json(output_dir / "m18_results.json", payload)
    _write_tables(output_dir, payload)
    print(f"M18 STATUS: {status}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(gates, indent=2), flush=True)
    return payload


def _write_tables(output_dir: Path, payload: dict) -> None:
    metric_rows, curve_rows, diagnostic_rows = [], [], []
    for replicate in payload["seed_results"]:
        for method, result in replicate["methods"].items():
            metric_rows.append({
                "replicate_index": replicate["replicate_index"],
                "class_order_seed": replicate["class_order_seed"],
                "projection_seed": replicate["projection_seed"],
                "method": method,
                "final_accuracy": result["final_accuracy"],
                "average_incremental_accuracy": result[
                    "average_incremental_accuracy"
                ],
                "forgetting": result["forgetting"],
                "persistent_state_bytes": result["persistent_state_bytes"],
                "total_update_seconds": result["total_update_seconds"],
                "total_inference_seconds": result["total_inference_seconds"],
                "maximum_solver_relative_residual": result[
                    "maximum_solver_relative_residual"
                ],
            })
            for task, accuracy in enumerate(result["stage_accuracy"], 1):
                curve_rows.append({
                    "replicate_index": replicate["replicate_index"],
                    "method": method,
                    "task": task,
                    "average_seen_accuracy": accuracy,
                })
            for record in result["diagnostics_by_task"]:
                diagnostic_rows.append({
                    "replicate_index": replicate["replicate_index"],
                    "method": method,
                    **record,
                })
    for filename, rows in (
        ("m18_metrics.csv", metric_rows),
        ("m18_task_curves.csv", curve_rows),
        ("m18_diagnostics.csv", diagnostic_rows),
    ):
        fields = sorted({key for row in rows for key in row})
        with (output_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("authorize", "extract-test", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--selection-artifact", required=True)
        child.add_argument("--feature-cache-dir", required=True)
        child.add_argument("--authorization", required=True)
        if name in {"extract-test", "run"}:
            child.add_argument("--device", default="cpu")
        if name == "extract-test":
            child.add_argument("--root", required=True)
            child.add_argument("--backbone-checkpoint", required=True)
            child.add_argument("--batch-size", type=int, default=128)
            child.add_argument("--num-workers", type=int, default=2)
        if name == "run":
            child.add_argument("--code-cache-root", required=True)
            child.add_argument("--output-dir", required=True)
            child.add_argument("--max-new-replicates", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "authorize":
        authorize(args)
    elif args.command == "extract-test":
        extract_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
