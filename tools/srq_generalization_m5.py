"""Train-only equal-budget/Pareto gate for the reusable SRQ backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import (  # noqa: E402
    DenseSquareRootBackend,
    ExactGramBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from methods.frontends import CountSketchAnalyticLearner, RanPACAnalyticLearner  # noqa: E402
from tools.experiment_runner import split, train_validation_indices, validate_cache  # noqa: E402
from tools import srq_generalization_m4 as m4  # noqa: E402


TOP_KEYS = {
    "schema_version",
    "study_id",
    "dataset",
    "model_name",
    "checkpoint_sha256",
    "uses_test_set",
    "accuracy_based_selection",
    "seed",
    "num_classes",
    "num_tasks",
    "outer_validation_fraction",
    "statistics_dtype",
    "solver_dtype",
    "ranpac",
    "ridge_selection",
    "p2b",
    "countsketch",
    "budget",
    "gates",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(parts: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.to(torch.int64).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M5 config keys/schema mismatch")
    if config["uses_test_set"] is not False:
        raise ValueError("M5 must remain train-only")
    if config["accuracy_based_selection"] is not False:
        raise ValueError("M5 cannot select dimensions using accuracy")
    if config["seed"] != 2025:
        raise ValueError("M5 seed must remain 2025")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("M5 locks FP32 statistics and solves")
    if (
        config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or not 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("invalid M5 class/task split")

    ranpac = config["ranpac"]
    required_ranpac = {
        "upstream_repository",
        "upstream_commit",
        "upstream_ranpac_py_sha256",
        "path",
        "feature_dimension",
        "full_expand_dimension",
        "projection_distribution",
        "activation",
        "projection_seed",
        "encode_batch_size",
        "evaluation_batch_size",
    }
    if set(ranpac) != required_ranpac:
        raise ValueError("M5 RanPAC fields mismatch")
    if (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or ranpac["projection_seed"] != 2025
        or min(
            ranpac[name]
            for name in (
                "feature_dimension",
                "full_expand_dimension",
                "encode_batch_size",
                "evaluation_batch_size",
            )
        )
        <= 0
        or len(ranpac["upstream_commit"]) != 40
        or len(ranpac["upstream_ranpac_py_sha256"]) != 64
    ):
        raise ValueError("invalid locked M5 RanPAC semantics")

    selection = config["ridge_selection"]
    if set(selection) != {
        "policy",
        "candidate_lambdas",
        "fit_samples_per_class",
        "validation_samples_per_class",
        "metric",
        "tie_break",
    }:
        raise ValueError("M5 Ridge-selection fields mismatch")
    candidates = list(map(float, selection["candidate_lambdas"]))
    if (
        selection["policy"] != "per_representation_train_only_calibration"
        or selection["metric"] != "mean_squared_error"
        or selection["tie_break"] != "smallest_lambda"
        or not candidates
        or candidates != sorted(set(candidates))
        or min(candidates) <= 0
        or min(
            selection[name]
            for name in ("fit_samples_per_class", "validation_samples_per_class")
        )
        <= 0
    ):
        raise ValueError("invalid M5 Ridge selection")

    p2b = config["p2b"]
    if set(p2b) != {
        "block_size",
        "group_size",
        "update_panel_size",
        "update_trailing_chunk_size",
        "first_update_backend",
        "quantization_backend",
        "quantization_batch_blocks",
    }:
        raise ValueError("M5 P2B fields mismatch")
    if (
        p2b["first_update_backend"] != "gram_cholesky"
        or p2b["quantization_backend"] != "streaming"
        or min(
            p2b[name]
            for name in (
                "block_size",
                "group_size",
                "update_panel_size",
                "quantization_batch_blocks",
            )
        )
        <= 0
    ):
        raise ValueError("M5 no longer matches frozen P2B")
    if p2b["update_trailing_chunk_size"] is not None and p2b[
        "update_trailing_chunk_size"
    ] <= 0:
        raise ValueError("invalid M5 trailing chunk size")

    sketch = config["countsketch"]
    if set(sketch) != {
        "family",
        "seed",
        "bucket_dtype",
        "sign_dtype",
        "normalization",
    } or sketch != {
        "family": "one_nonzero_signed_hash",
        "seed": 2025,
        "bucket_dtype": "int32",
        "sign_dtype": "int8",
        "normalization": "none",
    }:
        raise ValueError("invalid M5 CountSketch contract")

    budget = config["budget"]
    if set(budget) != {
        "target",
        "derive_dimensions_before_accuracy",
        "include_projection",
        "include_cross_counts_and_classifier",
        "maximum_underfill_fraction",
    }:
        raise ValueError("M5 budget fields mismatch")
    if (
        budget["target"]
        != "full_width_p2b_final_total_persistent_tensor_bytes"
        or budget["derive_dimensions_before_accuracy"] is not True
        or budget["include_projection"] is not True
        or budget["include_cross_counts_and_classifier"] is not True
        or not 0 <= budget["maximum_underfill_fraction"] < 1
    ):
        raise ValueError("invalid M5 budget policy")

    gates = config["gates"]
    if set(gates) != {
        "require_byte_derived_dimensions",
        "maximum_p2b_validation_aia_loss_pp",
        "maximum_solver_relative_residual",
        "maximum_equal_or_lower_state_aia_advantage_pp",
        "fail_if_alternative_pareto_dominates_srq",
    }:
        raise ValueError("M5 gate fields mismatch")
    if (
        gates["require_byte_derived_dimensions"] is not True
        or gates["fail_if_alternative_pareto_dominates_srq"] is not True
        or min(
            gates[name]
            for name in (
                "maximum_p2b_validation_aia_loss_pp",
                "maximum_solver_relative_residual",
                "maximum_equal_or_lower_state_aia_advantage_pp",
            )
        )
        < 0
    ):
        raise ValueError("invalid M5 gates")
    return config


def _compressed_upper_bytes(
    dimension: int, *, block_size: int, group_size: int, mode: str
) -> int:
    if dimension <= 0 or block_size <= 0 or group_size <= 0:
        raise ValueError("storage dimensions must be positive")
    if mode not in {"int8", "float16"}:
        raise ValueError("unsupported square-root storage mode")
    values = 0
    scales = 0
    block_count = math.ceil(dimension / block_size)
    for row_block in range(block_count):
        row_start = row_block * block_size
        rows = min(row_start + block_size, dimension) - row_start
        for column_block in range(row_block, block_count):
            column_start = column_block * block_size
            columns = min(column_start + block_size, dimension) - column_start
            count = (
                rows * (rows - 1) // 2
                if row_block == column_block
                else rows * columns
            )
            values += count
            scales += math.ceil(count / group_size)
    diagonal_bytes = 4 * dimension
    if mode == "float16":
        return diagonal_bytes + 2 * values
    return diagonal_bytes + values + 4 * scales


def _exact_backend_bytes(dimension: int, classes: int) -> int:
    return 4 * dimension * dimension + 8 * dimension * classes + 4 * classes


def _square_root_backend_bytes(
    dimension: int,
    classes: int,
    *,
    block_size: int,
    group_size: int,
    mode: str,
) -> int:
    return (
        _compressed_upper_bytes(
            dimension,
            block_size=block_size,
            group_size=group_size,
            mode=mode,
        )
        + 8 * dimension * classes
        + 4 * classes
    )


def _derive_budget_lock(config: dict) -> dict:
    ranpac = config["ranpac"]
    p2b = config["p2b"]
    feature_dimension = int(ranpac["feature_dimension"])
    full_dimension = int(ranpac["full_expand_dimension"])
    classes = int(config["num_classes"])
    projection_bytes = 4 * feature_dimension * full_dimension
    target = projection_bytes + _square_root_backend_bytes(
        full_dimension,
        classes,
        block_size=int(p2b["block_size"]),
        group_size=int(p2b["group_size"]),
        mode="int8",
    )

    reduced_dimension = 0
    reduced_bytes = 0
    for dimension in range(1, full_dimension + 1):
        candidate = (
            4 * feature_dimension * dimension
            + _exact_backend_bytes(dimension, classes)
        )
        if candidate <= target:
            reduced_dimension, reduced_bytes = dimension, candidate

    sketch_dimension = 0
    sketch_bytes = 0
    sketch_map_bytes = 4 * full_dimension + full_dimension
    for dimension in range(1, full_dimension + 1):
        candidate = (
            projection_bytes
            + sketch_map_bytes
            + _exact_backend_bytes(dimension, classes)
        )
        if candidate <= target:
            sketch_dimension, sketch_bytes = dimension, candidate

    if not reduced_dimension or not sketch_dimension:
        raise RuntimeError("M5 byte budget cannot fit an alternative")
    return {
        "policy": "largest_integer_dimension_not_exceeding_target_bytes",
        "locked_before_representation_encoding_or_accuracy": True,
        "target_p2b_total_persistent_bytes": target,
        "full_expand_dimension": full_dimension,
        "byte_matched_exact_dimension": reduced_dimension,
        "byte_matched_exact_total_persistent_bytes": reduced_bytes,
        "byte_matched_exact_underfill_bytes": target - reduced_bytes,
        "byte_matched_exact_underfill_fraction": (target - reduced_bytes) / target,
        "countsketch_dimension": sketch_dimension,
        "countsketch_total_persistent_bytes": sketch_bytes,
        "countsketch_underfill_bytes": target - sketch_bytes,
        "countsketch_underfill_fraction": (target - sketch_bytes) / target,
        "raw_ridge_total_persistent_bytes": _exact_backend_bytes(
            feature_dimension, classes
        ),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _encode_indices(
    encoder: Callable[[torch.Tensor], torch.Tensor],
    features: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(indices), batch_size):
        parts.append(encoder(features[indices[start : start + batch_size]]))
    return torch.cat(parts)


def _select_ridge(
    *,
    name: str,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    features: torch.Tensor,
    labels: torch.Tensor,
    fit_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    candidate_lambdas: list[float],
    num_classes: int,
    batch_size: int,
) -> dict:
    fit_codes = _encode_indices(encoder, features, fit_indices, batch_size)
    validation_codes = _encode_indices(
        encoder, features, validation_indices, batch_size
    )
    fit_targets = torch.nn.functional.one_hot(
        labels[fit_indices].to(device=fit_codes.device, dtype=torch.long),
        num_classes=num_classes,
    ).to(fit_codes.dtype)
    validation_targets = torch.nn.functional.one_hot(
        labels[validation_indices].to(
            device=validation_codes.device, dtype=torch.long
        ),
        num_classes=num_classes,
    ).to(validation_codes.dtype)
    kernel = fit_codes @ fit_codes.T
    kernel = (kernel + kernel.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    eigenvalues.clamp_min_(0)
    coordinates = eigenvectors.T @ fit_targets
    validation_kernel = validation_codes @ fit_codes.T
    scores = []
    for ridge in candidate_lambdas:
        dual = eigenvectors @ (coordinates / (eigenvalues[:, None] + ridge))
        prediction = validation_kernel @ dual
        mse = float(torch.mean((prediction - validation_targets) ** 2).item())
        if not math.isfinite(mse):
            raise RuntimeError(f"non-finite {name} calibration score at {ridge}")
        scores.append({"ridge_lambda": float(ridge), "validation_mse": mse})
    selected = min(
        scores, key=lambda item: (item["validation_mse"], item["ridge_lambda"])
    )
    result = {
        "representation": name,
        "selected_ridge_lambda": selected["ridge_lambda"],
        "scores": scores,
        "fit_samples": len(fit_indices),
        "validation_samples": len(validation_indices),
    }
    del fit_codes, validation_codes, fit_targets, validation_targets
    del kernel, eigenvalues, eigenvectors, coordinates, validation_kernel
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _evaluate(
    *,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    backends: dict[str, object],
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    correct = {name: 0 for name in backends}
    total = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        codes = encoder(features[batch_indices])
        targets = labels[batch_indices].cpu()
        for name, backend in backends.items():
            prediction = backend.predict(codes)
            correct[name] += int((prediction == targets).sum().item())
        total += len(batch_indices)
    return {name: 100.0 * value / total for name, value in correct.items()}


def _run_group(
    *,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    backends: dict[str, object],
    extra_tensors: dict[str, torch.Tensor],
    expected_total_bytes: dict[str, int],
    features: torch.Tensor,
    labels: torch.Tensor,
    training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    encode_batch_size: int,
    evaluation_batch_size: int,
    device: torch.device,
) -> dict:
    records = []
    update_seconds = {name: 0.0 for name in backends}
    encoding_seconds = 0.0
    for task_id, train_indices in enumerate(training_parts):
        _sync(device)
        started = time.perf_counter()
        codes = _encode_indices(
            encoder, features, train_indices, encode_batch_size
        )
        _sync(device)
        encoding_seconds += time.perf_counter() - started
        task_labels = labels[train_indices]
        for name, backend in backends.items():
            _sync(device)
            started = time.perf_counter()
            backend.update(codes, task_labels)
            _sync(device)
            update_seconds[name] += time.perf_counter() - started
        seen_validation = torch.cat(validation_parts[: task_id + 1])
        accuracy = _evaluate(
            encoder=encoder,
            backends=backends,
            features=features,
            labels=labels,
            indices=seen_validation,
            batch_size=evaluation_batch_size,
        )
        state = {}
        for name, backend in backends.items():
            tensors = dict(extra_tensors)
            tensors.update(backend.persistent_tensors())
            actual = persistent_tensor_bytes(tensors)
            if task_id + 1 == len(training_parts) and actual != expected_total_bytes[name]:
                raise AssertionError(
                    f"{name} symbolic/actual state mismatch: "
                    f"{expected_total_bytes[name]} != {actual}"
                )
            state[name] = {
                "total_persistent_bytes": actual,
                "solver_relative_residual": float(
                    backend.diagnostics["solver_relative_residual"]
                ),
            }
        records.append(
            {
                "task": task_id + 1,
                "accuracy_percent": accuracy,
                "state": state,
            }
        )
        print(
            f"TASK {task_id + 1}/{len(training_parts)} "
            + " ".join(f"{name}={accuracy[name]:.4f}" for name in backends),
            flush=True,
        )
        del codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "records": records,
        "encoding_seconds": encoding_seconds,
        "analytic_update_seconds": update_seconds,
    }


def _common_backend(dimension: int, ridge_lambda: float, device: torch.device) -> dict:
    return {
        "dimension": int(dimension),
        "ridge_lambda": float(ridge_lambda),
        "device": device,
        "statistics_dtype": torch.float32,
        "solver_dtype": torch.float32,
    }


def _summarize(groups: list[dict], config: dict, budget_lock: dict) -> dict:
    method_records: dict[str, list[dict]] = {}
    update_seconds: dict[str, float] = {}
    encoding_seconds: dict[str, float] = {}
    for group in groups:
        methods = list(group["records"][0]["accuracy_percent"])
        for name in methods:
            method_records[name] = group["records"]
            update_seconds[name] = group["analytic_update_seconds"][name]
            encoding_seconds[name] = group["encoding_seconds"]
    aia = {
        name: sum(record["accuracy_percent"][name] for record in records)
        / len(records)
        for name, records in method_records.items()
    }
    final_accuracy = {
        name: records[-1]["accuracy_percent"][name]
        for name, records in method_records.items()
    }
    final_state = {
        name: records[-1]["state"][name]["total_persistent_bytes"]
        for name, records in method_records.items()
    }
    maximum_residual = max(
        record["state"][name]["solver_relative_residual"]
        for name, records in method_records.items()
        for record in records
    )
    p2b = "p2b_int8"
    alternatives = (
        "byte_matched_exact",
        "countsketch_exact",
        "raw_feature_ridge",
    )
    lower_state = [name for name in alternatives if final_state[name] <= final_state[p2b]]
    best_advantage = max(aia[name] - aia[p2b] for name in lower_state)
    dominators = []
    for name in lower_state:
        no_worse = (
            aia[name] >= aia[p2b]
            and final_accuracy[name] >= final_accuracy[p2b]
            and update_seconds[name] <= update_seconds[p2b]
        )
        strict = (
            aia[name] > aia[p2b]
            or final_accuracy[name] > final_accuracy[p2b]
            or final_state[name] < final_state[p2b]
            or update_seconds[name] < update_seconds[p2b]
        )
        if no_worse and strict:
            dominators.append(name)
    thresholds = config["gates"]
    underfill_limit = config["budget"]["maximum_underfill_fraction"]
    p2b_loss = aia["full_width_exact"] - aia[p2b]
    gates = {
        "byte_dimensions_locked_before_accuracy": budget_lock[
            "locked_before_representation_encoding_or_accuracy"
        ],
        "byte_matched_exact_budget": budget_lock[
            "byte_matched_exact_underfill_fraction"
        ]
        <= underfill_limit,
        "countsketch_budget": budget_lock["countsketch_underfill_fraction"]
        <= underfill_limit,
        "p2b_accuracy_retention": p2b_loss
        <= thresholds["maximum_p2b_validation_aia_loss_pp"],
        "solver_residual": maximum_residual
        <= thresholds["maximum_solver_relative_residual"],
        "close_to_lower_state_frontier": best_advantage
        <= thresholds["maximum_equal_or_lower_state_aia_advantage_pp"],
        "not_pareto_dominated": not dominators,
    }
    return {
        "validation_aia_percent": aia,
        "final_validation_accuracy_percent": final_accuracy,
        "final_total_persistent_bytes": final_state,
        "analytic_update_seconds": update_seconds,
        "representation_encoding_seconds": encoding_seconds,
        "p2b_validation_aia_loss_pp": p2b_loss,
        "best_equal_or_lower_state_aia_advantage_over_p2b_pp": best_advantage,
        "pareto_dominators_of_p2b": dominators,
        "maximum_solver_relative_residual": maximum_residual,
        "gates": gates,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M5 requires a clean source checkout")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M5 refuses a visible test.pt")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    if int(train["features"].shape[1]) != int(config["ranpac"]["feature_dimension"]):
        raise ValueError("M5 feature dimension mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(
        range(config["num_classes"])
    ):
        raise ValueError("M5 training labels do not match locked classes")

    # This is deliberately the first method-dependent calculation.  It reads
    # only shapes/dtypes and locks both competing dimensions before any
    # representation is encoded or any validation accuracy is available.
    budget_lock = _derive_budget_lock(config)
    print("BUDGET LOCK", json.dumps(budget_lock, sort_keys=True), flush=True)

    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"],
        task_indices,
        config["seed"],
        config["outer_validation_fraction"],
    )
    selection = config["ridge_selection"]
    fit_indices, calibration_validation_indices = m4._calibration_indices(
        train["labels"],
        training_parts,
        seed=config["seed"],
        fit_per_class=int(selection["fit_samples_per_class"]),
        validation_per_class=int(selection["validation_samples_per_class"]),
    )
    device = torch.device(args.device)
    feature_dimension = int(config["ranpac"]["feature_dimension"])
    full_dimension = int(config["ranpac"]["full_expand_dimension"])
    batch_size = int(config["ranpac"]["encode_batch_size"])
    evaluation_batch_size = int(config["ranpac"]["evaluation_batch_size"])

    owner = DenseSquareRootBackend(
        dimension=full_dimension,
        ridge_lambda=1.0,
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        update_backend="blocked_qr",
        update_panel_size=int(config["p2b"]["update_panel_size"]),
    )
    full_frontend = RanPACAnalyticLearner(
        feature_dim=feature_dimension,
        backend=owner,
        seed=int(config["ranpac"]["projection_seed"]),
    )
    projection = full_frontend.projection
    full_encoder = full_frontend.encode
    reduced_dimension = int(budget_lock["byte_matched_exact_dimension"])
    reduced_projection = projection[:, :reduced_dimension].contiguous()
    reduced_encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ reduced_projection
    )
    raw_encoder = lambda values: values.to(device=device, dtype=torch.float32)
    sketch_dimension = int(budget_lock["countsketch_dimension"])
    sketch_owner = DenseSquareRootBackend(
        dimension=sketch_dimension,
        ridge_lambda=1.0,
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        update_backend="blocked_qr",
        update_panel_size=int(config["p2b"]["update_panel_size"]),
    )
    countsketch = CountSketchAnalyticLearner(
        input_dimension=full_dimension,
        backend=sketch_owner,
        seed=int(config["countsketch"]["seed"]),
    )
    sketch_encoder = lambda values: countsketch.encode_codes(full_encoder(values))

    candidates = list(map(float, selection["candidate_lambdas"]))
    selection_results = {}
    for name, encoder in (
        ("full_random_relu", full_encoder),
        ("reduced_width_random_relu", reduced_encoder),
        ("countsketch_random_relu", sketch_encoder),
        ("raw_features", raw_encoder),
    ):
        selection_results[name] = _select_ridge(
            name=name,
            encoder=encoder,
            features=train["features"],
            labels=train["labels"],
            fit_indices=fit_indices,
            validation_indices=calibration_validation_indices,
            candidate_lambdas=candidates,
            num_classes=int(config["num_classes"]),
            batch_size=batch_size,
        )
        print(
            f"RIDGE LOCKED {name}="
            f"{selection_results[name]['selected_ridge_lambda']}",
            flush=True,
        )
    del owner, sketch_owner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    p2b = config["p2b"]
    full_ridge = selection_results["full_random_relu"]["selected_ridge_lambda"]
    full_common = _common_backend(full_dimension, full_ridge, device)
    full_backends = {
        "full_width_exact": ExactGramBackend(**full_common),
        "fp16_square_root": SquareRootBackend(
            storage_mode="float16",
            block_size=int(p2b["block_size"]),
            group_size=int(p2b["group_size"]),
            update_panel_size=int(p2b["update_panel_size"]),
            update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
            first_update_backend=p2b["first_update_backend"],
            quantization_backend=p2b["quantization_backend"],
            quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
            **full_common,
        ),
        "p2b_int8": SquareRootBackend(
            storage_mode="int8",
            block_size=int(p2b["block_size"]),
            group_size=int(p2b["group_size"]),
            update_panel_size=int(p2b["update_panel_size"]),
            update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
            first_update_backend=p2b["first_update_backend"],
            quantization_backend=p2b["quantization_backend"],
            quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
            **full_common,
        ),
    }
    projection_bytes = 4 * feature_dimension * full_dimension
    expected_full = {
        "full_width_exact": projection_bytes
        + _exact_backend_bytes(full_dimension, int(config["num_classes"])),
        "fp16_square_root": projection_bytes
        + _square_root_backend_bytes(
            full_dimension,
            int(config["num_classes"]),
            block_size=int(p2b["block_size"]),
            group_size=int(p2b["group_size"]),
            mode="float16",
        ),
        "p2b_int8": budget_lock["target_p2b_total_persistent_bytes"],
    }
    numerical_failure = None
    groups = []
    try:
        print("START full-width methods", flush=True)
        groups.append(
            _run_group(
                encoder=full_encoder,
                backends=full_backends,
                extra_tensors={"projection": projection},
                expected_total_bytes=expected_full,
                features=train["features"],
                labels=train["labels"],
                training_parts=training_parts,
                validation_parts=validation_parts,
                encode_batch_size=batch_size,
                evaluation_batch_size=evaluation_batch_size,
                device=device,
            )
        )
        del full_backends
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        alternatives = (
            (
                "byte_matched_exact",
                reduced_dimension,
                selection_results["reduced_width_random_relu"][
                    "selected_ridge_lambda"
                ],
                reduced_encoder,
                {"projection": reduced_projection},
                budget_lock["byte_matched_exact_total_persistent_bytes"],
            ),
            (
                "countsketch_exact",
                sketch_dimension,
                selection_results["countsketch_random_relu"][
                    "selected_ridge_lambda"
                ],
                sketch_encoder,
                {
                    "projection": projection,
                    "countsketch.bucket_indices": countsketch.bucket_indices,
                    "countsketch.signs": countsketch.signs,
                },
                budget_lock["countsketch_total_persistent_bytes"],
            ),
            (
                "raw_feature_ridge",
                feature_dimension,
                selection_results["raw_features"]["selected_ridge_lambda"],
                raw_encoder,
                {},
                budget_lock["raw_ridge_total_persistent_bytes"],
            ),
        )
        for name, dimension, ridge, encoder, extras, expected in alternatives:
            print(f"START {name} dimension={dimension}", flush=True)
            backend = ExactGramBackend(
                **_common_backend(dimension, ridge, device)
            )
            groups.append(
                _run_group(
                    encoder=encoder,
                    backends={name: backend},
                    extra_tensors=extras,
                    expected_total_bytes={name: expected},
                    features=train["features"],
                    labels=train["labels"],
                    training_parts=training_parts,
                    validation_parts=validation_parts,
                    encode_batch_size=batch_size,
                    evaluation_batch_size=evaluation_batch_size,
                    device=device,
                )
            )
            del backend
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = _summarize(groups, config, budget_lock)
    except (RuntimeError, AssertionError, torch.linalg.LinAlgError) as error:
        numerical_failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        summary = {"gates": {"numerical_success": False}}

    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "config_sha256": _sha256_file(config_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "generic_backend_sha256": _sha256_file(
            ROOT / "methods/analytic_ridge/backends.py"
        ),
        "ranpac_frontend_sha256": _sha256_file(
            ROOT / "methods/frontends/ranpac.py"
        ),
        "countsketch_frontend_sha256": _sha256_file(
            ROOT / "methods/frontends/countsketch.py"
        ),
        "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
        "projection_sha256": m4._tensor_content_sha256(projection),
        "countsketch_bucket_sha256": hashlib.sha256(
            countsketch.bucket_indices.cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "countsketch_sign_sha256": hashlib.sha256(
            countsketch.signs.cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "class_order": class_order,
        "training_indices_sha256": _sequence_sha256(training_parts),
        "outer_validation_indices_sha256": _sequence_sha256(validation_parts),
        "calibration_fit_indices_sha256": _sequence_sha256([fit_indices]),
        "calibration_validation_indices_sha256": _sequence_sha256(
            [calibration_validation_indices]
        ),
        "upstream_repository": config["ranpac"]["upstream_repository"],
        "upstream_commit": config["ranpac"]["upstream_commit"],
        "upstream_ranpac_py_sha256": config["ranpac"][
            "upstream_ranpac_py_sha256"
        ],
    }
    passed = numerical_failure is None and all(summary["gates"].values())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M5_EQUAL_BUDGET_TRAIN_ONLY"
        if passed
        else "FAIL_M5_EQUAL_BUDGET_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "frontend": "RanPAC Phase-2 random-ReLU analytic head",
            "petl_reproduced": False,
            "comparison": "train-only accuracy-state-update Pareto control",
            "countsketch_is": "fixed signed-hash feature sketch before Exact Ridge",
        },
        "budget_lock": budget_lock,
        "ridge_selection": selection_results,
        "numerical_failure": numerical_failure,
        "provenance": provenance,
        "groups": groups,
        "summary": {key: value for key, value in summary.items() if key != "gates"},
        "gates": summary["gates"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "m5_results.json.tmp"
    destination = output_dir / "m5_results.json"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "budget_lock": budget_lock,
                "summary": payload["summary"],
                "gates": payload["gates"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise RuntimeError("M5 equal-budget train-only gate failed")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
