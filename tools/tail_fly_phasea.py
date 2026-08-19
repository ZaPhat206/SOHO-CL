"""Locked train-only development runner for TAIL-FLY Phase A.

The runner has no held-out evaluation path. Sample-level feature and WTA code
caches are experiment infrastructure and are never written to learner state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.flycl import select_ridge_parameter
from methods.tail_fly import (
    TAILFlyLearner,
    diagonal_tail,
    solve_diagonal_ridge,
    solve_tail_ridge,
    solve_truncated_svd_ridge,
)
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.twa_fly_pilot import (
    _atomic_json,
    _prepare_code_cache,
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _tensor_content_sha256,
)


SCHEMA_VERSION = 1
CONFIG_KEYS = {
    "schema_version",
    "study_id",
    "dataset",
    "model_name",
    "checkpoint_sha256",
    "seed",
    "num_classes",
    "num_tasks",
    "validation_fraction",
    "statistics_dtype",
    "representation",
    "search",
    "fly_control",
    "gates",
}
REPRESENTATION_KEYS = {
    "expand_dim",
    "synaptic_degree",
    "coding_level",
    "encode_batch_size",
    "evaluation_batch_size",
    "svd_update_batch_size",
}
SEARCH_KEYS = {"ranks", "ridge_lambdas", "raw_ridge_lambdas"}
FLY_KEYS = {"ridge_lower", "ridge_upper", "statistics_dtype"}
GATE_KEYS = {
    "maximum_solver_relative_residual",
    "minimum_tail_gain_over_plain_tsvd_pp",
    "maximum_gap_to_exact_fly_pp",
    "minimum_gain_over_raw_ridge_pp",
    "maximum_state_fraction_of_exact_fly",
}


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != CONFIG_KEYS:
        raise ValueError(
            f"config keys must be exactly {sorted(CONFIG_KEYS)}"
        )
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported TAIL-FLY config schema")
    if config["seed"] != 2025:
        raise ValueError("TAIL-FLY Phase A uses repository-default seed 2025")
    if (
        config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
    ):
        raise ValueError("invalid class/task count")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if config["statistics_dtype"] not in {"float32", "float64"}:
        raise ValueError("statistics_dtype must be float32 or float64")
    if set(config["representation"]) != REPRESENTATION_KEYS:
        raise ValueError("representation keys mismatch")
    representation = config["representation"]
    if min(
        representation["expand_dim"],
        representation["synaptic_degree"],
        representation["encode_batch_size"],
        representation["evaluation_batch_size"],
        representation["svd_update_batch_size"],
    ) <= 0:
        raise ValueError("representation dimensions and batch sizes must be positive")
    if not 0 < representation["coding_level"] <= 1:
        raise ValueError("coding_level must be in (0, 1]")
    if set(config["search"]) != SEARCH_KEYS:
        raise ValueError("search keys mismatch")
    search = config["search"]
    if any(not search[key] for key in SEARCH_KEYS):
        raise ValueError("search lists must be non-empty")
    if any(
        int(rank) <= 0 or int(rank) > representation["expand_dim"]
        for rank in search["ranks"]
    ):
        raise ValueError("invalid rank candidate")
    if len(set(map(int, search["ranks"]))) != len(search["ranks"]):
        raise ValueError("rank candidates must be unique")
    for key in ("ridge_lambdas", "raw_ridge_lambdas"):
        if any(float(value) <= 0 for value in search[key]):
            raise ValueError("Ridge candidates must be positive")
        if len(set(map(float, search[key]))) != len(search[key]):
            raise ValueError("Ridge candidates must be unique")
    if set(config["fly_control"]) != FLY_KEYS:
        raise ValueError("fly_control keys mismatch")
    fly = config["fly_control"]
    if fly["ridge_lower"] >= fly["ridge_upper"]:
        raise ValueError("invalid exact-FLY GCV range")
    if fly["statistics_dtype"] not in {"float32", "float64"}:
        raise ValueError("invalid exact-FLY dtype")
    if set(config["gates"]) != GATE_KEYS:
        raise ValueError("gate keys mismatch")
    return config


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _cache_config(config: dict) -> dict:
    """Minimal compatibility view for the shared verified WTA cache utility."""
    representation = config["representation"]
    return {
        "seed": config["seed"],
        "num_classes": config["num_classes"],
        "representation": {
            "expand_dim": representation["expand_dim"],
            "synaptic_degree": representation["synaptic_degree"],
            "coding_level": representation["coding_level"],
            "encode_batch_size": representation["encode_batch_size"],
        },
        "statistics_dtype": "float32",
        "raw_ridge_lambda": 1.0,
        "solver_tolerance": 1e-5,
        "solver_max_iterations": 100,
    }


def _git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def _dense_codes(
    indices: torch.Tensor,
    values: torch.Tensor,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    result = torch.zeros(
        (len(indices), dimension), device=device, dtype=dtype
    )
    result.scatter_(
        1,
        indices.to(device=device, dtype=torch.long),
        values.to(device=device, dtype=dtype),
    )
    return result


def _expand_cross(
    cross: torch.Tensor,
    counts: torch.Tensor,
    old_ids: list[int],
    new_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if old_ids == new_ids:
        return cross, counts
    updated = cross.new_zeros((cross.shape[0], len(new_ids)))
    updated_counts = counts.new_zeros(len(new_ids))
    columns = {class_id: index for index, class_id in enumerate(new_ids)}
    for old_column, class_id in enumerate(old_ids):
        new_column = columns[class_id]
        updated[:, new_column] = cross[:, old_column]
        updated_counts[new_column] = counts[old_column]
    return updated, updated_counts


def _targets(
    labels: torch.Tensor,
    class_ids: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    columns = {class_id: index for index, class_id in enumerate(class_ids)}
    encoded = torch.tensor(
        [columns[int(value)] for value in labels.detach().cpu().tolist()],
        device=device,
        dtype=torch.long,
    )
    return torch.nn.functional.one_hot(
        encoded, num_classes=len(class_ids)
    ).to(dtype)


def _code_accuracy(
    weights: torch.Tensor,
    class_ids: list[int],
    sample_indices: torch.Tensor,
    code_indices: torch.Tensor,
    code_values: torch.Tensor,
    labels: torch.Tensor,
    dimension: int,
    batch_size: int,
) -> float:
    correct = 0
    for start in range(0, len(sample_indices), batch_size):
        selected = sample_indices[start : start + batch_size]
        codes = _dense_codes(
            code_indices[selected],
            code_values[selected],
            dimension,
            device=weights.device,
            dtype=weights.dtype,
        )
        columns = (codes @ weights).argmax(1).detach().cpu().tolist()
        predictions = torch.tensor([class_ids[column] for column in columns])
        correct += int((predictions == labels[selected].cpu()).sum().item())
    return 100.0 * correct / max(len(sample_indices), 1)


def _raw_accuracy(
    weights: torch.Tensor,
    class_ids: list[int],
    sample_indices: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> float:
    correct = 0
    for start in range(0, len(sample_indices), batch_size):
        selected = sample_indices[start : start + batch_size]
        logits = features[selected].to(weights.device, weights.dtype) @ weights
        columns = logits.argmax(1).detach().cpu().tolist()
        predictions = torch.tensor([class_ids[column] for column in columns])
        correct += int((predictions == labels[selected].cpu()).sum().item())
    return 100.0 * correct / max(len(sample_indices), 1)


def _stage_code_accuracy(
    weights,
    class_ids,
    validation_parts,
    task,
    code_indices,
    code_values,
    labels,
    dimension,
    batch_size,
) -> float:
    values = [
        _code_accuracy(
            weights,
            class_ids,
            validation_parts[previous],
            code_indices,
            code_values,
            labels,
            dimension,
            batch_size,
        )
        for previous in range(task + 1)
    ]
    return sum(values) / len(values)


def _state_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return sum(
            part.numel() * part.element_size()
            for part in (
                tensor.ccol_indices(),
                tensor.row_indices(),
                tensor.values(),
            )
        )
    raise ValueError(f"unsupported state tensor layout {tensor.layout}")


def _solve_spd(gram: torch.Tensor, cross: torch.Tensor, ridge: float) -> torch.Tensor:
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    system = (gram + gram.T) * 0.5 + ridge * identity
    factor = torch.linalg.cholesky(system)
    return torch.cholesky_solve(cross, factor)


def _unit_path(output_dir: Path, name: str) -> Path:
    return output_dir / "units" / f"{name}.json"


def _load_unit(path: Path, context_sha256: str) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("context_sha256") != context_sha256:
        raise RuntimeError(f"stale unit artifact: {path}")
    print(f"RESUME {path.stem}", flush=True)
    return payload


def _save_unit(path: Path, context_sha256: str, payload: dict) -> dict:
    result = {"context_sha256": context_sha256, **payload}
    _atomic_json(path, result)
    return result


def _evaluate_rank(
    *,
    rank: int,
    config: dict,
    train: dict,
    code_indices: torch.Tensor,
    code_values: torch.Tensor,
    projection: torch.Tensor,
    training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    representation = config["representation"]
    dtype = _dtype(config["statistics_dtype"])
    ridges = list(map(float, config["search"]["ridge_lambdas"]))
    learner = TAILFlyLearner(
        feature_dim=int(train["features"].shape[1]),
        expand_dim=int(representation["expand_dim"]),
        synaptic_degree=int(representation["synaptic_degree"]),
        coding_level=float(representation["coding_level"]),
        max_rank=int(rank),
        ridge_lambda=ridges[0],
        seed=int(config["seed"]),
        device=device,
        dtype=dtype,
        projection=projection,
    )
    scores = {
        method: {str(ridge): [] for ridge in ridges}
        for method in ("tail_fly", "plain_tsvd_fly", "diagonal_only_fly")
    }
    task_diagnostics = []
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        task_started = time.perf_counter()
        batch_size = int(representation["svd_update_batch_size"])
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            codes = _dense_codes(
                code_indices[selected],
                code_values[selected],
                learner.expand_dim,
                device=device,
                dtype=dtype,
            )
            learner.accumulate_codes(codes, train["labels"][selected])
            del codes
        learner.finalize_update()
        tail = diagonal_tail(
            learner.exact_diagonal, learner.svd.U, learner.svd.s
        )
        maximum_residual = 0.0
        for ridge in ridges:
            solutions = {
                "tail_fly": solve_tail_ridge(
                    learner.svd.U, learner.svd.s, tail, learner.Q, ridge
                ),
                "plain_tsvd_fly": solve_truncated_svd_ridge(
                    learner.svd.U, learner.svd.s, learner.Q, ridge
                ),
                "diagonal_only_fly": solve_diagonal_ridge(
                    learner.exact_diagonal, learner.Q, ridge
                ),
            }
            maximum_residual = max(
                maximum_residual,
                solutions["tail_fly"].relative_residual,
                solutions["plain_tsvd_fly"].relative_residual,
                solutions["diagonal_only_fly"].relative_residual,
            )
            for method, solution in solutions.items():
                accuracy = _stage_code_accuracy(
                    solution.weights,
                    learner.class_ids,
                    validation_parts,
                    task,
                    code_indices,
                    code_values,
                    train["labels"],
                    learner.expand_dim,
                    int(representation["evaluation_batch_size"]),
                )
                scores[method][str(ridge)].append(accuracy)
        task_diagnostics.append(
            {
                "task": task + 1,
                "seen_classes": len(learner.class_ids),
                "effective_rank": learner.svd.effective_rank,
                "diagonal_tail_sum": float(tail.sum().item()),
                "diagonal_tail_fraction": float(
                    tail.sum().item()
                    / max(float(learner.exact_diagonal.sum().item()), 1e-30)
                ),
                "maximum_solver_relative_residual": maximum_residual,
                "resident_state_bytes": learner.persistent_state_bytes(),
                "aggregate_checkpoint_bytes": learner.persistent_state_bytes(
                    include_classifier=False
                ),
                "seconds": time.perf_counter() - task_started,
            }
        )
        print(
            f"TASK rank={rank} {task+1}/{len(training_parts)} "
            f"effective_rank={learner.svd.effective_rank} "
            f"tail={task_diagnostics[-1]['diagonal_tail_fraction']:.4f} "
            f"residual={maximum_residual:.3e} "
            f"elapsed={(time.perf_counter()-started)/60:.1f}m",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidates = []
    projection_bytes = _state_bytes(learner.flyhash.projection_matrix)
    for method, by_ridge in scores.items():
        for ridge, stage_accuracy in by_ridge.items():
            if method == "tail_fly":
                state_bytes = learner.persistent_state_bytes()
            elif method == "plain_tsvd_fly":
                tensors = (
                    learner.flyhash.projection_matrix,
                    learner.svd.U,
                    learner.svd.s,
                    learner.Q,
                    learner.counts,
                    learner.weights,
                )
                state_bytes = sum(_state_bytes(value) for value in tensors)
            else:
                state_bytes = (
                    projection_bytes
                    + _state_bytes(learner.exact_diagonal)
                    + _state_bytes(learner.Q)
                    + _state_bytes(learner.counts)
                    + _state_bytes(learner.weights)
                )
            candidates.append(
                {
                    "method": method,
                    "rank": rank if method != "diagonal_only_fly" else 0,
                    "ridge_lambda": float(ridge),
                    "validation_average_accuracy": sum(stage_accuracy)
                    / len(stage_accuracy),
                    "stage_accuracy": stage_accuracy,
                    "persistent_state_bytes": state_bytes,
                    "uses_test_set": False,
                    "exemplar_free": True,
                }
            )
    return {
        "rank": rank,
        "candidates": candidates,
        "task_diagnostics": task_diagnostics,
        "seconds": time.perf_counter() - started,
    }


def _evaluate_raw(
    config,
    train,
    training_parts,
    validation_parts,
    device,
) -> dict:
    dtype = _dtype(config["statistics_dtype"])
    dimension = int(train["features"].shape[1])
    ridges = list(map(float, config["search"]["raw_ridge_lambdas"]))
    gram = torch.zeros((dimension, dimension), device=device, dtype=dtype)
    cross = torch.zeros((dimension, 0), device=device, dtype=dtype)
    counts = torch.zeros(0, device=device, dtype=dtype)
    class_ids: list[int] = []
    scores = {str(ridge): [] for ridge in ridges}
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        values = train["features"][indices].to(device=device, dtype=dtype)
        labels = train["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        cross, counts = _expand_cross(cross, counts, class_ids, new_ids)
        class_ids = new_ids
        targets = _targets(labels, class_ids, device=device, dtype=dtype)
        gram += values.T @ values
        cross += values.T @ targets
        counts += targets.sum(0)
        for ridge in ridges:
            weights = _solve_spd(gram, cross, ridge)
            accuracy = sum(
                _raw_accuracy(
                    weights,
                    class_ids,
                    validation_parts[previous],
                    train["features"],
                    train["labels"],
                    int(config["representation"]["evaluation_batch_size"]),
                )
                for previous in range(task + 1)
            ) / (task + 1)
            scores[str(ridge)].append(accuracy)
        print(
            f"TASK raw {task+1}/{len(training_parts)} "
            f"elapsed={(time.perf_counter()-started)/60:.1f}m",
            flush=True,
        )
    candidates = []
    for ridge, stage_accuracy in scores.items():
        weights = _solve_spd(gram, cross, float(ridge))
        candidates.append(
            {
                "method": "raw_ridge",
                "rank": 0,
                "ridge_lambda": float(ridge),
                "validation_average_accuracy": sum(stage_accuracy)
                / len(stage_accuracy),
                "stage_accuracy": stage_accuracy,
                "persistent_state_bytes": sum(
                    _state_bytes(value)
                    for value in (gram, cross, counts, weights)
                ),
                "uses_test_set": False,
                "exemplar_free": True,
            }
        )
    return {"candidates": candidates, "seconds": time.perf_counter() - started}


def _evaluate_exact_fly(
    config,
    train,
    code_indices,
    code_values,
    projection,
    training_parts,
    validation_parts,
    device,
) -> dict:
    representation = config["representation"]
    dtype = _dtype(config["fly_control"]["statistics_dtype"])
    dimension = int(representation["expand_dim"])
    gram = torch.zeros((dimension, dimension), device=device, dtype=dtype)
    cross = torch.zeros((dimension, 0), device=device, dtype=dtype)
    counts = torch.zeros(0, device=device, dtype=dtype)
    class_ids: list[int] = []
    stage_accuracy, selected_ridges = [], []
    weights = None
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        codes = _dense_codes(
            code_indices[indices],
            code_values[indices],
            dimension,
            device=device,
            dtype=dtype,
        )
        labels = train["labels"][indices]
        new_ids = sorted(set(class_ids) | set(map(int, labels.tolist())))
        cross, counts = _expand_cross(cross, counts, class_ids, new_ids)
        class_ids = new_ids
        targets = _targets(labels, class_ids, device=device, dtype=dtype)
        gram += codes.T @ codes
        cross += codes.T @ targets
        counts += targets.sum(0)
        ridge = float(
            select_ridge_parameter(
                codes,
                targets,
                config["fly_control"]["ridge_lower"],
                config["fly_control"]["ridge_upper"],
            ).item()
        )
        selected_ridges.append(ridge)
        weights = _solve_spd(gram, cross, ridge)
        accuracy = _stage_code_accuracy(
            weights,
            class_ids,
            validation_parts,
            task,
            code_indices,
            code_values,
            train["labels"],
            dimension,
            int(representation["evaluation_batch_size"]),
        )
        stage_accuracy.append(accuracy)
        print(
            f"TASK matched_exact_fly {task+1}/{len(training_parts)} ridge={ridge:g} "
            f"AA={accuracy:.4f} elapsed={(time.perf_counter()-started)/60:.1f}m",
            flush=True,
        )
        del codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    state_bytes = sum(
        _state_bytes(value)
        for value in (projection, gram, cross, counts, weights)
    )
    return {
        "method": "matched_exact_fly",
        "rank": dimension,
        "ridge_lambda": selected_ridges[-1],
        "ridge_schedule": selected_ridges,
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy,
        "persistent_state_bytes": state_bytes,
        "seconds": time.perf_counter() - started,
        "uses_test_set": False,
        "exemplar_free": True,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    code_cache_dir = Path(args.code_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    test_path = feature_cache_dir / "test.pt"
    if args.require_test_hidden and test_path.exists():
        raise RuntimeError(f"held-out file is visible: {test_path}")

    cache_args = argparse.Namespace(
        dataset=config["dataset"], model_name=config["model_name"]
    )
    train, _, metadata = validate_cache(
        feature_cache_dir, cache_args, load_test=False
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    labels = sorted(map(int, torch.unique(train["labels"]).tolist()))
    if labels != list(range(config["num_classes"])):
        raise ValueError("training labels do not match locked global classes")

    train_path = feature_cache_dir / "train.pt"
    train_sha256 = _sha256_file(train_path)
    code_indices, code_values, code_metadata, projection = _prepare_code_cache(
        train=train,
        train_sha256=train_sha256,
        cache_dir=code_cache_dir,
        config=_cache_config(config),
        device=args.device,
    )
    projection_sha256 = _tensor_content_sha256(projection)
    if projection_sha256 != code_metadata["projection"]["sha256"]:
        raise RuntimeError("runtime projection does not match WTA cache")

    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"],
        task_indices,
        config["seed"],
        config["validation_fraction"],
    )
    context = {
        "config_sha256": _sha256_file(config_path),
        "train_sha256": train_sha256,
        "code_cache_identity_sha256": code_metadata["identity_sha256"],
        "projection_sha256": projection_sha256,
        "training_indices_sha256": _sequence_sha256(training_parts),
        "validation_indices_sha256": _sequence_sha256(validation_parts),
    }
    context_sha256 = _sha256_bytes(
        json.dumps(context, sort_keys=True).encode("utf-8")
    )
    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_path = _unit_path(output_dir, "matched_exact_fly")
    exact_unit = _load_unit(exact_path, context_sha256)
    if exact_unit is None:
        print("START matched_exact_fly", flush=True)
        exact = _evaluate_exact_fly(
            config,
            train,
            code_indices,
            code_values,
            projection,
            training_parts,
            validation_parts,
            device,
        )
        exact_unit = _save_unit(exact_path, context_sha256, exact)
        print("DONE matched_exact_fly", flush=True)

    raw_path = _unit_path(output_dir, "raw_ridge")
    raw_unit = _load_unit(raw_path, context_sha256)
    if raw_unit is None:
        print("START raw_ridge", flush=True)
        raw_unit = _save_unit(
            raw_path,
            context_sha256,
            _evaluate_raw(
                config, train, training_parts, validation_parts, device
            ),
        )
        print("DONE raw_ridge", flush=True)

    rank_units = []
    for rank in map(int, config["search"]["ranks"]):
        path = _unit_path(output_dir, f"rank_{rank}")
        unit = _load_unit(path, context_sha256)
        if unit is None:
            print(f"START rank={rank}", flush=True)
            unit = _save_unit(
                path,
                context_sha256,
                _evaluate_rank(
                    rank=rank,
                    config=config,
                    train=train,
                    code_indices=code_indices,
                    code_values=code_values,
                    projection=projection,
                    training_parts=training_parts,
                    validation_parts=validation_parts,
                    device=device,
                ),
            )
            print(f"DONE rank={rank}", flush=True)
        rank_units.append(unit)

    rank_candidates = []
    for unit_index, unit in enumerate(rank_units):
        rank_candidates.extend(
            candidate
            for candidate in unit["candidates"]
            if candidate["method"] != "diagonal_only_fly" or unit_index == 0
        )
    candidates = rank_candidates + raw_unit["candidates"] + [
        {
            key: value
            for key, value in exact_unit.items()
            if key != "context_sha256"
        }
    ]
    tail_candidates = [row for row in candidates if row["method"] == "tail_fly"]
    selected = max(
        tail_candidates,
        key=lambda row: (
            row["validation_average_accuracy"],
            -row["rank"],
            -row["ridge_lambda"],
        ),
    )
    same_config_plain = next(
        row
        for row in candidates
        if row["method"] == "plain_tsvd_fly"
        and row["rank"] == selected["rank"]
        and row["ridge_lambda"] == selected["ridge_lambda"]
    )
    raw = max(
        (row for row in candidates if row["method"] == "raw_ridge"),
        key=lambda row: row["validation_average_accuracy"],
    )
    exact = next(
        row for row in candidates if row["method"] == "matched_exact_fly"
    )
    maximum_residual = max(
        diagnostic["maximum_solver_relative_residual"]
        for unit in rank_units
        for diagnostic in unit["task_diagnostics"]
    )
    gates_config = config["gates"]
    diagnostics = {
        "selected_tail_gain_over_plain_tsvd_pp": selected[
            "validation_average_accuracy"
        ]
        - same_config_plain["validation_average_accuracy"],
        "selected_tail_gap_to_exact_fly_pp": exact[
            "validation_average_accuracy"
        ]
        - selected["validation_average_accuracy"],
        "selected_tail_gain_over_raw_ridge_pp": selected[
            "validation_average_accuracy"
        ]
        - raw["validation_average_accuracy"],
        "selected_tail_state_fraction_of_exact_fly": selected[
            "persistent_state_bytes"
        ]
        / exact["persistent_state_bytes"],
        "maximum_solver_relative_residual": maximum_residual,
    }
    gates = {
        "numerical_stability": maximum_residual
        <= gates_config["maximum_solver_relative_residual"],
        "tail_beats_plain_tsvd": diagnostics[
            "selected_tail_gain_over_plain_tsvd_pp"
        ]
        >= gates_config["minimum_tail_gain_over_plain_tsvd_pp"],
        "within_exact_fly_tolerance": diagnostics[
            "selected_tail_gap_to_exact_fly_pp"
        ]
        <= gates_config["maximum_gap_to_exact_fly_pp"],
        "not_worse_than_raw_ridge": diagnostics[
            "selected_tail_gain_over_raw_ridge_pp"
        ]
        >= gates_config["minimum_gain_over_raw_ridge_pp"],
        "state_budget": diagnostics[
            "selected_tail_state_fraction_of_exact_fly"
        ]
        <= gates_config["maximum_state_fraction_of_exact_fly"],
        "heldout_test_remained_hidden": not test_path.exists(),
    }
    decision = "REVIEW_CONFIRMATORY_PROTOCOL" if all(gates.values()) else "STOP_TAIL_FLY"
    provenance = {
        **_git_provenance(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "config_path": str(config_path),
        **context,
        "context_sha256": context_sha256,
        "feature_cache_metadata": metadata,
        "code_cache_dir": str(code_cache_dir),
        "code_cache_role": "sample_level_experiment_infrastructure_not_learner_state",
        "projection_sha256": projection_sha256,
        "class_order": class_order,
        "class_order_sha256": _sha256_bytes(
            ",".join(map(str, class_order)).encode("ascii")
        ),
        "heldout_test_path_visible": test_path.exists(),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config["study_id"],
        "status": "train_only_development_complete",
        "decision": decision,
        "uses_test_set": False,
        "held_out_test_authorized": False,
        "selected_tail_config": {
            "rank": selected["rank"],
            "ridge_lambda": selected["ridge_lambda"],
        },
        "selected": selected,
        "matched_plain_tsvd": same_config_plain,
        "selected_raw_ridge": raw,
        "exact_fly": exact,
        "candidates": candidates,
        "gates": gates,
        "gate_diagnostics": diagnostics,
        "provenance": provenance,
    }
    _atomic_json(output_dir / "phasea_results.json", payload)
    _atomic_json(
        output_dir / "gate_results.json",
        {
            "decision": decision,
            "gates": gates,
            "diagnostics": diagnostics,
            "held_out_test_authorized": False,
        },
    )
    print(json.dumps({"decision": decision, "gates": gates, "diagnostics": diagnostics}, indent=2), flush=True)
    print("TAIL-FLY PHASE A COMPLETE. Held-out evaluation remains unauthorized.", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--code-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
