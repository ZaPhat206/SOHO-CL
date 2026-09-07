"""Train-only regression between legacy and generic FLY analytic paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import ExactGramBackend, SquareRootBackend
from methods.frontends import FLYAnalyticLearner
from methods.srq_fly_optimized import SquareRootFLYLearner
from tools import srq_fly_d0 as d0
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.tail_fly_phasea import _expand_cross, _state_bytes, _targets
from tools.twa_fly_pilot import (
    _prepare_code_cache,
    _sequence_sha256,
    _sha256_file,
    _tensor_content_sha256,
)


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
    "validation_fraction",
    "statistics_dtype",
    "solver_dtype",
    "ridge_lambda",
    "representation",
    "p2b",
    "gates",
}
REPRESENTATION_KEYS = {
    "expand_dim",
    "synaptic_degree",
    "coding_level",
    "encode_batch_size",
    "evaluation_batch_size",
}
P2B_KEYS = {
    "storage_mode",
    "block_size",
    "group_size",
    "update_backend",
    "update_panel_size",
    "update_trailing_chunk_size",
    "first_update_backend",
    "quantization_backend",
    "quantization_batch_blocks",
}
GATE_KEYS = {
    "require_tensor_identity",
    "require_persistent_byte_identity",
    "minimum_prediction_agreement",
    "maximum_relative_logit_error",
    "maximum_solver_relative_residual",
}


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M3 config keys/schema mismatch")
    if config["uses_test_set"] is not False:
        raise ValueError("M3 must remain train-only")
    if config["accuracy_based_selection"] is not False:
        raise ValueError("M3 is a regression audit, not model selection")
    if config["seed"] != 2025:
        raise ValueError("new protocols require seed 2025")
    if (
        config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or not 0 < config["validation_fraction"] < 1
    ):
        raise ValueError("invalid M3 class/task split")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("locked FLY regression uses float32")
    if config["ridge_lambda"] <= 0:
        raise ValueError("Ridge coefficient must be positive")
    if set(config["representation"]) != REPRESENTATION_KEYS:
        raise ValueError("M3 representation fields mismatch")
    representation = config["representation"]
    if (
        min(
            representation[name]
            for name in (
                "expand_dim",
                "synaptic_degree",
                "encode_batch_size",
                "evaluation_batch_size",
            )
        )
        <= 0
        or not 0 < representation["coding_level"] <= 1
    ):
        raise ValueError("invalid M3 representation")
    if set(config["p2b"]) != P2B_KEYS:
        raise ValueError("M3 P2B fields mismatch")
    p2b = config["p2b"]
    if (
        p2b["storage_mode"] != "int8"
        or p2b["update_backend"] != "blocked_qr"
        or p2b["first_update_backend"] != "gram_cholesky"
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
        raise ValueError("M3 no longer matches the frozen P2B method")
    if set(config["gates"]) != GATE_KEYS:
        raise ValueError("M3 gate fields mismatch")
    gates = config["gates"]
    if gates["require_tensor_identity"] is not True:
        raise ValueError("M3 requires tensor identity")
    if gates["require_persistent_byte_identity"] is not True:
        raise ValueError("M3 requires persistent-byte identity")
    if not 0 <= gates["minimum_prediction_agreement"] <= 1:
        raise ValueError("invalid prediction agreement gate")
    if gates["maximum_relative_logit_error"] < 0:
        raise ValueError("invalid logit gate")
    if gates["maximum_solver_relative_residual"] <= 0:
        raise ValueError("invalid solver gate")
    return config


def _cache_config(config: dict) -> dict:
    return {
        "seed": config["seed"],
        "num_classes": config["num_classes"],
        "representation": dict(config["representation"]),
        "statistics_dtype": config["statistics_dtype"],
        "raw_ridge_lambda": 1.0,
        "solver_tolerance": config["gates"]["maximum_solver_relative_residual"],
        "solver_max_iterations": 100,
    }


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(expected).item()), 1.0)
    return float(torch.linalg.vector_norm(actual - expected).item()) / denominator


def _tensor_identity(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)


def _compressed_identity(left, right) -> bool:
    if (
        left.dimension != right.dimension
        or left.block_size != right.block_size
        or left.group_size != right.group_size
        or left.mode != right.mode
        or not torch.equal(left.diagonal, right.diagonal)
        or len(left.blocks) != len(right.blocks)
    ):
        return False
    return all(
        left_block.row_block == right_block.row_block
        and left_block.col_block == right_block.col_block
        and torch.equal(left_block.values, right_block.values)
        and (
            left.mode != "int8"
            or torch.equal(left_block.scales, right_block.scales)
        )
        for left_block, right_block in zip(left.blocks, right.blocks)
    )


def _new_learners(config: dict, feature_dim: int, projection, device):
    representation = config["representation"]
    p2b = config["p2b"]
    common_backend = {
        "dimension": int(representation["expand_dim"]),
        "ridge_lambda": float(config["ridge_lambda"]),
        "device": device,
        "statistics_dtype": torch.float32,
        "solver_dtype": torch.float32,
    }
    exact = FLYAnalyticLearner(
        feature_dim=feature_dim,
        synaptic_degree=int(representation["synaptic_degree"]),
        coding_level=float(representation["coding_level"]),
        backend=ExactGramBackend(**common_backend),
        seed=int(config["seed"]),
        projection=projection,
    )
    legacy_kwargs = {
        "feature_dim": feature_dim,
        "expand_dim": int(representation["expand_dim"]),
        "synaptic_degree": int(representation["synaptic_degree"]),
        "coding_level": float(representation["coding_level"]),
        "ridge_lambda": float(config["ridge_lambda"]),
        "block_size": int(p2b["block_size"]),
        "group_size": int(p2b["group_size"]),
        "seed": int(config["seed"]),
        "device": device,
        "statistics_dtype": torch.float32,
        "solver_dtype": torch.float32,
        "projection": projection,
        "storage_mode": p2b["storage_mode"],
        "update_backend": p2b["update_backend"],
        "update_panel_size": int(p2b["update_panel_size"]),
        "update_trailing_chunk_size": p2b["update_trailing_chunk_size"],
        "first_update_backend": p2b["first_update_backend"],
        "quantization_backend": p2b["quantization_backend"],
        "quantization_batch_blocks": int(p2b["quantization_batch_blocks"]),
    }
    legacy = SquareRootFLYLearner(**legacy_kwargs)
    generic = FLYAnalyticLearner(
        feature_dim=feature_dim,
        synaptic_degree=int(representation["synaptic_degree"]),
        coding_level=float(representation["coding_level"]),
        backend=SquareRootBackend(
            storage_mode=p2b["storage_mode"],
            block_size=int(p2b["block_size"]),
            group_size=int(p2b["group_size"]),
            update_panel_size=int(p2b["update_panel_size"]),
            update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
            first_update_backend=p2b["first_update_backend"],
            quantization_backend=p2b["quantization_backend"],
            quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
            **common_backend,
        ),
        seed=int(config["seed"]),
        projection=projection,
    )
    return exact, legacy, generic


def _legacy_exact_update(state: dict, codes: torch.Tensor, labels: torch.Tensor, ridge: float):
    class_ids = sorted(set(state["class_ids"]) | set(map(int, labels.cpu().tolist())))
    cross, counts = _expand_cross(
        state["cross"], state["counts"], state["class_ids"], class_ids
    )
    targets = _targets(labels, class_ids, device=codes.device, dtype=codes.dtype)
    state["gram"].add_(codes.T @ codes)
    cross.add_(codes.T @ targets)
    counts.add_(targets.sum(0))
    system = state["gram"].clone()
    system.diagonal().add_(ridge)
    symmetric = (system + system.T) * 0.5
    factor, info = torch.linalg.cholesky_ex(symmetric)
    if int(info.max().item()) != 0:
        raise RuntimeError("legacy Exact FLY regression solve failed")
    weights = torch.cholesky_solve(cross, factor)
    residual = _relative_error(symmetric @ weights, cross)
    state.update(class_ids=class_ids, cross=cross, counts=counts, weights=weights)
    return residual


def _prediction_comparison(
    *,
    code_indices: torch.Tensor,
    code_values: torch.Tensor,
    indices: torch.Tensor,
    dimension: int,
    batch_size: int,
    left_weights: torch.Tensor,
    right_weights: torch.Tensor,
    device,
) -> dict:
    agreements = 0
    total = 0
    squared_difference = 0.0
    squared_reference = 0.0
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        codes = d0._dense_codes(
            code_indices[batch],
            code_values[batch],
            dimension,
            device=device,
            dtype=torch.float32,
        )
        left = codes @ left_weights
        right = codes @ right_weights
        agreements += int((left.argmax(1) == right.argmax(1)).sum().item())
        total += len(codes)
        squared_difference += float(torch.sum((left - right) ** 2).item())
        squared_reference += float(torch.sum(left**2).item())
    return {
        "prediction_agreement": agreements / total,
        "relative_logit_error": (squared_difference**0.5)
        / max(squared_reference**0.5, 1.0),
    }


def _run_tasks(
    *,
    config: dict,
    train: dict,
    code_indices: torch.Tensor,
    code_values: torch.Tensor,
    projection: torch.Tensor,
    training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    device,
) -> dict:
    dimension = int(config["representation"]["expand_dim"])
    exact, legacy_p2b, generic_p2b = _new_learners(
        config, int(train["features"].shape[1]), projection, device
    )
    legacy_exact = {
        "gram": torch.zeros((dimension, dimension), device=device),
        "cross": torch.zeros((dimension, 0), device=device),
        "counts": torch.zeros(0, device=device),
        "class_ids": [],
        "weights": None,
    }
    records = []
    for task_id, indices in enumerate(training_parts):
        codes = d0._dense_codes(
            code_indices[indices],
            code_values[indices],
            dimension,
            device=device,
            dtype=torch.float32,
        )
        labels = train["labels"][indices]
        legacy_exact_residual = _legacy_exact_update(
            legacy_exact, codes, labels, float(config["ridge_lambda"])
        )
        exact.update_codes(codes, labels)
        legacy_p2b.update_codes(codes, labels)
        generic_p2b.update_codes(codes, labels)
        seen_validation = torch.cat(validation_parts[: task_id + 1])
        exact_prediction = _prediction_comparison(
            code_indices=code_indices,
            code_values=code_values,
            indices=seen_validation,
            dimension=dimension,
            batch_size=int(config["representation"]["evaluation_batch_size"]),
            left_weights=legacy_exact["weights"],
            right_weights=exact.weights,
            device=device,
        )
        p2b_prediction = _prediction_comparison(
            code_indices=code_indices,
            code_values=code_values,
            indices=seen_validation,
            dimension=dimension,
            batch_size=int(config["representation"]["evaluation_batch_size"]),
            left_weights=legacy_p2b.weights,
            right_weights=generic_p2b.weights,
            device=device,
        )
        legacy_exact_bytes = sum(
            _state_bytes(value)
            for value in (
                projection,
                legacy_exact["gram"],
                legacy_exact["cross"],
                legacy_exact["counts"],
                legacy_exact["weights"],
            )
        )
        record = {
            "task": task_id + 1,
            "exact": {
                "gram_identical": _tensor_identity(
                    legacy_exact["gram"], exact.backend.gram
                ),
                "cross_identical": _tensor_identity(
                    legacy_exact["cross"], exact.Q
                ),
                "counts_identical": _tensor_identity(
                    legacy_exact["counts"], exact.counts
                ),
                "weights_identical": _tensor_identity(
                    legacy_exact["weights"], exact.weights
                ),
                "class_ids_identical": legacy_exact["class_ids"] == exact.class_ids,
                "legacy_persistent_state_bytes": legacy_exact_bytes,
                "generic_persistent_state_bytes": exact.persistent_state_bytes(),
                "legacy_solver_relative_residual": legacy_exact_residual,
                "generic_solver_relative_residual": exact.backend.diagnostics[
                    "solver_relative_residual"
                ],
                **exact_prediction,
            },
            "p2b": {
                "factor_identical": _compressed_identity(
                    legacy_p2b.factor, generic_p2b.backend.factor
                ),
                "cross_identical": _tensor_identity(legacy_p2b.Q, generic_p2b.Q),
                "counts_identical": _tensor_identity(
                    legacy_p2b.counts, generic_p2b.counts
                ),
                "weights_identical": _tensor_identity(
                    legacy_p2b.weights, generic_p2b.weights
                ),
                "class_ids_identical": legacy_p2b.class_ids == generic_p2b.class_ids,
                "legacy_persistent_state_bytes": legacy_p2b.persistent_state_bytes(),
                "generic_persistent_state_bytes": generic_p2b.persistent_state_bytes(),
                "legacy_solver_relative_residual": legacy_p2b.diagnostics[
                    "solver_relative_residual"
                ],
                "generic_solver_relative_residual": generic_p2b.backend.diagnostics[
                    "solver_relative_residual"
                ],
                **p2b_prediction,
            },
        }
        records.append(record)
        print(
            f"TASK {task_id + 1}/{len(training_parts)} "
            f"exact_identity={record['exact']['weights_identical']} "
            f"p2b_identity={record['p2b']['factor_identical']} "
            f"agreement={record['p2b']['prediction_agreement']:.6f}",
            flush=True,
        )
        del codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tensor_flags = [
        record[method][field]
        for record in records
        for method, fields in (
            (
                "exact",
                (
                    "gram_identical",
                    "cross_identical",
                    "counts_identical",
                    "weights_identical",
                    "class_ids_identical",
                ),
            ),
            (
                "p2b",
                (
                    "factor_identical",
                    "cross_identical",
                    "counts_identical",
                    "weights_identical",
                    "class_ids_identical",
                ),
            ),
        )
        for field in fields
    ]
    byte_flags = [
        record[method]["legacy_persistent_state_bytes"]
        == record[method]["generic_persistent_state_bytes"]
        for record in records
        for method in ("exact", "p2b")
    ]
    maximum_residual = max(
        float(record[method][field])
        for record in records
        for method in ("exact", "p2b")
        for field in (
            "legacy_solver_relative_residual",
            "generic_solver_relative_residual",
        )
    )
    minimum_agreement = min(
        float(record[method]["prediction_agreement"])
        for record in records
        for method in ("exact", "p2b")
    )
    maximum_logit_error = max(
        float(record[method]["relative_logit_error"])
        for record in records
        for method in ("exact", "p2b")
    )
    gates = {
        "tensor_identity": all(tensor_flags),
        "persistent_byte_identity": all(byte_flags),
        "prediction_agreement": minimum_agreement
        >= config["gates"]["minimum_prediction_agreement"],
        "relative_logit_error": maximum_logit_error
        <= config["gates"]["maximum_relative_logit_error"],
        "solver_residual": maximum_residual
        <= config["gates"]["maximum_solver_relative_residual"],
    }
    return {
        "records": records,
        "summary": {
            "minimum_prediction_agreement": minimum_agreement,
            "maximum_relative_logit_error": maximum_logit_error,
            "maximum_solver_relative_residual": maximum_residual,
            "final_exact_persistent_state_bytes": exact.persistent_state_bytes(),
            "final_p2b_persistent_state_bytes": generic_p2b.persistent_state_bytes(),
        },
        "gates": gates,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    code_cache_dir = Path(args.code_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M3 requires a clean source checkout")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M3 refuses a visible test.pt")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(
        range(config["num_classes"])
    ):
        raise ValueError("training labels do not match locked classes")
    train_sha256 = _sha256_file(feature_cache_dir / "train.pt")
    code_indices, code_values, code_metadata, projection = _prepare_code_cache(
        train=train,
        train_sha256=train_sha256,
        cache_dir=code_cache_dir,
        config=_cache_config(config),
        device=args.device,
    )
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
    result = _run_tasks(
        config=config,
        train=train,
        code_indices=code_indices,
        code_values=code_values,
        projection=projection,
        training_parts=training_parts,
        validation_parts=validation_parts,
        device=torch.device(args.device),
    )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    provenance = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_sha256": _sha256_file(config_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "legacy_learner_sha256": _sha256_file(
            ROOT / "methods/srq_fly_optimized/learner.py"
        ),
        "generic_backend_sha256": _sha256_file(
            ROOT / "methods/analytic_ridge/backends.py"
        ),
        "fly_frontend_sha256": _sha256_file(ROOT / "methods/frontends/fly.py"),
        "train_sha256": train_sha256,
        "code_identity_sha256": code_metadata["identity_sha256"],
        "projection_sha256": _tensor_content_sha256(projection),
        "training_indices_sha256": _sequence_sha256(training_parts),
        "validation_indices_sha256": _sequence_sha256(validation_parts),
    }
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M3_FLY_REGRESSION"
        if all(result["gates"].values())
        else "FAIL_M3_FLY_REGRESSION",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "provenance": provenance,
        **result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "m3_results.json.tmp"
    destination = output_dir / "m3_results.json"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
    if payload["status"] != "PASS_M3_FLY_REGRESSION":
        raise RuntimeError("M3 regression gate failed")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--code-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
