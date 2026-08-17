"""Train-only raw/FLY complementarity diagnostic for the TWA-FLY branch.

This runner has no held-out evaluation mode. It measures whether a raw Ridge
view can correct errors made by matched FLY before any joint residual learner is
implemented. Sample-level feature and WTA tensors are experiment caches only.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.flycl import select_ridge_parameter
from methods.twa_fly import TWAStatistics
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.twa_fly_pilot import (
    _atomic_json,
    _dense_codes,
    _git_provenance,
    _new_learner,
    _prepare_code_cache,
    _projection_identity,
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _solve_spd,
    _tensor_content_sha256,
)


CONFIG_KEYS = {
    "schema_version", "study_id", "dataset", "model_name", "checkpoint_sha256",
    "seed", "num_classes", "num_tasks", "validation_fraction", "representation",
    "raw_ridge_lambda", "fly_ridge_lower", "fly_ridge_upper", "fusion_alphas",
    "solver_tolerance", "solver_max_iterations", "statistics_dtype", "gate",
}
REPRESENTATION_KEYS = {"expand_dim", "synaptic_degree", "coding_level", "encode_batch_size"}
GATE_KEYS = {
    "required_raw_ridge_lambda", "minimum_oracle_headroom_pp",
    "minimum_fusion_gain_pp", "maximum_solver_relative_residual",
}


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"config keys must be exactly {sorted(CONFIG_KEYS)}")
    if config["schema_version"] != 1:
        raise ValueError("unsupported TWA-FLY D0 config schema")
    if set(config["representation"]) != REPRESENTATION_KEYS or set(config["gate"]) != GATE_KEYS:
        raise ValueError("invalid nested config keys")
    if config["num_classes"] <= 1 or config["num_tasks"] <= 0 or config["num_classes"] % config["num_tasks"]:
        raise ValueError("invalid class/task count")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    alphas = list(map(float, config["fusion_alphas"]))
    if not alphas or 0.0 not in alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("fusion_alphas must be non-negative and include 0")
    if len(set(alphas)) != len(alphas):
        raise ValueError("fusion_alphas must be unique")
    if config["statistics_dtype"] not in {"float32", "float64"}:
        raise ValueError("statistics_dtype must be float32 or float64")
    if config["raw_ridge_lambda"] <= 0 or config["fly_ridge_lower"] >= config["fly_ridge_upper"]:
        raise ValueError("invalid Ridge configuration")
    if config["solver_tolerance"] <= 0 or config["solver_max_iterations"] <= 0:
        raise ValueError("invalid shared learner solver configuration")
    if config["gate"]["maximum_solver_relative_residual"] <= 0:
        raise ValueError("invalid solver residual threshold")
    return config


def _centered_logit_scale(gram: torch.Tensor, weights: torch.Tensor, sample_count: int) -> float:
    """Training-only RMS after removing each sample's class-common offset."""
    centered_weights = weights - weights.mean(dim=1, keepdim=True)
    energy = torch.sum(centered_weights * (gram @ centered_weights)).clamp_min(0)
    denominator = max(int(sample_count) * int(weights.shape[1]), 1)
    scale = torch.sqrt(energy / denominator)
    return float(scale.clamp_min(torch.finfo(weights.dtype).eps).item())


def _relative_ridge_residual(
    gram: torch.Tensor, cross: torch.Tensor, weights: torch.Tensor, ridge: float
) -> float:
    residual = gram @ weights + ridge * weights - cross
    denominator = torch.linalg.vector_norm(cross).clamp_min(torch.finfo(weights.dtype).eps)
    return float((torch.linalg.vector_norm(residual) / denominator).item())


def complementarity_metrics(
    raw_logits: torch.Tensor,
    fly_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    raw_scale: float,
    fly_scale: float,
    fusion_alphas: list[float],
) -> dict:
    if raw_logits.shape != fly_logits.shape or raw_logits.ndim != 2:
        raise ValueError("raw and FLY logits must have the same two-dimensional shape")
    if labels.ndim != 1 or labels.shape[0] != raw_logits.shape[0]:
        raise ValueError("labels must match the logit batch")
    if raw_scale <= 0 or fly_scale <= 0:
        raise ValueError("logit scales must be positive")
    labels = labels.to(raw_logits.device, torch.long)
    raw_predictions = raw_logits.argmax(1)
    fly_predictions = fly_logits.argmax(1)
    raw_correct = raw_predictions.eq(labels)
    fly_correct = fly_predictions.eq(labels)
    sample_count = int(labels.numel())
    both_correct = int((raw_correct & fly_correct).sum().item())
    raw_only = int((raw_correct & ~fly_correct).sum().item())
    fly_only = int((~raw_correct & fly_correct).sum().item())
    both_wrong = sample_count - both_correct - raw_only - fly_only

    raw_centered = raw_logits - raw_logits.mean(dim=1, keepdim=True)
    fly_centered = fly_logits - fly_logits.mean(dim=1, keepdim=True)
    correlation_denominator = (
        torch.linalg.vector_norm(raw_centered) * torch.linalg.vector_norm(fly_centered)
    ).clamp_min(torch.finfo(raw_logits.dtype).eps)
    correlation = float(((raw_centered * fly_centered).sum() / correlation_denominator).item())

    fusion = {}
    normalized_raw = raw_centered / raw_scale
    normalized_fly = fly_centered / fly_scale
    for alpha in map(float, fusion_alphas):
        predictions = (normalized_fly + alpha * normalized_raw).argmax(1)
        fusion[str(alpha)] = float(predictions.eq(labels).float().mean().item() * 100)

    fly_accuracy = 100.0 * (both_correct + fly_only) / sample_count
    raw_accuracy = 100.0 * (both_correct + raw_only) / sample_count
    oracle_accuracy = 100.0 * (both_correct + raw_only + fly_only) / sample_count
    return {
        "sample_count": sample_count,
        "raw_accuracy": raw_accuracy,
        "fly_accuracy": fly_accuracy,
        "oracle_union_accuracy": oracle_accuracy,
        "oracle_headroom_over_fly_pp": oracle_accuracy - fly_accuracy,
        "prediction_disagreement_fraction": float(raw_predictions.ne(fly_predictions).float().mean().item()),
        "centered_logit_correlation": correlation,
        "raw_centered_logit_rms": float(raw_centered.square().mean().sqrt().item()),
        "fly_centered_logit_rms": float(fly_centered.square().mean().sqrt().item()),
        "both_correct": both_correct,
        "raw_only_correct": raw_only,
        "fly_only_correct": fly_only,
        "both_wrong": both_wrong,
        "raw_rescue_rate_among_fly_errors": raw_only / max(raw_only + both_wrong, 1),
        "fusion_accuracy": fusion,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    code_cache_dir = Path(args.code_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    test_path = feature_cache_dir / "test.pt"
    heldout_hidden = not test_path.exists()
    if args.require_test_hidden and not heldout_hidden:
        raise RuntimeError(f"held-out file is visible: {test_path}; rename it before diagnostics")

    cache_args = argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"])
    train, _, cache_metadata = validate_cache(feature_cache_dir, cache_args, load_test=False)
    if cache_metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint hash does not match locked config")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(range(config["num_classes"])):
        raise ValueError("training labels do not match locked global class IDs")

    train_path = feature_cache_dir / "train.pt"
    train_sha256 = _sha256_file(train_path)
    code_indices, code_values, code_metadata, projection = _prepare_code_cache(
        train=train, train_sha256=train_sha256, cache_dir=code_cache_dir,
        config=config, device=args.device,
    )
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float64": torch.float64}[config["statistics_dtype"]]
    raw_dim = int(train["features"].shape[1])
    fly_dim = int(config["representation"]["expand_dim"])
    projection_sha256 = _tensor_content_sha256(projection)
    if projection_sha256 != code_metadata["projection"]["sha256"]:
        raise RuntimeError("runtime projection does not match verified WTA code cache")

    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    tasks = split(train["labels"], class_order, config["num_tasks"])
    train_parts, val_parts = train_validation_indices(
        train["labels"], tasks, config["seed"], config["validation_fraction"]
    )
    statistics = TWAStatistics(raw_dim, fly_dim, config["num_classes"], device=device, dtype=dtype)
    fusion_alphas = list(map(float, config["fusion_alphas"]))
    stage_diagnostics = []
    solver_residuals = []
    started = time.perf_counter()

    for task_index, update_indices in enumerate(train_parts):
        task_started = time.perf_counter()
        x = train["features"][update_indices].to(device=device, dtype=dtype)
        z = _dense_codes(code_indices[update_indices], code_values[update_indices], fly_dim, device, dtype)
        y = train["labels"][update_indices].to(device)
        targets = torch.nn.functional.one_hot(y, config["num_classes"]).to(dtype)
        statistics.update(x, z, y)
        fly_ridge = float(select_ridge_parameter(
            z, targets, config["fly_ridge_lower"], config["fly_ridge_upper"]
        ).item())
        raw_ridge = float(config["raw_ridge_lambda"])
        raw_weights = _solve_spd(statistics.G_xx, statistics.Q_x, raw_ridge)
        fly_weights = _solve_spd(statistics.G_zz, statistics.Q_z, fly_ridge)
        raw_residual = _relative_ridge_residual(
            statistics.G_xx, statistics.Q_x, raw_weights, raw_ridge
        )
        fly_residual = _relative_ridge_residual(
            statistics.G_zz, statistics.Q_z, fly_weights, fly_ridge
        )
        solver_residuals.extend((raw_residual, fly_residual))
        raw_scale = _centered_logit_scale(
            statistics.G_xx, raw_weights, statistics.total_count
        )
        fly_scale = _centered_logit_scale(
            statistics.G_zz, fly_weights, statistics.total_count
        )

        validation_indices = torch.cat(val_parts[:task_index + 1])
        validation_x = train["features"][validation_indices].to(device=device, dtype=dtype)
        validation_z = _dense_codes(
            code_indices[validation_indices], code_values[validation_indices],
            fly_dim, device, dtype,
        )
        metrics = complementarity_metrics(
            validation_x @ raw_weights,
            validation_z @ fly_weights,
            train["labels"][validation_indices].to(device),
            raw_scale=raw_scale,
            fly_scale=fly_scale,
            fusion_alphas=fusion_alphas,
        )
        metrics.update({
            "task": task_index + 1,
            "seen_classes": (task_index + 1) * config["num_classes"] // config["num_tasks"],
            "raw_ridge_lambda": raw_ridge,
            "fly_ridge_lambda": fly_ridge,
            "raw_training_logit_scale": raw_scale,
            "fly_training_logit_scale": fly_scale,
            "raw_solver_relative_residual": raw_residual,
            "fly_solver_relative_residual": fly_residual,
        })
        stage_diagnostics.append(metrics)
        print(
            f"TASK {task_index+1}/{len(train_parts)} fly={metrics['fly_accuracy']:.4f} "
            f"raw={metrics['raw_accuracy']:.4f} oracle={metrics['oracle_union_accuracy']:.4f} "
            f"raw_only={metrics['raw_only_correct']} disagree={100*metrics['prediction_disagreement_fraction']:.2f}% "
            f"elapsed={time.perf_counter()-task_started:.1f}s total={(time.perf_counter()-started)/60:.1f}m",
            flush=True,
        )
        del x, z, targets, validation_x, validation_z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fly_average = sum(row["fly_accuracy"] for row in stage_diagnostics) / len(stage_diagnostics)
    raw_average = sum(row["raw_accuracy"] for row in stage_diagnostics) / len(stage_diagnostics)
    oracle_average = sum(row["oracle_union_accuracy"] for row in stage_diagnostics) / len(stage_diagnostics)
    alpha_results = []
    for alpha in fusion_alphas:
        accuracy = sum(row["fusion_accuracy"][str(alpha)] for row in stage_diagnostics) / len(stage_diagnostics)
        alpha_results.append({
            "alpha": alpha,
            "validation_average_accuracy": accuracy,
            "gain_over_fly_pp": accuracy - fly_average,
        })
    selected_fusion = max(
        alpha_results, key=lambda row: (row["validation_average_accuracy"], -row["alpha"])
    )
    maximum_residual = max(solver_residuals)
    thresholds = config["gate"]
    gates = {
        "projection_cache_verified": bool(code_metadata["projection"]["probe"]["verified"]),
        "raw_ridge_matches_locked_protocol": bool(
            config["raw_ridge_lambda"] == thresholds["required_raw_ridge_lambda"]
        ),
        "oracle_headroom": bool(
            oracle_average - fly_average >= thresholds["minimum_oracle_headroom_pp"]
        ),
        "fusion_gain": bool(
            selected_fusion["gain_over_fly_pp"] >= thresholds["minimum_fusion_gain_pp"]
        ),
        "numerical_stability": bool(
            maximum_residual <= thresholds["maximum_solver_relative_residual"]
        ),
        "heldout_test_remained_hidden": bool(heldout_hidden),
    }
    decision = "REVIEW_JOINT_RESIDUAL_IMPLEMENTATION" if all(gates.values()) else "STOP_TWO_VIEW_BRANCH"
    gate = {
        "decision": decision,
        "gates": gates,
        "diagnostics": {
            "fly_validation_average_accuracy": fly_average,
            "raw_validation_average_accuracy": raw_average,
            "oracle_union_validation_average_accuracy": oracle_average,
            "oracle_headroom_over_fly_pp": oracle_average - fly_average,
            "selected_fusion": selected_fusion,
            "maximum_solver_relative_residual": maximum_residual,
        },
    }
    provenance = {
        **_git_provenance(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "feature_cache_dir": str(feature_cache_dir),
        "feature_cache_metadata": cache_metadata,
        "train_pt_sha256": train_sha256,
        "code_cache_dir": str(code_cache_dir),
        "code_cache_identity_sha256": code_metadata["identity_sha256"],
        "projection_sha256": projection_sha256,
        "projection_identity": _projection_identity(config, raw_dim),
        "class_order": class_order,
        "class_order_sha256": _sha256_bytes(",".join(map(str, class_order)).encode("ascii")),
        "training_indices_sha256": _sequence_sha256(train_parts),
        "validation_indices_sha256": _sequence_sha256(val_parts),
        "heldout_test_path_visible": not heldout_hidden,
    }
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "selection_protocol": "train-only cumulative per-stage complementarity and normalized-logit fusion",
        "uses_test_set": False,
        "held_out_test_authorized": False,
        "config": config,
        "run_provenance": provenance,
        "code_cache": code_metadata,
        "stage_diagnostics": stage_diagnostics,
        "fusion_candidates": alpha_results,
        "gate": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "diagnostics.json", payload)
    _atomic_json(output_dir / "gate_results.json", gate)
    print(json.dumps(gate, indent=2), flush=True)
    print("D0 TRAIN-ONLY COMPLETE. Held-out evaluation remains unauthorized.", flush=True)
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
