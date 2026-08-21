"""Locked CUB-200 train-only replication of the SRQ-FLY state claim.

There is deliberately no held-out evaluation mode. Frozen train features and
sample-level WTA codes are resumable experiment infrastructure, never learner
state, and are not copied into the evidence bundle by this runner.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly import projected_srq_state_bytes
from tools import srq_fly_d0 as d0
from tools import srq_fly_d1 as d1
from tools import srq_fly_d2_state_match as d2
from tools import srq_fly_d21_lambda_robustness as d21
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.tail_fly_phasea import _git_provenance, _load_unit, _save_unit, _unit_path
from tools.twa_fly_pilot import (
    _prepare_code_cache,
    _sequence_sha256,
    _sha256_bytes,
    _sha256_file,
    _tensor_content_sha256,
)


TOP_KEYS = {
    "schema_version", "study_id", "dataset_identity", "model_name",
    "feature_dim", "checkpoint_sha256", "seed", "num_classes", "num_tasks",
    "outer_validation_fraction", "inner_validation_fraction",
    "statistics_dtype", "solver_dtype", "raw_statistics_dtype",
    "selection_lambdas", "raw_selection_lambdas", "large_representation",
    "matched_representation", "storage", "expected_state", "gates",
}
DATASET_KEYS = {
    "dataset", "dataset_version", "dataset_identity_sha256",
    "class_mapping_sha256", "train_content_manifest_sha256",
    "test_content_manifest_sha256", "train_samples", "test_samples",
}
STATE_KEYS = {
    "large_projection_nonzeros", "matched_projection_nonzeros",
    "exact_large_bytes", "srq_large_bytes", "exact_matched_bytes",
}
GATE_KEYS = {
    "maximum_solver_relative_residual",
    "maximum_average_gap_to_exact_large_pp",
    "maximum_final_gap_to_exact_large_pp", "minimum_prediction_agreement",
    "maximum_state_fraction_of_exact_large", "maximum_state_mismatch_fraction",
    "minimum_average_gain_over_state_matched_fly_pp",
    "minimum_final_gain_over_state_matched_fly_pp",
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config key: {key}")
        result[key] = value
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _positive_grid(values, name: str) -> list[float]:
    if (
        not isinstance(values, list) or not values
        or any(not isinstance(value, (int, float)) for value in values)
    ):
        raise ValueError(f"{name} must be a non-empty numeric list")
    parsed = list(map(float, values))
    if (
        any(not math.isfinite(value) or value <= 0 for value in parsed)
        or parsed != sorted(set(parsed))
    ):
        raise ValueError(f"{name} must be sorted, unique, finite, and positive")
    return parsed


def _projection_state_bytes(
    *, feature_dim: int, expand_dim: int, nonzeros: int, num_classes: int,
    gram_or_factor_bytes: int,
) -> int:
    projection = nonzeros * 12 + (feature_dim + 1) * 8
    classifier = 2 * expand_dim * num_classes * 4 + num_classes * 4
    return projection + gram_or_factor_bytes + classifier


def _read_config(path: Path) -> dict:
    config = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("config keys/schema mismatch")
    identity = config["dataset_identity"]
    if set(identity) != DATASET_KEYS or identity["dataset"] != "CUB-200-2011":
        raise ValueError("locked CUB dataset identity mismatch")
    if identity["train_samples"] <= 0 or identity["test_samples"] <= 0:
        raise ValueError("invalid CUB sample counts")
    if config["seed"] != 2025:
        raise ValueError("new SRQ-FLY protocols require repository seed 2025")
    if (
        config["feature_dim"] <= 0 or config["num_classes"] != 200
        or config["num_tasks"] != 20
        or config["num_classes"] % config["num_tasks"]
    ):
        raise ValueError("invalid CUB class/task dimensions")
    for fraction in ("outer_validation_fraction", "inner_validation_fraction"):
        if not 0 < config[fraction] < 1:
            raise ValueError(f"{fraction} must be in (0,1)")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("SRQ/FLY state study requires float32")
    if config["raw_statistics_dtype"] != "float64":
        raise ValueError("CUB raw-Ridge numerical control is locked to float64")
    _positive_grid(config["selection_lambdas"], "selection_lambdas")
    _positive_grid(config["raw_selection_lambdas"], "raw_selection_lambdas")
    for name in ("large_representation", "matched_representation"):
        representation = config[name]
        if set(representation) != d0.REPRESENTATION_KEYS:
            raise ValueError(f"{name} keys mismatch")
        if (
            min(
                representation["expand_dim"], representation["synaptic_degree"],
                representation["encode_batch_size"],
                representation["evaluation_batch_size"],
            ) <= 0
            or representation["synaptic_degree"] > config["feature_dim"]
            or not 0 < representation["coding_level"] <= 1
        ):
            raise ValueError(f"invalid {name}")
    large = config["large_representation"]
    matched = config["matched_representation"]
    if large["expand_dim"] <= matched["expand_dim"]:
        raise ValueError("large representation must exceed state-matched representation")
    if (
        large["synaptic_degree"] != matched["synaptic_degree"]
        or large["coding_level"] != matched["coding_level"]
    ):
        raise ValueError("representations must share projection/WTA policy")
    if set(config["storage"]) != d0.STORAGE_KEYS or min(config["storage"].values()) <= 0:
        raise ValueError("invalid storage configuration")
    state = config["expected_state"]
    if set(state) != STATE_KEYS or min(state.values()) <= 0:
        raise ValueError("invalid expected state")
    large_nnz = state["large_projection_nonzeros"]
    matched_nnz = state["matched_projection_nonzeros"]
    if large_nnz > large["expand_dim"] * large["synaptic_degree"]:
        raise ValueError("large projection nonzero count is impossible")
    if matched_nnz > matched["expand_dim"] * matched["synaptic_degree"]:
        raise ValueError("matched projection nonzero count is impossible")
    exact_large = _projection_state_bytes(
        feature_dim=config["feature_dim"], expand_dim=large["expand_dim"],
        nonzeros=large_nnz, num_classes=config["num_classes"],
        gram_or_factor_bytes=large["expand_dim"] ** 2 * 4,
    )
    exact_matched = _projection_state_bytes(
        feature_dim=config["feature_dim"], expand_dim=matched["expand_dim"],
        nonzeros=matched_nnz, num_classes=config["num_classes"],
        gram_or_factor_bytes=matched["expand_dim"] ** 2 * 4,
    )
    nominal_srq = projected_srq_state_bytes(
        feature_dim=config["feature_dim"], expand_dim=large["expand_dim"],
        synaptic_degree=large["synaptic_degree"], num_classes=config["num_classes"],
        block_size=config["storage"]["block_size"],
        group_size=config["storage"]["group_size"],
    )["compressed_total_bytes"]
    missing_projection_bytes = (
        large["expand_dim"] * large["synaptic_degree"] - large_nnz
    ) * 12
    if (
        exact_large != state["exact_large_bytes"]
        or exact_matched != state["exact_matched_bytes"]
        or nominal_srq - missing_projection_bytes != state["srq_large_bytes"]
    ):
        raise ValueError("expected state does not match analytic accounting")
    gates = config["gates"]
    if set(gates) != GATE_KEYS:
        raise ValueError("gate keys mismatch")
    if (
        gates["maximum_solver_relative_residual"] <= 0
        or gates["maximum_average_gap_to_exact_large_pp"] < 0
        or gates["maximum_final_gap_to_exact_large_pp"] < 0
        or not 0 <= gates["minimum_prediction_agreement"] <= 1
        or not 0 < gates["maximum_state_fraction_of_exact_large"] <= 1
        or not 0 <= gates["maximum_state_mismatch_fraction"] < 1
        or gates["minimum_average_gain_over_state_matched_fly_pp"] < 0
        or gates["minimum_final_gain_over_state_matched_fly_pp"] < 0
    ):
        raise ValueError("invalid gates")
    return config


def _cache_config(config: dict, representation: dict) -> dict:
    return {
        "seed": config["seed"], "num_classes": config["num_classes"],
        "representation": dict(representation),
        "statistics_dtype": config["statistics_dtype"],
        "raw_ridge_lambda": 1.0, "solver_tolerance": 1e-5,
        "solver_max_iterations": 100,
    }


def _validate_dataset_audit(path: Path, config: dict) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = config["dataset_identity"]
    expected = {
        "dataset": identity["dataset"],
        "dataset_identity_sha256": identity["dataset_identity_sha256"],
        "class_mapping_sha256": identity["class_mapping_sha256"],
        "cross_split_duplicate_content_count": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"CUB dataset audit mismatch for {key}")
    if payload.get("train", {}).get("image_count") != identity["train_samples"]:
        raise ValueError("CUB train count mismatch")
    if payload.get("test", {}).get("image_count") != identity["test_samples"]:
        raise ValueError("CUB test count mismatch")
    if payload.get("train", {}).get("content_manifest_sha256") != identity["train_content_manifest_sha256"]:
        raise ValueError("CUB train manifest mismatch")
    if payload.get("test", {}).get("content_manifest_sha256") != identity["test_content_manifest_sha256"]:
        raise ValueError("CUB test manifest mismatch")
    return payload, _sha256_file(path)


def _validate_feature_cache(config: dict, feature_cache_dir: Path):
    identity = config["dataset_identity"]
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=identity["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    expected = {
        "dataset": identity["dataset"], "dataset_version": identity["dataset_version"],
        "backbone_model": config["model_name"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "preprocessing": "vit", "feature_dim": config["feature_dim"],
        "finite": True, "test_features_materialized": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"feature-cache metadata mismatch for {key}")
    if metadata.get("split_sizes") != {
        "train": identity["train_samples"], "test": identity["test_samples"]
    }:
        raise ValueError("feature-cache split sizes mismatch")
    if tuple(train["features"].shape) != (identity["train_samples"], config["feature_dim"]):
        raise ValueError("feature-cache train shape mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(range(config["num_classes"])):
        raise ValueError("training labels do not match locked CUB classes")
    if not bool(torch.isfinite(train["features"]).all()):
        raise ValueError("training features contain NaN or Inf")
    return train, metadata


def _verify_projection_prefix(large: torch.Tensor, matched: torch.Tensor) -> dict:
    large_dense = large.to_dense() if large.layout == torch.sparse_csc else large
    matched_dense = matched.to_dense() if matched.layout == torch.sparse_csc else matched
    if large_dense.shape[1] != matched_dense.shape[1] or large_dense.shape[0] < matched_dense.shape[0]:
        raise ValueError("projection shapes cannot form a row prefix")
    if not torch.equal(large_dense[: matched_dense.shape[0]].cpu(), matched_dense.cpu()):
        raise ValueError("state-matched projection is not the exact prefix of FLY-10000")
    return {
        "verified": True,
        "large_shape": list(large_dense.shape),
        "matched_shape": list(matched_dense.shape),
        "semantics": "same seeded projection rows; WTA Top-K is recomputed at each dimension",
    }


def _candidate_name(scope: str, index: int, ridge_lambda: float) -> str:
    value = f"{ridge_lambda:.0e}".replace("+", "")
    return f"inner_{scope}_{index:02d}_lambda_{value}"


def _choose_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("no candidates")
    return min(
        candidates,
        key=lambda item: (-float(item["validation_average_accuracy"]), float(item["ridge_lambda"])),
    )


def _exact_config(config: dict, ridge_lambda: float) -> dict:
    return {
        "statistics_dtype": config["statistics_dtype"],
        "ridge_lambda": float(ridge_lambda),
    }


def _evaluate_exact(
    *, name: str, ridge_lambda: float, config: dict, representation: dict,
    train: dict, code_indices: torch.Tensor, code_values: torch.Tensor,
    projection: torch.Tensor, training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor], device: torch.device,
) -> dict:
    return d0._evaluate_exact(
        name=name, config=_exact_config(config, ridge_lambda),
        representation=representation, train=train, code_indices=code_indices,
        code_values=code_values, projection=projection,
        training_parts=training_parts, validation_parts=validation_parts, device=device,
    )


def _evaluate_raw(
    *, name: str, ridge_lambda: float, config: dict, train: dict,
    training_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    result = d0._evaluate_raw(
        config={
            "statistics_dtype": config["raw_statistics_dtype"],
            "raw_ridge_lambda": float(ridge_lambda),
        },
        train=train, training_parts=training_parts,
        validation_parts=validation_parts, device=device,
    )
    result["method"] = name
    return result


def _validate_result(
    result: dict, *, name: str, ridge_lambda: float, num_tasks: int,
    expected_state_bytes: int | None = None,
) -> None:
    stages = result.get("stage_accuracy")
    forbidden = ("sample", "feature", "label", "code", "history")
    if (
        any(any(token in str(key).lower() for token in forbidden) for key in result)
        or result.get("method") != name or result.get("status") != "complete"
        or result.get("uses_test_set") is not False
        or result.get("exemplar_free") is not True
        or float(result.get("ridge_lambda", -1)) != float(ridge_lambda)
        or not isinstance(stages, list) or len(stages) != num_tasks
        or any(not math.isfinite(float(value)) for value in stages)
    ):
        raise ValueError(f"invalid result contract for {name}")
    if expected_state_bytes is not None and result.get("persistent_state_bytes") != expected_state_bytes:
        raise ValueError(f"persistent state mismatch for {name}")
    average = float(result.get("validation_average_accuracy", float("nan")))
    residual = float(result.get("maximum_solver_relative_residual", float("nan")))
    if (
        not math.isfinite(average)
        or abs(average - sum(map(float, stages)) / len(stages)) > 1e-10
        or not math.isfinite(residual) or residual < 0
    ):
        raise ValueError(f"invalid metrics for {name}")


def _paired_config(config: dict, ridge_lambda: float) -> dict:
    return {
        "seed": config["seed"], "statistics_dtype": config["statistics_dtype"],
        "solver_dtype": config["solver_dtype"], "ridge_lambda": float(ridge_lambda),
        "large_representation": dict(config["large_representation"]),
        "storage": dict(config["storage"]),
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    large_cache_dir = Path(args.large_code_cache_dir).resolve()
    matched_cache_dir = Path(args.matched_code_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    if args.require_test_hidden and (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("held-out test.pt is visible")
    dataset_audit, dataset_audit_sha256 = _validate_dataset_audit(
        Path(args.dataset_audit).resolve(), config
    )
    train, metadata = _validate_feature_cache(config, feature_cache_dir)
    train_path = feature_cache_dir / "train.pt"
    train_sha256 = _sha256_file(train_path)
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    outer_training, outer_validation = train_validation_indices(
        train["labels"], task_indices, config["seed"], config["outer_validation_fraction"]
    )
    inner_fit, inner_validation = train_validation_indices(
        train["labels"], outer_training, config["seed"], config["inner_validation_fraction"]
    )
    d21._validate_nested_parts(
        outer_training, outer_validation, inner_fit, inner_validation
    )
    print("CACHE START FLY-10000", flush=True)
    large = _prepare_code_cache(
        train=train, train_sha256=train_sha256, cache_dir=large_cache_dir,
        config=_cache_config(config, config["large_representation"]), device=args.device,
    )
    print("CACHE START FLY-4518", flush=True)
    matched = _prepare_code_cache(
        train=train, train_sha256=train_sha256, cache_dir=matched_cache_dir,
        config=_cache_config(config, config["matched_representation"]), device=args.device,
    )
    prefix = _verify_projection_prefix(large[3], matched[3])
    state_expected = config["expected_state"]
    observed_nnz = {
        "large": int(large[3].values().numel()),
        "matched": int(matched[3].values().numel()),
    }
    if observed_nnz != {
        "large": state_expected["large_projection_nonzeros"],
        "matched": state_expected["matched_projection_nonzeros"],
    }:
        raise ValueError(f"projection nonzero identity mismatch: {observed_nnz}")
    git = _git_provenance()
    context = {
        "config_sha256": _sha256_file(config_path),
        "dataset_audit_sha256": dataset_audit_sha256,
        "dataset_identity_sha256": dataset_audit["dataset_identity_sha256"],
        "train_sha256": train_sha256,
        "large_code_identity": large[2]["identity_sha256"],
        "matched_code_identity": matched[2]["identity_sha256"],
        "large_projection_sha256": _tensor_content_sha256(large[3]),
        "matched_projection_sha256": _tensor_content_sha256(matched[3]),
        "outer_training_indices_sha256": _sequence_sha256(outer_training),
        "outer_validation_indices_sha256": _sequence_sha256(outer_validation),
        "inner_fit_indices_sha256": _sequence_sha256(inner_fit),
        "inner_validation_indices_sha256": _sequence_sha256(inner_validation),
        "runner_git_commit": git["git_commit"], "runner_git_dirty": git["git_dirty"],
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "d0_runner_source_sha256": _sha256_file(ROOT / "tools/srq_fly_d0.py"),
        "d1_runner_source_sha256": _sha256_file(ROOT / "tools/srq_fly_d1.py"),
        "learner_source_sha256": _sha256_file(ROOT / "methods/srq_fly/learner.py"),
        "storage_source_sha256": _sha256_file(ROOT / "methods/srq_fly/storage.py"),
    }
    context_sha256 = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    def select_exact(scope, representation, cache, expected_state):
        candidates = []
        for index, ridge_lambda in enumerate(map(float, config["selection_lambdas"])):
            name = _candidate_name(scope, index, ridge_lambda)
            path = _unit_path(output_dir, name)
            result = _load_unit(path, context_sha256)
            if result is None:
                print(f"INNER START {scope} lambda={ridge_lambda:g}", flush=True)
                result = _save_unit(path, context_sha256, _evaluate_exact(
                    name=name, ridge_lambda=ridge_lambda, config=config,
                    representation=representation, train=train,
                    code_indices=cache[0], code_values=cache[1], projection=cache[3],
                    training_parts=inner_fit, validation_parts=inner_validation,
                    device=device,
                ))
                print(f"INNER DONE {scope} lambda={ridge_lambda:g} AA={result['validation_average_accuracy']:.6f}", flush=True)
            _validate_result(
                result, name=name, ridge_lambda=ridge_lambda,
                num_tasks=config["num_tasks"], expected_state_bytes=expected_state,
            )
            candidates.append(result)
        return candidates, _choose_candidate(candidates)

    large_candidates, large_selected = select_exact(
        "fly10000", config["large_representation"], large,
        state_expected["exact_large_bytes"],
    )
    matched_candidates, matched_selected = select_exact(
        "fly4518", config["matched_representation"], matched,
        state_expected["exact_matched_bytes"],
    )
    raw_candidates = []
    for index, ridge_lambda in enumerate(map(float, config["raw_selection_lambdas"])):
        name = _candidate_name("raw", index, ridge_lambda)
        path = _unit_path(output_dir, name)
        result = _load_unit(path, context_sha256)
        if result is None:
            print(f"INNER START raw lambda={ridge_lambda:g}", flush=True)
            result = _save_unit(path, context_sha256, _evaluate_raw(
                name=name, ridge_lambda=ridge_lambda, config=config, train=train,
                training_parts=inner_fit, validation_parts=inner_validation, device=device,
            ))
            print(f"INNER DONE raw lambda={ridge_lambda:g} AA={result['validation_average_accuracy']:.6f}", flush=True)
        _validate_result(
            result, name=name, ridge_lambda=ridge_lambda, num_tasks=config["num_tasks"]
        )
        raw_candidates.append(result)
    raw_selected = _choose_candidate(raw_candidates)
    selection_payload = {
        "schema_version": 1,
        "protocol": "nested inner split of outer-training only",
        "uses_outer_validation_for_selection": False, "uses_test_set": False,
        "context_sha256": context_sha256,
        "tie_break": "maximum inner AA, then smaller lambda",
        "selected_lambdas": {
            "exact_fly_10000_and_srq_10000": large_selected["ridge_lambda"],
            "state_matched_exact_fly_4518": matched_selected["ridge_lambda"],
            "raw_ridge": raw_selected["ridge_lambda"],
        },
        "large_candidates": large_candidates,
        "matched_candidates": matched_candidates,
        "raw_candidates": raw_candidates,
    }
    selection_path = output_dir / "lambda_selection.json"
    _atomic_json(selection_path, selection_payload)
    print(f"LOCKED lambda large/srq={large_selected['ridge_lambda']:g} matched={matched_selected['ridge_lambda']:g} raw={raw_selected['ridge_lambda']:g}", flush=True)
    outer_context = {
        **context, "lambda_selection_sha256": _sha256_file(selection_path),
        "selected_lambdas": selection_payload["selected_lambdas"],
    }
    outer_context_sha256 = _sha256_bytes(json.dumps(outer_context, sort_keys=True).encode())

    paired_path = _unit_path(output_dir, "outer_paired_exact_srq_10000")
    paired = _load_unit(paired_path, outer_context_sha256)
    if paired is None:
        print("OUTER START paired exact/SRQ FLY-10000", flush=True)
        paired = _save_unit(paired_path, outer_context_sha256, d1._evaluate_paired_exact_srq(
            config=_paired_config(config, large_selected["ridge_lambda"]),
            train=train, code_indices=large[0], code_values=large[1],
            projection=large[3], training_parts=outer_training,
            validation_parts=outer_validation, device=device,
        ))
        print("OUTER DONE paired exact/SRQ FLY-10000", flush=True)
    exact_large, srq = paired["exact"], paired["srq"]
    _validate_result(
        exact_large, name="exact_fly_10000", ridge_lambda=large_selected["ridge_lambda"],
        num_tasks=config["num_tasks"], expected_state_bytes=state_expected["exact_large_bytes"],
    )
    _validate_result(
        srq, name="srq_int8", ridge_lambda=large_selected["ridge_lambda"],
        num_tasks=config["num_tasks"], expected_state_bytes=state_expected["srq_large_bytes"],
    )

    def outer_unit(name, evaluator, ridge_lambda, expected_state=None):
        path = _unit_path(output_dir, name)
        result = _load_unit(path, outer_context_sha256)
        if result is None:
            print(f"OUTER START {name}", flush=True)
            result = _save_unit(path, outer_context_sha256, evaluator())
            print(f"OUTER DONE {name} AA={result['validation_average_accuracy']:.6f}", flush=True)
        _validate_result(
            result, name=name, ridge_lambda=ridge_lambda,
            num_tasks=config["num_tasks"], expected_state_bytes=expected_state,
        )
        return result

    matched_outer = outer_unit(
        "exact_fly_4518",
        lambda: _evaluate_exact(
            name="exact_fly_4518", ridge_lambda=matched_selected["ridge_lambda"],
            config=config, representation=config["matched_representation"], train=train,
            code_indices=matched[0], code_values=matched[1], projection=matched[3],
            training_parts=outer_training, validation_parts=outer_validation, device=device,
        ),
        matched_selected["ridge_lambda"], state_expected["exact_matched_bytes"],
    )
    raw_outer = outer_unit(
        "raw_ridge",
        lambda: _evaluate_raw(
            name="raw_ridge", ridge_lambda=raw_selected["ridge_lambda"],
            config=config, train=train, training_parts=outer_training,
            validation_parts=outer_validation, device=device,
        ),
        raw_selected["ridge_lambda"],
    )
    average_gap_large = exact_large["validation_average_accuracy"] - srq["validation_average_accuracy"]
    final_gap_large = exact_large["stage_accuracy"][-1] - srq["stage_accuracy"][-1]
    average_gain_matched = srq["validation_average_accuracy"] - matched_outer["validation_average_accuracy"]
    final_gain_matched = srq["stage_accuracy"][-1] - matched_outer["stage_accuracy"][-1]
    state_mismatch = abs(srq["persistent_state_bytes"] - matched_outer["persistent_state_bytes"]) / srq["persistent_state_bytes"]
    gates_config = config["gates"]
    residuals = [
        *(item["maximum_solver_relative_residual"] for item in large_candidates),
        *(item["maximum_solver_relative_residual"] for item in matched_candidates),
        *(item["maximum_solver_relative_residual"] for item in raw_candidates),
        exact_large["maximum_solver_relative_residual"],
        srq["maximum_solver_relative_residual"],
        matched_outer["maximum_solver_relative_residual"],
        raw_outer["maximum_solver_relative_residual"],
    ]
    raw_dominates = (
        raw_outer["validation_average_accuracy"] >= srq["validation_average_accuracy"]
        and raw_outer["persistent_state_bytes"] <= srq["persistent_state_bytes"]
        and (
            raw_outer["validation_average_accuracy"] > srq["validation_average_accuracy"]
            or raw_outer["persistent_state_bytes"] < srq["persistent_state_bytes"]
        )
    )
    gates = {
        "selection_complete": len(large_candidates) == len(config["selection_lambdas"])
        and len(matched_candidates) == len(config["selection_lambdas"])
        and len(raw_candidates) == len(config["raw_selection_lambdas"]),
        "outer_validation_not_used_for_selection": selection_payload[
            "uses_outer_validation_for_selection"
        ] is False,
        "heldout_test_remained_hidden": not (feature_cache_dir / "test.pt").exists(),
        "projection_prefix_verified": prefix["verified"],
        "numerical_stability": max(map(float, residuals)) <= gates_config["maximum_solver_relative_residual"],
        "runtime_state_matches_preregistered_accounting": exact_large["persistent_state_bytes"] == state_expected["exact_large_bytes"]
        and srq["persistent_state_bytes"] == state_expected["srq_large_bytes"]
        and matched_outer["persistent_state_bytes"] == state_expected["exact_matched_bytes"],
        "srq_tracks_exact_large_average": average_gap_large <= gates_config["maximum_average_gap_to_exact_large_pp"],
        "srq_tracks_exact_large_final": final_gap_large <= gates_config["maximum_final_gap_to_exact_large_pp"],
        "prediction_agreement_within_gate": paired["minimum_prediction_agreement"] >= gates_config["minimum_prediction_agreement"],
        "compressed_state_fraction_within_gate": srq["persistent_state_bytes"] / exact_large["persistent_state_bytes"] <= gates_config["maximum_state_fraction_of_exact_large"],
        "state_budget_matched": state_mismatch <= gates_config["maximum_state_mismatch_fraction"],
        "srq_average_gain_over_state_matched_fly": average_gain_matched >= gates_config["minimum_average_gain_over_state_matched_fly_pp"],
        "srq_final_gain_over_state_matched_fly": final_gain_matched >= gates_config["minimum_final_gain_over_state_matched_fly_pp"],
        "not_pareto_dominated_by_raw_ridge": not raw_dominates,
    }
    decision = "PASS_REVIEW_D3" if all(gates.values()) else "STOP_SRQ_FLY_D3"
    payload = {
        "schema_version": 1, "study_id": config["study_id"], "status": decision,
        "uses_test_set": False, "held_out_test_authorized": False,
        "uses_outer_validation_for_selection": False,
        "provenance": outer_context, "source_feature_metadata": metadata,
        "class_order": class_order, "projection_prefix": prefix,
        "projection_nonzeros": observed_nnz,
        "selection": selection_payload["selected_lambdas"],
        "results": [exact_large, srq, matched_outer, raw_outer],
        "paired_diagnostics": paired["paired_diagnostics"],
        "comparison": {
            "srq_average_difference_from_exact_large_pp": -average_gap_large,
            "srq_final_difference_from_exact_large_pp": -final_gap_large,
            "srq_average_gain_over_state_matched_fly_pp": average_gain_matched,
            "srq_final_gain_over_state_matched_fly_pp": final_gain_matched,
            "state_mismatch_fraction": state_mismatch,
            "srq_state_fraction_of_exact_large": srq["persistent_state_bytes"] / exact_large["persistent_state_bytes"],
            "raw_ridge_pareto_dominates_srq": raw_dominates,
        },
        "gates": gates,
    }
    _atomic_json(output_dir / "d3_results.json", payload)
    print(f"SRQ-FLY D3 {decision}", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--large-code-cache-dir", required=True)
    parser.add_argument("--matched-code-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
