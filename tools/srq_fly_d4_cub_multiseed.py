"""Prospective five-fresh-seed CUB train-only confirmation for SRQ-FLY.

FLY hyperparameters are transferred unchanged from D3. Only one raw-Ridge
lambda is selected by mean inner-validation accuracy across the five seeds.
The runner has no held-out evaluation mode.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly import projected_srq_state_bytes
from tools import srq_fly_d1 as d1
from tools import srq_fly_d3_cub as d3
from tools.experiment_runner import split, train_validation_indices
from tools.tail_fly_phasea import _git_provenance, _load_unit, _save_unit, _unit_path
from tools.twa_fly_pilot import (
    _prepare_code_cache, _sequence_sha256, _sha256_bytes, _sha256_file,
    _tensor_content_sha256,
)


TOP_KEYS = {
    "schema_version", "study_id", "dataset_identity", "model_name",
    "feature_dim", "checkpoint_sha256", "seeds", "num_classes", "num_tasks",
    "outer_validation_fraction", "inner_validation_fraction",
    "statistics_dtype", "solver_dtype", "raw_statistics_dtype",
    "fixed_fly_ridge_lambda", "raw_selection_lambdas",
    "large_representation", "matched_representation", "storage",
    "expected_state", "reference_d3", "gates",
}
STATE_KEYS = {
    "nominal_large_projection_entries", "nominal_matched_projection_entries",
    "maximum_missing_projection_entries", "nominal_exact_large_bytes",
    "nominal_srq_large_bytes", "nominal_exact_matched_bytes",
}
REFERENCE_KEYS = {
    "artifact_sha256", "result_sha256", "config_sha256", "runner_git_commit",
    "train_sha256", "status", "selected_fly_ridge_lambda",
    "srq_average_accuracy", "srq_final_accuracy", "srq_persistent_state_bytes",
    "matched_average_accuracy", "matched_final_accuracy",
    "matched_persistent_state_bytes",
}
GATE_KEYS = {
    "maximum_search_candidate_solver_relative_residual",
    "maximum_outer_solver_relative_residual",
    "maximum_average_gap_to_exact_large_pp",
    "maximum_final_gap_to_exact_large_pp", "minimum_prediction_agreement",
    "maximum_state_fraction_of_exact_large", "maximum_state_mismatch_fraction",
    "minimum_mean_average_gain_over_state_matched_fly_pp",
    "minimum_median_average_gain_over_state_matched_fly_pp",
    "minimum_mean_final_gain_over_state_matched_fly_pp",
    "minimum_seed_win_fraction", "maximum_worst_seed_average_loss_pp",
}


def _read_config(path: Path) -> dict:
    config = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=d3._reject_duplicate_keys
    )
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("config keys/schema mismatch")
    identity = config["dataset_identity"]
    if set(identity) != d3.DATASET_KEYS or identity["dataset"] != "CUB-200-2011":
        raise ValueError("locked CUB identity mismatch")
    if config["seeds"] != [2026, 2027, 2028, 2029, 2030]:
        raise ValueError("D4 requires five fresh preregistered seeds")
    if config["num_classes"] != 200 or config["num_tasks"] != 20:
        raise ValueError("D4 requires the complete 200-class/20-task stream")
    if config["feature_dim"] <= 0:
        raise ValueError("feature_dim must be positive")
    for name in ("outer_validation_fraction", "inner_validation_fraction"):
        if not 0 < config[name] < 1:
            raise ValueError(f"{name} must be in (0,1)")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("SRQ/FLY is locked to float32")
    if config["raw_statistics_dtype"] != "float64":
        raise ValueError("raw Ridge is locked to float64")
    if not math.isfinite(float(config["fixed_fly_ridge_lambda"])) or config["fixed_fly_ridge_lambda"] <= 0:
        raise ValueError("fixed FLY Ridge must be positive")
    raw_grid = d3._positive_grid(config["raw_selection_lambdas"], "raw_selection_lambdas")
    if min(raw_grid) >= 10 or max(raw_grid) <= 10:
        raise ValueError("raw grid must bracket the D3 boundary value 10")
    for name in ("large_representation", "matched_representation"):
        representation = config[name]
        if set(representation) != d3.d0.REPRESENTATION_KEYS:
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
    large, matched = config["large_representation"], config["matched_representation"]
    if (
        large["expand_dim"] <= matched["expand_dim"]
        or large["synaptic_degree"] != matched["synaptic_degree"]
        or large["coding_level"] != matched["coding_level"]
    ):
        raise ValueError("invalid matched representation relation")
    if set(config["storage"]) != d3.d0.STORAGE_KEYS or min(config["storage"].values()) <= 0:
        raise ValueError("invalid storage")
    state = config["expected_state"]
    if set(state) != STATE_KEYS or min(state.values()) <= 0:
        raise ValueError("invalid expected-state keys")
    if state["nominal_large_projection_entries"] != large["expand_dim"] * large["synaptic_degree"]:
        raise ValueError("large nominal projection accounting mismatch")
    if state["nominal_matched_projection_entries"] != matched["expand_dim"] * matched["synaptic_degree"]:
        raise ValueError("matched nominal projection accounting mismatch")
    exact_large = d3._projection_state_bytes(
        feature_dim=config["feature_dim"], expand_dim=large["expand_dim"],
        nonzeros=state["nominal_large_projection_entries"],
        num_classes=config["num_classes"],
        gram_or_factor_bytes=large["expand_dim"] ** 2 * 4,
    )
    exact_matched = d3._projection_state_bytes(
        feature_dim=config["feature_dim"], expand_dim=matched["expand_dim"],
        nonzeros=state["nominal_matched_projection_entries"],
        num_classes=config["num_classes"],
        gram_or_factor_bytes=matched["expand_dim"] ** 2 * 4,
    )
    srq = projected_srq_state_bytes(
        feature_dim=config["feature_dim"], expand_dim=large["expand_dim"],
        synaptic_degree=large["synaptic_degree"], num_classes=config["num_classes"],
        block_size=config["storage"]["block_size"],
        group_size=config["storage"]["group_size"],
    )["compressed_total_bytes"]
    if (
        exact_large != state["nominal_exact_large_bytes"]
        or exact_matched != state["nominal_exact_matched_bytes"]
        or srq != state["nominal_srq_large_bytes"]
    ):
        raise ValueError("nominal state bytes mismatch")
    reference = config["reference_d3"]
    if set(reference) != REFERENCE_KEYS or reference["status"] != "STOP_SRQ_FLY_D3":
        raise ValueError("invalid D3 reference")
    if float(reference["selected_fly_ridge_lambda"]) != float(config["fixed_fly_ridge_lambda"]):
        raise ValueError("D4 did not transfer the locked D3 FLY lambda")
    gates = config["gates"]
    if set(gates) != GATE_KEYS:
        raise ValueError("gate keys mismatch")
    if (
        gates["maximum_search_candidate_solver_relative_residual"] <= 0
        or gates["maximum_outer_solver_relative_residual"] <= 0
        or gates["maximum_search_candidate_solver_relative_residual"] < gates["maximum_outer_solver_relative_residual"]
        or gates["maximum_average_gap_to_exact_large_pp"] < 0
        or gates["maximum_final_gap_to_exact_large_pp"] < 0
        or not 0 <= gates["minimum_prediction_agreement"] <= 1
        or not 0 < gates["maximum_state_fraction_of_exact_large"] <= 1
        or not 0 <= gates["maximum_state_mismatch_fraction"] < 1
        or gates["minimum_mean_average_gain_over_state_matched_fly_pp"] < 0
        or gates["minimum_median_average_gain_over_state_matched_fly_pp"] < 0
        or gates["minimum_mean_final_gain_over_state_matched_fly_pp"] < 0
        or not 0 < gates["minimum_seed_win_fraction"] <= 1
        or gates["maximum_worst_seed_average_loss_pp"] < 0
    ):
        raise ValueError("invalid gates")
    return config


def _verify_d3_reference(path: Path, config: dict, train_sha256: str) -> dict:
    reference = config["reference_d3"]
    if _sha256_file(path) != reference["result_sha256"]:
        raise ValueError("D3 result SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != reference["status"]
        or payload.get("uses_test_set") is not False
        or payload.get("held_out_test_authorized") is not False
        or payload.get("uses_outer_validation_for_selection") is not False
    ):
        raise ValueError("D3 reference contract mismatch")
    provenance = payload.get("provenance", {})
    expected_provenance = {
        "config_sha256": reference["config_sha256"],
        "runner_git_commit": reference["runner_git_commit"],
        "train_sha256": reference["train_sha256"],
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise ValueError(f"D3 provenance mismatch for {key}")
    if provenance.get("runner_git_dirty") is not False or train_sha256 != reference["train_sha256"]:
        raise ValueError("runtime cache differs from clean D3 reference")
    failed = sorted(key for key, value in payload.get("gates", {}).items() if not value)
    if failed != ["numerical_stability"]:
        raise ValueError("D3 did not fail solely on the recorded numerical gate")
    results = {item.get("method"): item for item in payload.get("results", [])}
    checks = {
        "srq_average_accuracy": results.get("srq_int8", {}).get("validation_average_accuracy"),
        "srq_final_accuracy": results.get("srq_int8", {}).get("stage_accuracy", [None])[-1],
        "srq_persistent_state_bytes": results.get("srq_int8", {}).get("persistent_state_bytes"),
        "matched_average_accuracy": results.get("exact_fly_4518", {}).get("validation_average_accuracy"),
        "matched_final_accuracy": results.get("exact_fly_4518", {}).get("stage_accuracy", [None])[-1],
        "matched_persistent_state_bytes": results.get("exact_fly_4518", {}).get("persistent_state_bytes"),
    }
    for key, observed in checks.items():
        if observed is None or abs(float(observed) - float(reference[key])) > 1e-10:
            raise ValueError(f"D3 metric mismatch for {key}")
    if float(payload.get("selection", {}).get("exact_fly_10000_and_srq_10000", -1)) != float(config["fixed_fly_ridge_lambda"]):
        raise ValueError("D3 selected FLY lambda mismatch")
    return payload


def _parts(config: dict, train: dict, seed: int):
    order = random.Random(seed).sample(list(range(config["num_classes"])), config["num_classes"])
    tasks = split(train["labels"], order, config["num_tasks"])
    outer_fit, outer_validation = train_validation_indices(
        train["labels"], tasks, seed, config["outer_validation_fraction"]
    )
    inner_fit, inner_validation = train_validation_indices(
        train["labels"], outer_fit, seed, config["inner_validation_fraction"]
    )
    d3.d21._validate_nested_parts(
        outer_fit, outer_validation, inner_fit, inner_validation
    )
    return order, outer_fit, outer_validation, inner_fit, inner_validation


def _mean_std_ci(values: list[float]) -> dict:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("summary requires finite values")
    parsed = list(map(float, values))
    mean = statistics.fmean(parsed)
    std = statistics.stdev(parsed) if len(parsed) > 1 else 0.0
    half = 2.776 * std / math.sqrt(len(parsed)) if len(parsed) == 5 else None
    return {
        "values": parsed, "mean": mean, "sample_std": std,
        "ci95_low": None if half is None else mean - half,
        "ci95_high": None if half is None else mean + half,
    }


def _raw_candidate_name(seed: int, index: int, ridge_lambda: float) -> str:
    value = f"{ridge_lambda:.0e}".replace("+", "")
    return f"inner_raw_seed_{seed}_{index:02d}_lambda_{value}"


def _evaluate_inner_raw_quiet(**kwargs) -> dict:
    # Forty tiny raw-Ridge candidates would otherwise emit 800 TASK lines.
    # D4 exposes one bounded START/DONE line per candidate instead.
    with contextlib.redirect_stdout(io.StringIO()):
        return d3._evaluate_raw(**kwargs)


def _runtime_state(config: dict, large_projection: torch.Tensor, matched_projection: torch.Tensor) -> dict:
    expected = config["expected_state"]
    large_nnz = int(large_projection.values().numel())
    matched_nnz = int(matched_projection.values().numel())
    missing_large = expected["nominal_large_projection_entries"] - large_nnz
    missing_matched = expected["nominal_matched_projection_entries"] - matched_nnz
    if (
        missing_large < 0 or missing_matched < 0
        or max(missing_large, missing_matched) > expected["maximum_missing_projection_entries"]
    ):
        raise ValueError("seeded projection stored-entry count is outside preregistered bounds")
    return {
        "large_projection_nonzeros": large_nnz,
        "matched_projection_nonzeros": matched_nnz,
        "missing_large_projection_entries": missing_large,
        "missing_matched_projection_entries": missing_matched,
        "exact_large_bytes": expected["nominal_exact_large_bytes"] - 12 * missing_large,
        "srq_large_bytes": expected["nominal_srq_large_bytes"] - 12 * missing_large,
        "exact_matched_bytes": expected["nominal_exact_matched_bytes"] - 12 * missing_matched,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    code_cache_root = Path(args.code_cache_root).resolve()
    config = _read_config(config_path)
    if args.require_test_hidden and (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("held-out test.pt is visible")
    dataset_audit, dataset_audit_sha = d3._validate_dataset_audit(
        Path(args.dataset_audit).resolve(), config
    )
    train, metadata = d3._validate_feature_cache(config, feature_cache_dir)
    train_sha = _sha256_file(feature_cache_dir / "train.pt")
    d3_reference = _verify_d3_reference(
        Path(args.d3_result).resolve(), config, train_sha
    )
    git = _git_provenance()
    split_by_seed = {
        seed: _parts(config, train, seed) for seed in config["seeds"]
    }
    base_context = {
        "config_sha256": _sha256_file(config_path),
        "d3_result_sha256": _sha256_file(Path(args.d3_result).resolve()),
        "dataset_audit_sha256": dataset_audit_sha,
        "dataset_identity_sha256": dataset_audit["dataset_identity_sha256"],
        "train_sha256": train_sha,
        "runner_git_commit": git["git_commit"], "runner_git_dirty": git["git_dirty"],
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "d3_runner_source_sha256": _sha256_file(ROOT / "tools/srq_fly_d3_cub.py"),
        "d1_runner_source_sha256": _sha256_file(ROOT / "tools/srq_fly_d1.py"),
        "learner_source_sha256": _sha256_file(ROOT / "methods/srq_fly/learner.py"),
        "storage_source_sha256": _sha256_file(ROOT / "methods/srq_fly/storage.py"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    raw_candidates = []
    for seed in config["seeds"]:
        order, outer_fit, outer_val, inner_fit, inner_val = split_by_seed[seed]
        raw_context = {
            **base_context, "seed": seed, "class_order": order,
            "outer_training_indices_sha256": _sequence_sha256(outer_fit),
            "outer_validation_indices_sha256": _sequence_sha256(outer_val),
            "inner_fit_indices_sha256": _sequence_sha256(inner_fit),
            "inner_validation_indices_sha256": _sequence_sha256(inner_val),
        }
        raw_context_sha = _sha256_bytes(json.dumps(raw_context, sort_keys=True).encode())
        for index, ridge_lambda in enumerate(map(float, config["raw_selection_lambdas"])):
            name = _raw_candidate_name(seed, index, ridge_lambda)
            path = _unit_path(output_dir, name)
            result = _load_unit(path, raw_context_sha)
            if result is None:
                print(f"RAW INNER START seed={seed} lambda={ridge_lambda:g}", flush=True)
                result = _save_unit(path, raw_context_sha, _evaluate_inner_raw_quiet(
                    name=name, ridge_lambda=ridge_lambda, config=config, train=train,
                    training_parts=inner_fit, validation_parts=inner_val, device=device,
                ))
                print(f"RAW INNER DONE seed={seed} lambda={ridge_lambda:g} AA={result['validation_average_accuracy']:.6f}", flush=True)
            d3._validate_result(
                result, name=name, ridge_lambda=ridge_lambda,
                num_tasks=config["num_tasks"],
            )
            raw_candidates.append({"seed": seed, **result})
    aggregate_candidates = []
    for ridge_lambda in map(float, config["raw_selection_lambdas"]):
        values = [
            item["validation_average_accuracy"] for item in raw_candidates
            if float(item["ridge_lambda"]) == ridge_lambda
        ]
        if len(values) != len(config["seeds"]):
            raise RuntimeError("raw candidate seed coverage mismatch")
        aggregate_candidates.append({
            "ridge_lambda": ridge_lambda,
            "mean_inner_validation_accuracy": statistics.fmean(values),
            "per_seed_accuracy": values,
        })
    raw_selected = min(
        aggregate_candidates,
        key=lambda item: (-item["mean_inner_validation_accuracy"], item["ridge_lambda"]),
    )
    selection_payload = {
        "schema_version": 1,
        "protocol": "one global raw-Ridge lambda selected by mean inner AA across five fresh seeds",
        "uses_outer_validation_for_selection": False, "uses_test_set": False,
        "fixed_fly_ridge_lambda": config["fixed_fly_ridge_lambda"],
        "selected_raw_ridge_lambda": raw_selected["ridge_lambda"],
        "tie_break": "maximum mean inner AA, then smaller lambda",
        "aggregate_candidates": aggregate_candidates,
        "per_seed_candidates": raw_candidates,
    }
    selection_path = output_dir / "raw_lambda_selection.json"
    d3._atomic_json(selection_path, selection_payload)
    print(f"LOCKED raw_lambda={raw_selected['ridge_lambda']:g}; fly_lambda={config['fixed_fly_ridge_lambda']:g}", flush=True)

    seed_results = []
    for seed in config["seeds"]:
        print(f"SEED START {seed}", flush=True)
        order, outer_fit, outer_val, inner_fit, inner_val = split_by_seed[seed]
        seed_cache = code_cache_root / f"seed_{seed}"
        seed_view = {**config, "seed": seed}
        print(f"CACHE START seed={seed} FLY-10000", flush=True)
        large = _prepare_code_cache(
            train=train, train_sha256=train_sha, cache_dir=seed_cache / "large",
            config=d3._cache_config(seed_view, config["large_representation"]),
            device=args.device,
        )
        print(f"CACHE START seed={seed} FLY-4518", flush=True)
        matched = _prepare_code_cache(
            train=train, train_sha256=train_sha, cache_dir=seed_cache / "matched",
            config=d3._cache_config(seed_view, config["matched_representation"]),
            device=args.device,
        )
        prefix = d3._verify_projection_prefix(large[3], matched[3])
        runtime_state = _runtime_state(config, large[3], matched[3])
        context = {
            **base_context, "seed": seed, "class_order": order,
            "raw_lambda_selection_sha256": _sha256_file(selection_path),
            "large_code_identity": large[2]["identity_sha256"],
            "matched_code_identity": matched[2]["identity_sha256"],
            "large_projection_sha256": _tensor_content_sha256(large[3]),
            "matched_projection_sha256": _tensor_content_sha256(matched[3]),
            "outer_training_indices_sha256": _sequence_sha256(outer_fit),
            "outer_validation_indices_sha256": _sequence_sha256(outer_val),
            "inner_fit_indices_sha256": _sequence_sha256(inner_fit),
            "inner_validation_indices_sha256": _sequence_sha256(inner_val),
        }
        context_sha = _sha256_bytes(json.dumps(context, sort_keys=True).encode())
        fly_lambda = float(config["fixed_fly_ridge_lambda"])
        paired_name = f"outer_seed_{seed}_paired_exact_srq_10000"
        paired_path = _unit_path(output_dir, paired_name)
        paired = _load_unit(paired_path, context_sha)
        if paired is None:
            print(f"OUTER START seed={seed} paired exact/SRQ FLY-10000", flush=True)
            paired = _save_unit(paired_path, context_sha, d1._evaluate_paired_exact_srq(
                config=d3._paired_config(seed_view, fly_lambda), train=train,
                code_indices=large[0], code_values=large[1], projection=large[3],
                training_parts=outer_fit, validation_parts=outer_val, device=device,
            ))
            print(f"OUTER DONE seed={seed} paired exact/SRQ FLY-10000", flush=True)
        exact, srq = paired["exact"], paired["srq"]
        d3._validate_result(
            exact, name="exact_fly_10000", ridge_lambda=fly_lambda,
            num_tasks=config["num_tasks"], expected_state_bytes=runtime_state["exact_large_bytes"],
        )
        d3._validate_result(
            srq, name="srq_int8", ridge_lambda=fly_lambda,
            num_tasks=config["num_tasks"], expected_state_bytes=runtime_state["srq_large_bytes"],
        )

        matched_name = f"outer_seed_{seed}_exact_fly_4518"
        matched_path = _unit_path(output_dir, matched_name)
        matched_result = _load_unit(matched_path, context_sha)
        if matched_result is None:
            print(f"OUTER START seed={seed} exact FLY-4518", flush=True)
            matched_result = _save_unit(matched_path, context_sha, d3._evaluate_exact(
                name=matched_name, ridge_lambda=fly_lambda, config=config,
                representation=config["matched_representation"], train=train,
                code_indices=matched[0], code_values=matched[1], projection=matched[3],
                training_parts=outer_fit, validation_parts=outer_val, device=device,
            ))
            print(f"OUTER DONE seed={seed} exact FLY-4518 AA={matched_result['validation_average_accuracy']:.6f}", flush=True)
        d3._validate_result(
            matched_result, name=matched_name, ridge_lambda=fly_lambda,
            num_tasks=config["num_tasks"], expected_state_bytes=runtime_state["exact_matched_bytes"],
        )

        raw_name = f"outer_seed_{seed}_raw_ridge"
        raw_path = _unit_path(output_dir, raw_name)
        raw = _load_unit(raw_path, context_sha)
        if raw is None:
            print(f"OUTER START seed={seed} raw Ridge", flush=True)
            raw = _save_unit(raw_path, context_sha, d3._evaluate_raw(
                name=raw_name, ridge_lambda=raw_selected["ridge_lambda"],
                config=config, train=train, training_parts=outer_fit,
                validation_parts=outer_val, device=device,
            ))
            print(f"OUTER DONE seed={seed} raw Ridge AA={raw['validation_average_accuracy']:.6f}", flush=True)
        d3._validate_result(
            raw, name=raw_name, ridge_lambda=raw_selected["ridge_lambda"],
            num_tasks=config["num_tasks"],
        )
        comparison = {
            "srq_average_difference_from_exact_large_pp": srq["validation_average_accuracy"] - exact["validation_average_accuracy"],
            "srq_final_difference_from_exact_large_pp": srq["stage_accuracy"][-1] - exact["stage_accuracy"][-1],
            "srq_average_gain_over_state_matched_fly_pp": srq["validation_average_accuracy"] - matched_result["validation_average_accuracy"],
            "srq_final_gain_over_state_matched_fly_pp": srq["stage_accuracy"][-1] - matched_result["stage_accuracy"][-1],
            "srq_average_gain_over_raw_ridge_pp": srq["validation_average_accuracy"] - raw["validation_average_accuracy"],
            "state_mismatch_fraction": abs(srq["persistent_state_bytes"] - matched_result["persistent_state_bytes"]) / srq["persistent_state_bytes"],
            "srq_state_fraction_of_exact_large": srq["persistent_state_bytes"] / exact["persistent_state_bytes"],
            "minimum_prediction_agreement": paired["minimum_prediction_agreement"],
        }
        seed_results.append({
            "seed": seed, "class_order": order, "provenance": context,
            "projection_prefix": prefix, "runtime_state": runtime_state,
            "exact_fly_10000": exact, "srq_fly_10000": srq,
            "exact_fly_4518": matched_result, "raw_ridge": raw,
            "paired_diagnostics": paired["paired_diagnostics"],
            "comparison": comparison,
        })
        print(f"SEED DONE {seed} gain_matched={comparison['srq_average_gain_over_state_matched_fly_pp']:+.4f}pp", flush=True)

    average_gains = [item["comparison"]["srq_average_gain_over_state_matched_fly_pp"] for item in seed_results]
    final_gains = [item["comparison"]["srq_final_gain_over_state_matched_fly_pp"] for item in seed_results]
    exact_differences = [item["comparison"]["srq_average_difference_from_exact_large_pp"] for item in seed_results]
    raw_gains = [item["comparison"]["srq_average_gain_over_raw_ridge_pp"] for item in seed_results]
    srq_average = [item["srq_fly_10000"]["validation_average_accuracy"] for item in seed_results]
    matched_average = [item["exact_fly_4518"]["validation_average_accuracy"] for item in seed_results]
    raw_average = [item["raw_ridge"]["validation_average_accuracy"] for item in seed_results]
    summaries = {
        "srq_average_accuracy": _mean_std_ci(srq_average),
        "matched_average_accuracy": _mean_std_ci(matched_average),
        "raw_average_accuracy": _mean_std_ci(raw_average),
        "srq_average_gain_over_matched_pp": _mean_std_ci(average_gains),
        "srq_final_gain_over_matched_pp": _mean_std_ci(final_gains),
        "srq_average_difference_from_exact_large_pp": _mean_std_ci(exact_differences),
        "srq_average_gain_over_raw_pp": _mean_std_ci(raw_gains),
    }
    gates_config = config["gates"]
    search_max_residual = max(float(item["maximum_solver_relative_residual"]) for item in raw_candidates)
    outer_results = [
        result for item in seed_results
        for result in (item["exact_fly_10000"], item["srq_fly_10000"], item["exact_fly_4518"], item["raw_ridge"])
    ]
    outer_max_residual = max(float(item["maximum_solver_relative_residual"]) for item in outer_results)
    win_fraction = sum(gain > 0 for gain in average_gains) / len(average_gains)
    median_gain = statistics.median(average_gains)
    raw_grid = list(map(float, config["raw_selection_lambdas"]))
    gates = {
        "d3_reference_verified": d3_reference["status"] == "STOP_SRQ_FLY_D3",
        "all_five_fresh_seeds_complete": len(seed_results) == 5,
        "heldout_test_remained_hidden": not (feature_cache_dir / "test.pt").exists(),
        "outer_validation_not_used_for_selection": selection_payload["uses_outer_validation_for_selection"] is False,
        "raw_lambda_is_bracketed": raw_selected["ridge_lambda"] not in {raw_grid[0], raw_grid[-1]},
        "search_candidate_numerical_stability": search_max_residual <= gates_config["maximum_search_candidate_solver_relative_residual"],
        "outer_numerical_stability": outer_max_residual <= gates_config["maximum_outer_solver_relative_residual"],
        "projection_prefix_verified_every_seed": all(item["projection_prefix"]["verified"] for item in seed_results),
        "runtime_state_accounting_matches_every_seed": all(
            item["exact_fly_10000"]["persistent_state_bytes"] == item["runtime_state"]["exact_large_bytes"]
            and item["srq_fly_10000"]["persistent_state_bytes"] == item["runtime_state"]["srq_large_bytes"]
            and item["exact_fly_4518"]["persistent_state_bytes"] == item["runtime_state"]["exact_matched_bytes"]
            for item in seed_results
        ),
        "srq_tracks_exact_large_average_every_seed": all(
            -difference <= gates_config["maximum_average_gap_to_exact_large_pp"]
            for difference in exact_differences
        ),
        "srq_tracks_exact_large_final_every_seed": all(
            -(item["comparison"]["srq_final_difference_from_exact_large_pp"])
            <= gates_config["maximum_final_gap_to_exact_large_pp"]
            for item in seed_results
        ),
        "prediction_agreement_every_seed": all(
            item["comparison"]["minimum_prediction_agreement"] >= gates_config["minimum_prediction_agreement"]
            for item in seed_results
        ),
        "compressed_state_fraction_every_seed": all(
            item["comparison"]["srq_state_fraction_of_exact_large"] <= gates_config["maximum_state_fraction_of_exact_large"]
            for item in seed_results
        ),
        "state_budget_matched_every_seed": all(
            item["comparison"]["state_mismatch_fraction"] <= gates_config["maximum_state_mismatch_fraction"]
            for item in seed_results
        ),
        "mean_average_gain_over_matched": statistics.fmean(average_gains) >= gates_config["minimum_mean_average_gain_over_state_matched_fly_pp"],
        "median_average_gain_over_matched": median_gain >= gates_config["minimum_median_average_gain_over_state_matched_fly_pp"],
        "mean_final_gain_over_matched": statistics.fmean(final_gains) >= gates_config["minimum_mean_final_gain_over_state_matched_fly_pp"],
        "seed_win_fraction": win_fraction >= gates_config["minimum_seed_win_fraction"],
        "worst_seed_loss_within_gate": min(average_gains) >= -gates_config["maximum_worst_seed_average_loss_pp"],
        "raw_ridge_does_not_pareto_dominate_mean": statistics.fmean(raw_average) < statistics.fmean(srq_average),
    }
    decision = "PASS_REVIEW_D4" if all(gates.values()) else "STOP_SRQ_FLY_D4"
    payload = {
        "schema_version": 1, "study_id": config["study_id"], "status": decision,
        "uses_test_set": False, "held_out_test_authorized": False,
        "uses_outer_validation_for_selection": False,
        "provenance": {
            **base_context, "raw_lambda_selection_sha256": _sha256_file(selection_path),
        },
        "source_feature_metadata": metadata,
        "d3_reference": {
            "result_sha256": config["reference_d3"]["result_sha256"],
            "status": d3_reference["status"],
        },
        "fixed_fly_ridge_lambda": config["fixed_fly_ridge_lambda"],
        "selected_raw_ridge_lambda": raw_selected["ridge_lambda"],
        "seed_results": seed_results, "summaries": summaries,
        "diagnostics": {
            "maximum_raw_search_candidate_residual": search_max_residual,
            "maximum_outer_residual": outer_max_residual,
            "seed_win_fraction": win_fraction,
            "median_average_gain_over_matched_pp": median_gain,
            "raw_selected_at_grid_boundary": raw_selected["ridge_lambda"] in {raw_grid[0], raw_grid[-1]},
        },
        "gates": gates,
    }
    d3._atomic_json(output_dir / "d4_results.json", payload)
    print(f"SRQ-FLY D4 {decision}", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--d3-result", required=True)
    parser.add_argument("--code-cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
