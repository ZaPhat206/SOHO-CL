"""Fresh train-only Phase-1B allocation study for MARS-SOHO.

Phase 1B does not retune the replay distribution.  It locks the Phase-1
inner-selected Ridge and reconstruction settings, uses fresh train-only outer
splits, and tests whether continuous Top-K turnover or sufficient-statistic
variance can allocate the same pseudo budget better than uniform allocation.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from copy import deepcopy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import mars_soho_phase1 as base


METHODS = (
    "exact_replay_oracle",
    "heterogeneous_spherical",
    "turnover_aware",
    "shuffled_turnover",
    "statistic_variance_aware",
    "shuffled_statistic_variance",
)


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "study_id", "phase1_artifact", "backbone",
        "phase1b", "datasets",
    }
    if not required.issubset(payload):
        raise ValueError(f"config missing fields: {sorted(required - set(payload))}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported MARS-SOHO Phase-1B schema")
    phase, artifact = payload["phase1b"], payload["phase1_artifact"]
    locked = phase["reconstruction"]
    if (
        float(phase["ridge_lambda"]) != float(artifact["locked_ridge_lambda"])
        or int(locked["covariance_rank"])
        != int(artifact["locked_covariance_rank"])
        or float(locked["shrinkage"])
        != float(artifact["locked_shrinkage"])
    ):
        raise ValueError("Phase-1B settings do not match the locked Phase-1 selection")
    return payload


def _runtime_config(config: dict) -> dict:
    """Adapt the isolated Phase-1B section to the shared evaluator contract."""
    runtime = deepcopy(config)
    runtime["phase1"] = deepcopy(config["phase1b"])
    return runtime


def _fraction(flags: list[bool]) -> float:
    return statistics.fmean(map(float, flags)) if flags else 0.0


def _allocation_diagnostics(outer: dict, phase: dict) -> dict:
    proposed = outer["statistic_variance_aware"]
    shuffled = outer["shuffled_statistic_variance"]
    noncollapsed, nonuniform, distinct = [], [], []
    spreads = []
    for proposed_result, shuffled_result in zip(proposed, shuffled):
        for proposed_task, shuffled_task in zip(
            proposed_result["task_diagnostics"],
            shuffled_result["task_diagnostics"],
        ):
            if proposed_task.get("old_class_count", 0) < 2:
                continue
            values = list(
                proposed_task.get("pilot_risks", {})
                .get("statistic_variance", {})
                .values()
            )
            if not values:
                noncollapsed.append(False)
                nonuniform.append(False)
                distinct.append(False)
                spreads.append(0.0)
                continue
            spread = max(values) - min(values)
            spreads.append(spread)
            noncollapsed.append(spread > phase["gates"]["minimum_risk_spread"])
            proposed_allocation = proposed_task.get("pseudo_allocation", {})
            shuffled_allocation = shuffled_task.get("pseudo_allocation", {})
            nonuniform.append(len(set(proposed_allocation.values())) > 1)
            distinct.append(proposed_allocation != shuffled_allocation)
    return {
        "risk_spreads": spreads,
        "noncollapsed_fraction": _fraction(noncollapsed),
        "nonuniform_allocation_fraction": _fraction(nonuniform),
        "distinct_from_shuffled_fraction": _fraction(distinct),
        "evaluated_task_count": len(spreads),
    }


def run(
    *,
    config_path: Path,
    dataset_key: str,
    feature_cache_dir: Path,
    output_root: Path,
    device: str,
) -> dict:
    config = _read_config(config_path)
    if dataset_key not in config["datasets"]:
        raise ValueError(f"unknown dataset key: {dataset_key}")
    cached = base._validate_train_cache(feature_cache_dir, config, dataset_key)
    stream = {"features": cached["features"], "labels": cached["labels"]}
    dataset, phase = config["datasets"][dataset_key], config["phase1b"]
    runtime = _runtime_config(config)
    output_dir = output_root / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "config_sha256": base._sha256(config_path),
        "runner_sha256": base._sha256(Path(__file__).resolve()),
        "train_sha256": base._sha256(feature_cache_dir / "train.pt"),
        "phase1_artifact": config["phase1_artifact"],
        "dataset_key": dataset_key,
        "device": device,
        "environment": base._environment(device),
        "method_sha256": {
            relative: base._sha256(REPOSITORY_ROOT / relative)
            for relative in (
                "methods/mars_soho/statistics.py",
                "methods/mars_soho/geometry.py",
                "methods/mars_soho/reconstruction.py",
                "methods/mars_soho/learner.py",
            )
        },
    }
    replicates = []
    for index, replicate in enumerate(phase["validation_replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        parts = base._nested_parts(
            stream["labels"],
            class_order,
            dataset["num_tasks"],
            phase["split_seed"],
            phase["outer_validation_fraction"],
            phase["inner_validation_fraction"],
        )
        replicates.append({
            "index": index,
            "replicate": replicate,
            "class_order": class_order,
            "fit_parts": parts[2],
            "validation_parts": parts[3],
        })
    candidate = dict(phase["reconstruction"])
    ridge_lambda = float(phase["ridge_lambda"])
    outer = {method: [] for method in METHODS}
    for method in METHODS:
        for item in replicates:
            method_candidate = None if method == "exact_replay_oracle" else candidate
            context = {
                **source,
                "stage": "fresh_train_only_validation",
                "method": method,
                "candidate": method_candidate,
                "ridge_lambda": ridge_lambda,
                **item["replicate"],
            }
            result = base._unit(
                output_dir / "outer" / f"{method}_r{item['index']}.json",
                context,
                lambda item=item, method=method, method_candidate=method_candidate:
                base._evaluate(
                    method=method,
                    config=runtime,
                    dataset_key=dataset_key,
                    stream=stream,
                    fit_parts=item["fit_parts"],
                    validation_parts=item["validation_parts"],
                    projection_seed=item["replicate"]["projection_seed"],
                    ridge_lambda=ridge_lambda,
                    candidate=method_candidate,
                    device=device,
                ),
            )
            outer[method].append(result)
    outer_aia = {
        method: base._mean_aia(results) for method, results in outer.items()
    }
    proposed = outer_aia["statistic_variance_aware"]
    allocation = _allocation_diagnostics(outer, phase)
    gate_config = phase["gates"]
    gates = {
        "variance_gap_to_oracle_at_most_pp": {
            "threshold": gate_config["max_oracle_gap_pp"],
            "observed": outer_aia["exact_replay_oracle"] - proposed,
            "pass": outer_aia["exact_replay_oracle"] - proposed
            <= gate_config["max_oracle_gap_pp"],
        },
        "variance_gain_over_uniform_at_least_pp": {
            "threshold": gate_config["min_uniform_gain_pp"],
            "observed": proposed - outer_aia["heterogeneous_spherical"],
            "pass": proposed - outer_aia["heterogeneous_spherical"]
            >= gate_config["min_uniform_gain_pp"],
        },
        "variance_gain_over_shuffled_at_least_pp": {
            "threshold": gate_config["min_shuffled_gain_pp"],
            "observed": proposed - outer_aia["shuffled_statistic_variance"],
            "pass": proposed - outer_aia["shuffled_statistic_variance"]
            >= gate_config["min_shuffled_gain_pp"],
        },
        "continuous_risk_noncollapsed": {
            "threshold": gate_config["minimum_noncollapsed_fraction"],
            "observed": allocation["noncollapsed_fraction"],
            "pass": allocation["noncollapsed_fraction"]
            >= gate_config["minimum_noncollapsed_fraction"],
        },
        "allocation_is_nonuniform": {
            "threshold": gate_config["minimum_noncollapsed_fraction"],
            "observed": allocation["nonuniform_allocation_fraction"],
            "pass": allocation["nonuniform_allocation_fraction"]
            >= gate_config["minimum_noncollapsed_fraction"],
        },
        "shuffled_control_changes_allocation": {
            "threshold": gate_config["minimum_distinct_allocation_fraction"],
            "observed": allocation["distinct_from_shuffled_fraction"],
            "pass": allocation["distinct_from_shuffled_fraction"]
            >= gate_config["minimum_distinct_allocation_fraction"],
        },
        "test_remained_hidden": {
            "pass": not (feature_cache_dir / "test.pt").exists()
        },
    }
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "phase1b_pass"
        if all(value["pass"] for value in gates.values())
        else "phase1b_failed",
        "uses_test_set": False,
        "source": source,
        "dataset_key": dataset_key,
        "locked_ridge_lambda": ridge_lambda,
        "locked_reconstruction": candidate,
        "replicates": [
            {
                "replicate": item["replicate"],
                "class_order": item["class_order"],
                "class_order_sha256": base._json_hash(item["class_order"]),
            }
            for item in replicates
        ],
        "outer_validation": outer,
        "outer_mean_aia": outer_aia,
        "allocation_diagnostics": allocation,
        "gates": gates,
    }
    base._atomic_json(output_dir / "phase1b_results.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "outer_mean_aia": outer_aia,
        "allocation_diagnostics": allocation,
        "gates": gates,
    }, indent=2), flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--feature-cache-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(
        config_path=args.config.resolve(),
        dataset_key=args.dataset_key,
        feature_cache_dir=args.feature_cache_dir.resolve(),
        output_root=args.output_root.resolve(),
        device=args.device,
    )


if __name__ == "__main__":
    main()
