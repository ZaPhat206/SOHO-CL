"""Locked nested train-only gate for WBT-SOHO Phase 1D."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.mars_soho.learner import MARSExactReplayOracle
from methods.wbt_soho.learner import WBTSOHOLearner
from methods.wbt_soho.transport import topk_gap
from tools import mars_soho_phase1 as base


METHODS = (
    "exact_replay_oracle",
    "tangent_gaussian",
    "mean_shift_empirical",
    "covariance_transport",
    "wta_boundary_transport",
    "shuffled_enemy_boundary_transport",
)


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "study_id", "phase1c_artifact", "backbone",
        "phase1d", "datasets",
    }
    if not required.issubset(payload):
        raise ValueError(f"config missing fields: {sorted(required - set(payload))}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported WBT-SOHO Phase-1D schema")
    if payload["phase1c_artifact"].get("status") != "phase1c_failed":
        raise ValueError("Phase 1D requires the locked Phase-1C negative artifact")
    return payload


def _base_kwargs(config: dict, dataset_key: str, seed: int, device: str) -> dict:
    phase = config["phase1d"]
    selected = config["datasets"][dataset_key]["locked_soho"]
    return {
        "feature_dim": config["backbone"]["feature_dim"],
        "expand_dim": phase["expand_dim"],
        "density": selected["density"],
        "olda_dim": phase["olda_dim"],
        "use_etf": selected["use_etf"],
        "coding_level": selected["coding_level"],
        "ridge_lambda": phase["ridge_lambda"],
        "seed": seed,
        "device": device,
        "dtype": torch.float32,
    }


def _candidate_grid(config: dict) -> list[dict]:
    search = config["phase1d"]["boundary_grid"]
    return [
        {
            "boundary_fraction": float(fraction),
            "boundary_strength": float(strength),
        }
        for fraction in search["boundary_fraction"]
        for strength in search["boundary_strength"]
    ]


def _evaluate(
    *,
    method: str,
    config: dict,
    dataset_key: str,
    stream: dict,
    fit_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    projection_seed: int,
    candidate: dict,
    reference_statistics: list[dict[str, torch.Tensor]],
    device: str,
) -> dict:
    kwargs = _base_kwargs(config, dataset_key, projection_seed, device)
    if method == "exact_replay_oracle":
        raise ValueError("exact oracle is evaluated through the shared reference path")
    if len(reference_statistics) != len(fit_parts):
        raise ValueError("oracle reference does not align with task stream")
    learner = WBTSOHOLearner(
        **kwargs,
        tangent_rank=config["phase1d"]["tangent_rank"],
        pseudo_per_class=config["phase1d"]["pseudo_per_class"],
        mode=method,
        boundary_fraction=candidate["boundary_fraction"],
        boundary_strength=candidate["boundary_strength"],
    )
    matrix: list[list[float]] = []
    low_margin_matrix: list[list[float]] = []
    gram_errors, cross_errors, task_diagnostics = [], [], []
    started = time.perf_counter()
    maximum_residual = 0.0
    for task, indices in enumerate(fit_parts):
        task_features = stream["features"][indices]
        task_labels = stream["labels"][indices]
        learner.update(task_features, task_labels)
        target_gram = reference_statistics[task]["gram"].to(device)
        target_cross = reference_statistics[task]["cross"].to(device)
        epsilon = torch.finfo(learner.G.dtype).eps
        gram_errors.append(float(
            (learner.G - target_gram).norm()
            .div(target_gram.norm().clamp_min(epsilon)).item()
        ))
        cross_errors.append(float(
            (learner.Q - target_cross).norm()
            .div(target_cross.norm().clamp_min(epsilon)).item()
        ))
        maximum_residual = max(
            maximum_residual,
            float(learner.diagnostics["solver_relative_residual"]),
        )
        row, low_margin_row = [], []
        for previous in range(task + 1):
            validation = validation_parts[previous]
            values = stream["features"][validation]
            labels = stream["labels"][validation].cpu()
            predictions = learner.predict(values)
            correct = predictions == labels
            row.append(float(correct.float().mean().item() * 100))
            expanded = learner.encoder.expanded(values.to(device))
            gaps = topk_gap(expanded, learner.encoder.k)
            low_count = max(1, round(len(gaps) * 0.25))
            low_indices = torch.argsort(gaps)[:low_count].cpu()
            low_margin_row.append(float(correct[low_indices].float().mean().item() * 100))
        matrix.append(row)
        low_margin_matrix.append(low_margin_row)
        task_diagnostics.append(dict(learner.diagnostics))
        print(
            f"TASK method={method} task={task+1}/{len(fit_parts)} "
            f"seen_AA={statistics.fmean(row):.4f} "
            f"Gerr={gram_errors[-1]:.5f} Qerr={cross_errors[-1]:.5f}",
            flush=True,
        )
    metrics = base._metrics(matrix)
    low_margin_metrics = base._metrics(low_margin_matrix)
    audit = base._state_audit(learner)
    dominance_values = [
        float(item["old_dominance_fraction"])
        for item in task_diagnostics
        if item.get("old_dominance_fraction") is not None
    ]
    gap_before_values = [
        float(item["mean_topk_gap_before"])
        for item in task_diagnostics
        if item.get("mean_topk_gap_before") is not None
    ]
    gap_after_values = [
        float(item["mean_topk_gap_after"])
        for item in task_diagnostics
        if item.get("mean_topk_gap_after") is not None
    ]
    if method != "exact_replay_oracle" and (
        audit["historical_feature_rows"]
        or audit["historical_label_rows"]
        or audit["sample_level_bytes"]
    ):
        raise AssertionError("WBT control retained sample-level state")
    return {
        "status": "complete",
        "method": method,
        "uses_test_set": False,
        "candidate": candidate,
        **metrics,
        "low_margin_average_incremental_accuracy": low_margin_metrics[
            "average_incremental_accuracy"
        ],
        "mean_gram_relative_error": statistics.fmean(gram_errors),
        "mean_cross_relative_error": statistics.fmean(cross_errors),
        "mean_combined_stat_error": statistics.fmean([
            (gram + cross) * 0.5
            for gram, cross in zip(gram_errors, cross_errors)
        ]),
        "solver_relative_residual_max": maximum_residual,
        "minimum_old_dominance_fraction": min(dominance_values)
        if dominance_values else 1.0,
        "mean_topk_gap_before": statistics.fmean(gap_before_values)
        if gap_before_values else None,
        "mean_topk_gap_after": statistics.fmean(gap_after_values)
        if gap_after_values else None,
        "persistent_state_bytes": learner.persistent_state_bytes(),
        "exemplar_free": learner.is_exemplar_free,
        "state_audit": audit,
        "task_diagnostics": task_diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _evaluate_oracle_reference(
    *,
    config: dict,
    dataset_key: str,
    stream: dict,
    fit_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    projection_seed: int,
    device: str,
) -> tuple[dict, list[dict[str, torch.Tensor]]]:
    learner = MARSExactReplayOracle(
        **_base_kwargs(config, dataset_key, projection_seed, device)
    )
    matrix, low_margin_matrix, statistics_by_task, diagnostics = [], [], [], []
    started = time.perf_counter()
    maximum_residual = 0.0
    for task, indices in enumerate(fit_parts):
        learner.update(stream["features"][indices], stream["labels"][indices])
        statistics_by_task.append({
            "gram": learner.G.detach().cpu().clone(),
            "cross": learner.Q.detach().cpu().clone(),
        })
        maximum_residual = max(
            maximum_residual,
            float(learner.diagnostics["solver_relative_residual"]),
        )
        row, low_margin_row = [], []
        for previous in range(task + 1):
            validation = validation_parts[previous]
            values = stream["features"][validation]
            labels = stream["labels"][validation].cpu()
            predictions = learner.predict(values)
            correct = predictions == labels
            row.append(float(correct.float().mean().item() * 100))
            gaps = topk_gap(learner.encoder.expanded(values.to(device)), learner.encoder.k)
            low_count = max(1, round(len(gaps) * 0.25))
            low_indices = torch.argsort(gaps)[:low_count].cpu()
            low_margin_row.append(float(correct[low_indices].float().mean().item() * 100))
        matrix.append(row)
        low_margin_matrix.append(low_margin_row)
        diagnostics.append(dict(learner.diagnostics))
        print(
            f"TASK method=exact_replay_oracle task={task+1}/{len(fit_parts)} "
            f"seen_AA={statistics.fmean(row):.4f}",
            flush=True,
        )
    metrics = base._metrics(matrix)
    low_margin_metrics = base._metrics(low_margin_matrix)
    result = {
        "status": "complete",
        "method": "exact_replay_oracle",
        "uses_test_set": False,
        "candidate": None,
        **metrics,
        "low_margin_average_incremental_accuracy": low_margin_metrics[
            "average_incremental_accuracy"
        ],
        "mean_gram_relative_error": 0.0,
        "mean_cross_relative_error": 0.0,
        "mean_combined_stat_error": 0.0,
        "solver_relative_residual_max": maximum_residual,
        "minimum_old_dominance_fraction": 1.0,
        "mean_topk_gap_before": None,
        "mean_topk_gap_after": None,
        "persistent_state_bytes": learner.persistent_state_bytes(),
        "exemplar_free": False,
        "state_audit": base._state_audit(learner),
        "task_diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result, statistics_by_task


def _reference_unit(
    *,
    result_path: Path,
    tensor_path: Path,
    context: dict,
    evaluator,
) -> tuple[dict, list[dict[str, torch.Tensor]]]:
    context_hash = base._json_hash(context)
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("context_sha256") != context_hash:
            raise RuntimeError(f"resume context mismatch: {result_path}")
        if not tensor_path.is_file() or payload.get("tensor_sha256") != base._sha256(tensor_path):
            raise RuntimeError(f"oracle reference tensor mismatch: {tensor_path}")
        tensors = torch.load(tensor_path, weights_only=True, map_location="cpu")
        print(f"RESTORED {result_path.stem}", flush=True)
        return payload["result"], tensors["statistics"]
    print(f"START {result_path.stem}", flush=True)
    result, statistics_by_task = evaluator()
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    torch.save({"statistics": statistics_by_task}, temporary)
    temporary.replace(tensor_path)
    base._atomic_json(result_path, {
        "context_sha256": context_hash,
        "tensor_sha256": base._sha256(tensor_path),
        "result": result,
    })
    print(f"DONE {result_path.stem}", flush=True)
    return result, statistics_by_task


def _mean(results: list[dict], key: str) -> float:
    return statistics.fmean(result[key] for result in results)


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
    phase = config["phase1d"]
    dataset = config["datasets"][dataset_key]
    output_dir = output_root / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "config_sha256": base._sha256(config_path),
        "runner_sha256": base._sha256(Path(__file__).resolve()),
        "train_sha256": base._sha256(feature_cache_dir / "train.pt"),
        "phase1c_artifact": config["phase1c_artifact"],
        "dataset_key": dataset_key,
        "device": device,
        "environment": base._environment(device),
        "method_sha256": {
            relative: base._sha256(REPOSITORY_ROOT / relative)
            for relative in (
                "methods/mars_soho/statistics.py",
                "methods/mars_soho/geometry.py",
                "methods/mars_soho/learner.py",
                "methods/mars_soho/tangent.py",
                "methods/wbt_soho/transport.py",
                "methods/wbt_soho/learner.py",
            )
        },
    }
    replicates = []
    for index, replicate in enumerate(phase["development_replicates"]):
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
            "parts": parts,
        })

    for item in replicates:
        common_context = {
            **source,
            **item["replicate"],
            "class_order_sha256": base._json_hash(item["class_order"]),
        }
        inner_result, inner_statistics = _reference_unit(
            result_path=output_dir / "reference" / f"inner_oracle_r{item['index']}.json",
            tensor_path=output_dir / "reference" / f"inner_oracle_r{item['index']}.pt",
            context={**common_context, "stage": "inner_oracle_reference"},
            evaluator=lambda item=item: _evaluate_oracle_reference(
                config=config,
                dataset_key=dataset_key,
                stream=stream,
                fit_parts=item["parts"][0],
                validation_parts=item["parts"][1],
                projection_seed=item["replicate"]["projection_seed"],
                device=device,
            ),
        )
        outer_result, outer_statistics = _reference_unit(
            result_path=output_dir / "reference" / f"outer_oracle_r{item['index']}.json",
            tensor_path=output_dir / "reference" / f"outer_oracle_r{item['index']}.pt",
            context={**common_context, "stage": "outer_oracle_reference"},
            evaluator=lambda item=item: _evaluate_oracle_reference(
                config=config,
                dataset_key=dataset_key,
                stream=stream,
                fit_parts=item["parts"][2],
                validation_parts=item["parts"][3],
                projection_seed=item["replicate"]["projection_seed"],
                device=device,
            ),
        )
        item["inner_oracle_result"] = inner_result
        item["inner_oracle_statistics"] = inner_statistics
        item["outer_oracle_result"] = outer_result
        item["outer_oracle_statistics"] = outer_statistics

    candidates = []
    for candidate_index, candidate in enumerate(_candidate_grid(config)):
        results = []
        for item in replicates:
            context = {
                **source,
                "stage": "inner_boundary_selection",
                "candidate": candidate,
                **item["replicate"],
                "class_order_sha256": base._json_hash(item["class_order"]),
            }
            result = base._unit(
                output_dir / "inner" / f"wbt_c{candidate_index}_r{item['index']}.json",
                context,
                lambda item=item, candidate=candidate: _evaluate(
                    method="wta_boundary_transport",
                    config=config,
                    dataset_key=dataset_key,
                    stream=stream,
                    fit_parts=item["parts"][0],
                    validation_parts=item["parts"][1],
                    projection_seed=item["replicate"]["projection_seed"],
                    candidate=candidate,
                    reference_statistics=item["inner_oracle_statistics"],
                    device=device,
                ),
            )
            results.append(result)
        candidates.append({
            "candidate": candidate,
            "mean_inner_aia": _mean(results, "average_incremental_accuracy"),
            "mean_inner_stat_error": _mean(results, "mean_combined_stat_error"),
            "replicates": results,
        })
    best_aia = max(item["mean_inner_aia"] for item in candidates)
    eligible = [
        item for item in candidates
        if item["mean_inner_aia"] >= best_aia - phase["near_tie_tolerance_pp"]
    ]
    selected = min(
        eligible,
        key=lambda item: (
            item["candidate"]["boundary_fraction"],
            item["candidate"]["boundary_strength"],
        ),
    )["candidate"]

    outer = {method: [] for method in METHODS}
    outer["exact_replay_oracle"] = [
        item["outer_oracle_result"] for item in replicates
    ]
    for method in METHODS[1:]:
        for item in replicates:
            context = {
                **source,
                "stage": "outer_validation",
                "method": method,
                "selected_candidate": selected,
                **item["replicate"],
                "class_order_sha256": base._json_hash(item["class_order"]),
            }
            result = base._unit(
                output_dir / "outer" / f"{method}_r{item['index']}.json",
                context,
                lambda item=item, method=method: _evaluate(
                    method=method,
                    config=config,
                    dataset_key=dataset_key,
                    stream=stream,
                    fit_parts=item["parts"][2],
                    validation_parts=item["parts"][3],
                    projection_seed=item["replicate"]["projection_seed"],
                    candidate=selected,
                    reference_statistics=item["outer_oracle_statistics"],
                    device=device,
                ),
            )
            outer[method].append(result)
    summary = {
        method: {
            key: _mean(results, key)
            for key in (
                "average_incremental_accuracy",
                "final_accuracy",
                "forgetting",
                "low_margin_average_incremental_accuracy",
                "mean_gram_relative_error",
                "mean_cross_relative_error",
                "mean_combined_stat_error",
                "minimum_old_dominance_fraction",
                "persistent_state_bytes",
                "elapsed_seconds",
            )
        }
        for method, results in outer.items()
    }
    proposal = summary["wta_boundary_transport"]
    tangent = summary["tangent_gaussian"]
    shuffled = summary["shuffled_enemy_boundary_transport"]
    oracle = summary["exact_replay_oracle"]
    initial_gap = oracle["average_incremental_accuracy"] - tangent[
        "average_incremental_accuracy"
    ]
    proposal_gain = proposal["average_incremental_accuracy"] - tangent[
        "average_incremental_accuracy"
    ]
    gap_closure = proposal_gain / initial_gap if initial_gap > 1e-12 else 1.0
    stat_reduction = (
        tangent["mean_combined_stat_error"] - proposal["mean_combined_stat_error"]
    ) / max(tangent["mean_combined_stat_error"], 1e-12)
    replicate_gains = [
        proposed["average_incremental_accuracy"] - baseline["average_incremental_accuracy"]
        for proposed, baseline in zip(
            outer["wta_boundary_transport"], outer["tangent_gaussian"]
        )
    ]
    gates_config = phase["gates"]
    gates = {
        "oracle_gap_closed_fraction": {
            "threshold": gates_config["minimum_oracle_gap_closed_fraction"],
            "observed": gap_closure,
            "pass": gap_closure
            >= gates_config["minimum_oracle_gap_closed_fraction"],
        },
        "relative_stat_error_reduction": {
            "threshold": gates_config["minimum_relative_stat_error_reduction"],
            "observed": stat_reduction,
            "pass": stat_reduction
            >= gates_config["minimum_relative_stat_error_reduction"],
        },
        "gain_over_shuffled_enemy_pp": {
            "threshold": gates_config["minimum_shuffled_gain_pp"],
            "observed": proposal["average_incremental_accuracy"]
            - shuffled["average_incremental_accuracy"],
            "pass": proposal["average_incremental_accuracy"]
            - shuffled["average_incremental_accuracy"]
            >= gates_config["minimum_shuffled_gain_pp"],
        },
        "gap_to_oracle_pp": {
            "threshold": gates_config["maximum_oracle_gap_pp"],
            "observed": oracle["average_incremental_accuracy"]
            - proposal["average_incremental_accuracy"],
            "pass": oracle["average_incremental_accuracy"]
            - proposal["average_incremental_accuracy"]
            <= gates_config["maximum_oracle_gap_pp"],
        },
        "positive_each_replicate": {
            "observed": replicate_gains,
            "pass": all(gain > 0 for gain in replicate_gains),
        },
        "numerical_stability": {
            "threshold": gates_config["maximum_solver_relative_residual"],
            "observed": max(
                result["solver_relative_residual_max"]
                for method in METHODS
                for result in outer[method]
            ),
            "pass": all(
                result["solver_relative_residual_max"]
                <= gates_config["maximum_solver_relative_residual"]
                for method in METHODS for result in outer[method]
            ),
        },
        "old_class_dominance": {
            "threshold": gates_config["minimum_old_dominance_fraction"],
            "observed": min(
                result["minimum_old_dominance_fraction"]
                for result in outer["wta_boundary_transport"]
            ),
            "pass": all(
                result["minimum_old_dominance_fraction"]
                >= gates_config["minimum_old_dominance_fraction"]
                for result in outer["wta_boundary_transport"]
            ),
        },
        "sample_free_state": {
            "pass": all(
                result["state_audit"]["sample_level_bytes"] == 0
                for method in METHODS if method != "exact_replay_oracle"
                for result in outer[method]
            )
        },
        "test_remained_hidden": {
            "pass": not (feature_cache_dir / "test.pt").exists()
        },
    }
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "phase1d_pass"
        if all(item["pass"] for item in gates.values())
        else "phase1d_failed",
        "uses_test_set": False,
        "source": source,
        "dataset_key": dataset_key,
        "replicates": [
            {
                "replicate": item["replicate"],
                "class_order": item["class_order"],
                "class_order_sha256": base._json_hash(item["class_order"]),
            }
            for item in replicates
        ],
        "inner_boundary_selection": candidates,
        "selected_boundary": selected,
        "outer_validation": outer,
        "outer_summary": summary,
        "gates": gates,
    }
    base._atomic_json(output_dir / "phase1d_results.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_boundary": selected,
        "outer_summary": summary,
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
