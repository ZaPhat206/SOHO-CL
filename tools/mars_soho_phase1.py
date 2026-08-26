"""Locked train-only Phase-1 study for MARS-SOHO moment reconstruction.

The runner refuses a visible ``test.pt``. It selects a fixed Ridge coefficient
with the exact replay oracle, selects each reconstruction family from an equal
predeclared grid on the same inner splits, then evaluates locked candidates on
untouched outer validation partitions. Completed units resume from JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.mars_soho import MARSExactReplayOracle, MARSSOHOLearner
from methods.mars_soho.learner import _solve_ridge


METHODS = (
    "exact_replay_oracle",
    "shared_gaussian",
    "heterogeneous_spherical",
    "support_aware",
    "shuffled_support",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _environment(device: str) -> dict:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    resolved = torch.device(device)
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(resolved),
        "cuda_device": torch.cuda.get_device_name(resolved)
        if resolved.type == "cuda" else None,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "study_id", "backbone", "phase1", "datasets"}
    if not required.issubset(payload):
        raise ValueError(f"config missing fields: {sorted(required - set(payload))}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported MARS-SOHO Phase-1 schema")
    return payload


def _validate_train_cache(cache_dir: Path, config: dict, dataset_key: str) -> dict:
    if (cache_dir / "test.pt").exists():
        raise RuntimeError("Phase 1 refuses a visible test.pt")
    metadata_path, train_path = cache_dir / "metadata.json", cache_dir / "train.pt"
    if not metadata_path.is_file() or not train_path.is_file():
        raise FileNotFoundError("Phase 1 requires metadata.json and train.pt")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = config["datasets"][dataset_key]
    backbone = config["backbone"]
    expected = {
        "dataset": dataset["dataset"],
        "backbone_model": backbone["model_name"],
        "checkpoint_sha256": backbone["checkpoint_sha256"],
        "preprocessing": backbone["preprocessing"],
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items() if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"feature cache identity mismatch: {mismatches}")
    train = torch.load(train_path, weights_only=True, map_location="cpu")
    features, labels = train.get("features"), train.get("labels")
    if (
        not isinstance(features, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or features.shape != (dataset["train_samples"], backbone["feature_dim"])
        or labels.shape != (dataset["train_samples"],)
        or not bool(torch.isfinite(features).all())
        or sorted(map(int, torch.unique(labels).tolist()))
        != list(range(dataset["num_classes"]))
    ):
        raise ValueError("invalid train-only feature cache tensors")
    return {"features": features, "labels": labels, "metadata": metadata}


def _nested_parts(
    labels: torch.Tensor,
    class_order: list[int],
    tasks: int,
    split_seed: int,
    outer_fraction: float,
    inner_fraction: float,
):
    if len(class_order) % tasks:
        raise ValueError("class count must be divisible by number of tasks")
    per_class = {}
    for class_id in sorted(map(int, torch.unique(labels).tolist())):
        indices = torch.nonzero(labels == class_id).flatten()
        generator = torch.Generator().manual_seed(split_seed * 1_000 + class_id)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        outer_count = max(1, round(len(indices) * outer_fraction))
        development, outer_validation = indices[outer_count:], indices[:outer_count]
        inner_count = max(1, round(len(development) * inner_fraction))
        inner_fit, inner_validation = development[inner_count:], development[:inner_count]
        if min(len(inner_fit), len(inner_validation), len(outer_validation)) <= 0:
            raise ValueError(f"empty nested partition for class {class_id}")
        per_class[class_id] = (
            inner_fit, inner_validation, development, outer_validation
        )
    classes_per_task = len(class_order) // tasks
    grouped = [[], [], [], []]
    for task in range(tasks):
        task_classes = class_order[
            task * classes_per_task : (task + 1) * classes_per_task
        ]
        for part in range(4):
            grouped[part].append(
                torch.cat([per_class[class_id][part] for class_id in task_classes])
            )
    return tuple(grouped)


def _metrics(matrix: list[list[float]]) -> dict:
    stages = [statistics.fmean(row) for row in matrix]
    forgetting = []
    for task in range(max(len(matrix) - 1, 0)):
        best = max(matrix[stage][task] for stage in range(task, len(matrix) - 1))
        forgetting.append(best - matrix[-1][task])
    return {
        "accuracy_matrix": matrix,
        "stage_accuracy": stages,
        "final_accuracy": stages[-1],
        "average_incremental_accuracy": statistics.fmean(stages),
        "forgetting": statistics.fmean(forgetting) if forgetting else 0.0,
    }


def _state_audit(learner) -> dict:
    inventory = {}
    for name, tensor in learner.persistent_tensors().items():
        inventory[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": tensor.numel() * tensor.element_size(),
        }
    feature_rows = sum(
        int(value.shape[0]) for value in getattr(learner, "feature_history", [])
    )
    label_rows = sum(
        int(value.shape[0]) for value in getattr(learner, "label_history", [])
    )
    sample_level_bytes = sum(
        item["bytes"] for name, item in inventory.items()
        if name.startswith(("feature_history_", "label_history_"))
    )
    audit = {
        "exemplar_free": bool(learner.is_exemplar_free),
        "historical_feature_rows": feature_rows,
        "historical_label_rows": label_rows,
        "sample_level_bytes": sample_level_bytes,
        "persistent_tensor_bytes": sum(item["bytes"] for item in inventory.values()),
        "persistent_tensors": inventory,
    }
    if learner.is_exemplar_free:
        learner.assert_exemplar_free_state()
        if feature_rows or label_rows or sample_level_bytes:
            raise AssertionError("exemplar-free learner retained sample-level state")
    return audit


def _base_kwargs(config: dict, dataset_key: str, seed: int, device: str) -> dict:
    phase = config["phase1"]
    selected = config["datasets"][dataset_key]["locked_soho"]
    return {
        "feature_dim": config["backbone"]["feature_dim"],
        "expand_dim": phase["expand_dim"],
        "density": selected["density"],
        "olda_dim": phase["olda_dim"],
        "use_etf": selected["use_etf"],
        "coding_level": selected["coding_level"],
        "seed": seed,
        "device": device,
        # Match current SOHO's cached-feature numeric policy and avoid slow
        # float64 dense WTA Gram products on consumer Colab GPUs.
        "dtype": torch.float32,
    }


def _reconstruction_candidates(config: dict) -> list[dict]:
    search = config["phase1"]["reconstruction_grid"]
    candidates = []
    for rank in search["covariance_rank"]:
        for shrinkage in search["shrinkage"]:
            for pseudo_count in search["pseudo_per_class"]:
                candidates.append({
                    "covariance_rank": int(rank),
                    "shrinkage": float(shrinkage),
                    "pseudo_per_class": int(pseudo_count),
                    "pilot_per_class": int(search["pilot_per_class"]),
                    "minimum_per_class": int(search["minimum_per_class"]),
                    "risk_floor": float(search["risk_floor"]),
                })
    return candidates


def _evaluate(
    *,
    method: str,
    config: dict,
    dataset_key: str,
    stream: dict,
    fit_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    projection_seed: int,
    ridge_lambda: float,
    candidate: dict | None,
    device: str,
) -> dict:
    kwargs = _base_kwargs(config, dataset_key, projection_seed, device)
    kwargs["ridge_lambda"] = ridge_lambda
    if method == "exact_replay_oracle":
        learner = MARSExactReplayOracle(**kwargs)
    else:
        learner = MARSSOHOLearner(
            **kwargs,
            model_mode=method,
            **candidate,
        )
    matrix, task_diagnostics = [], []
    started = time.perf_counter()
    for task, indices in enumerate(fit_parts):
        learner.update(stream["features"][indices], stream["labels"][indices])
        row = []
        for previous in range(task + 1):
            validation = validation_parts[previous]
            predictions = learner.predict(stream["features"][validation])
            labels = stream["labels"][validation].cpu()
            row.append(float((predictions == labels).float().mean().item() * 100))
        matrix.append(row)
        task_diagnostics.append(dict(learner.diagnostics))
        print(
            f"TASK method={method} task={task+1}/{len(fit_parts)} "
            f"seen_AA={statistics.fmean(row):.4f}",
            flush=True,
        )
    return {
        "status": "complete",
        "method": method,
        "uses_test_set": False,
        "ridge_lambda": ridge_lambda,
        "candidate": candidate,
        **_metrics(matrix),
        "persistent_state_bytes": learner.persistent_state_bytes(),
        "exemplar_free": learner.is_exemplar_free,
        "state_audit": _state_audit(learner),
        "task_diagnostics": task_diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _evaluate_oracle_ridge_grid(
    *, config: dict, dataset_key: str, stream: dict,
    fit_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    projection_seed: int, ridge_grid: list[float], device: str,
) -> dict:
    """Run the expensive oracle map once and solve every declared λ per task."""
    kwargs = _base_kwargs(config, dataset_key, projection_seed, device)
    kwargs["ridge_lambda"] = ridge_grid[0]
    learner = MARSExactReplayOracle(**kwargs)
    matrices = {value: [] for value in ridge_grid}
    for task, indices in enumerate(fit_parts):
        learner.update(stream["features"][indices], stream["labels"][indices])
        for ridge in ridge_grid:
            weights, _ = _solve_ridge(learner.G, learner.Q, ridge)
            row = []
            for previous in range(task + 1):
                validation = validation_parts[previous]
                codes = learner.encoder.encode(stream["features"][validation])
                columns = (codes @ weights).argmax(dim=1).detach().cpu().tolist()
                predicted = torch.tensor([learner.class_ids[index] for index in columns])
                labels = stream["labels"][validation].cpu()
                row.append(float((predicted == labels).float().mean().item() * 100))
            matrices[ridge].append(row)
        print(f"RIDGE GRID task={task+1}/{len(fit_parts)}", flush=True)
    return {
        str(ridge): {"ridge_lambda": ridge, **_metrics(matrix)}
        for ridge, matrix in matrices.items()
    }


def _unit(path: Path, context: dict, evaluator) -> dict:
    context_hash = _json_hash(context)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("context_sha256") != context_hash:
            raise RuntimeError(f"resume context mismatch: {path}")
        print(f"RESTORED {path.stem}", flush=True)
        return payload["result"]
    print(f"START {path.stem}", flush=True)
    result = evaluator()
    _atomic_json(path, {"context_sha256": context_hash, "result": result})
    print(f"DONE {path.stem}", flush=True)
    return result


def _mean_aia(results: list[dict]) -> float:
    return statistics.fmean(result["average_incremental_accuracy"] for result in results)


def _select_candidate(results: list[dict], tolerance_pp: float) -> dict:
    best_score = max(result["mean_inner_aia"] for result in results)
    eligible = [
        result for result in results
        if result["mean_inner_aia"] >= best_score - tolerance_pp
    ]
    # Within a near tie, prefer fewer pseudo directions, lower covariance rank,
    # then stronger shrinkage for memory/time/stability.
    return min(
        eligible,
        key=lambda result: (
            result["candidate"]["pseudo_per_class"],
            result["candidate"]["covariance_rank"],
            -result["candidate"]["shrinkage"],
        ),
    )


def run(
    *, config_path: Path, dataset_key: str, feature_cache_dir: Path,
    output_root: Path, device: str,
) -> dict:
    config = _read_config(config_path)
    if dataset_key not in config["datasets"]:
        raise ValueError(f"unknown dataset key: {dataset_key}")
    cached = _validate_train_cache(feature_cache_dir, config, dataset_key)
    stream = {"features": cached["features"], "labels": cached["labels"]}
    dataset, phase = config["datasets"][dataset_key], config["phase1"]
    output_dir = output_root / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "train_sha256": _sha256(feature_cache_dir / "train.pt"),
        "dataset_key": dataset_key,
        "device": device,
        "environment": _environment(device),
        "method_sha256": {
            relative: _sha256(REPOSITORY_ROOT / relative)
            for relative in (
                "methods/mars_soho/statistics.py",
                "methods/mars_soho/geometry.py",
                "methods/mars_soho/reconstruction.py",
                "methods/mars_soho/learner.py",
            )
        },
    }
    replicates = []
    for index, replicate in enumerate(phase["development_replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(dataset["num_classes"]), dataset["num_classes"]
        )
        parts = _nested_parts(
            stream["labels"], class_order, dataset["num_tasks"],
            phase["split_seed"], phase["outer_validation_fraction"],
            phase["inner_validation_fraction"],
        )
        replicates.append({
            "index": index, "replicate": replicate,
            "class_order": class_order, "parts": parts,
        })
    ridge_grid = list(map(float, phase["ridge_grid"]))
    ridge_results = []
    for item in replicates:
        context = {**source, "stage": "ridge_grid", **item["replicate"], "ridge_grid": ridge_grid}
        result = _unit(
            output_dir / "inner" / f"ridge_rep{item['index']}.json",
            context,
            lambda item=item: _evaluate_oracle_ridge_grid(
                config=config, dataset_key=dataset_key, stream=stream,
                fit_parts=item["parts"][0], validation_parts=item["parts"][1],
                projection_seed=item["replicate"]["projection_seed"],
                ridge_grid=ridge_grid, device=device,
            ),
        )
        ridge_results.append(result)
    ridge_scores = {
        ridge: statistics.fmean(
            result[str(ridge)]["average_incremental_accuracy"]
            for result in ridge_results
        )
        for ridge in ridge_grid
    }
    # Deterministic conservative tie-break: larger λ within declared tolerance.
    ridge_best = max(ridge_scores.values())
    ridge_lambda = max(
        ridge for ridge, score in ridge_scores.items()
        if score >= ridge_best - phase["near_tie_tolerance_pp"]
    )
    candidate_grid = _reconstruction_candidates(config)
    selected = {}
    inner_summary = {}
    # Tune the two distribution models. Support-aware and shuffled-risk inherit
    # the selected heterogeneous model, isolating allocation as their only
    # experimental difference.
    for method in ("shared_gaussian", "heterogeneous_spherical"):
        candidates = []
        for candidate_index, candidate in enumerate(candidate_grid):
            per_replicate = []
            for item in replicates:
                context = {
                    **source, "stage": "reconstruction_selection", "method": method,
                    "candidate": candidate, "ridge_lambda": ridge_lambda,
                    **item["replicate"],
                }
                result = _unit(
                    output_dir / "inner" / f"{method}_c{candidate_index}_r{item['index']}.json",
                    context,
                    lambda item=item, candidate=candidate, method=method: _evaluate(
                        method=method, config=config, dataset_key=dataset_key,
                        stream=stream, fit_parts=item["parts"][0],
                        validation_parts=item["parts"][1],
                        projection_seed=item["replicate"]["projection_seed"],
                        ridge_lambda=ridge_lambda, candidate=candidate,
                        device=device,
                    ),
                )
                per_replicate.append(result)
            candidates.append({
                "candidate": candidate,
                "mean_inner_aia": _mean_aia(per_replicate),
                "replicates": per_replicate,
            })
        chosen = _select_candidate(candidates, phase["near_tie_tolerance_pp"])
        selected[method] = chosen["candidate"]
        inner_summary[method] = {
            "selected": chosen["candidate"],
            "selected_mean_inner_aia": chosen["mean_inner_aia"],
            "candidates": candidates,
        }
    for method in ("support_aware", "shuffled_support"):
        selected[method] = dict(selected["heterogeneous_spherical"])
        inner_summary[method] = {
            "selected": selected[method],
            "selection_policy": (
                "inherits heterogeneous_spherical config; allocation-only control"
            ),
            "candidates": [],
        }
    outer = {method: [] for method in METHODS}
    for method in METHODS:
        for item in replicates:
            candidate = None if method == "exact_replay_oracle" else selected[method]
            context = {
                **source, "stage": "outer_validation", "method": method,
                "candidate": candidate, "ridge_lambda": ridge_lambda,
                **item["replicate"],
            }
            result = _unit(
                output_dir / "outer" / f"{method}_r{item['index']}.json",
                context,
                lambda item=item, method=method, candidate=candidate: _evaluate(
                    method=method, config=config, dataset_key=dataset_key,
                    stream=stream, fit_parts=item["parts"][2],
                    validation_parts=item["parts"][3],
                    projection_seed=item["replicate"]["projection_seed"],
                    ridge_lambda=ridge_lambda, candidate=candidate,
                    device=device,
                ),
            )
            outer[method].append(result)
    outer_aia = {method: _mean_aia(results) for method, results in outer.items()}
    proposed = outer_aia["support_aware"]
    gates = {
        "support_gap_to_oracle_at_most_pp": {
            "threshold": phase["gates"]["max_oracle_gap_pp"],
            "observed": outer_aia["exact_replay_oracle"] - proposed,
            "pass": outer_aia["exact_replay_oracle"] - proposed
            <= phase["gates"]["max_oracle_gap_pp"],
        },
        "support_gain_over_shared_at_least_pp": {
            "threshold": phase["gates"]["min_shared_gain_pp"],
            "observed": proposed - outer_aia["shared_gaussian"],
            "pass": proposed - outer_aia["shared_gaussian"]
            >= phase["gates"]["min_shared_gain_pp"],
        },
        "support_gain_over_shuffled_at_least_pp": {
            "threshold": phase["gates"]["min_shuffled_gain_pp"],
            "observed": proposed - outer_aia["shuffled_support"],
            "pass": proposed - outer_aia["shuffled_support"]
            >= phase["gates"]["min_shuffled_gain_pp"],
        },
        "test_remained_hidden": {"pass": not (feature_cache_dir / "test.pt").exists()},
    }
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "phase1_pass" if all(value["pass"] for value in gates.values()) else "phase1_failed",
        "uses_test_set": False,
        "source": source,
        "dataset_key": dataset_key,
        "replicates": [
            {
                "replicate": item["replicate"],
                "class_order": item["class_order"],
                "class_order_sha256": _json_hash(item["class_order"]),
            }
            for item in replicates
        ],
        "ridge_scores": ridge_scores,
        "selected_ridge_lambda": ridge_lambda,
        "inner_selection": inner_summary,
        "selected_reconstruction": selected,
        "outer_validation": outer,
        "outer_mean_aia": outer_aia,
        "gates": gates,
    }
    _atomic_json(output_dir / "phase1_results.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_ridge_lambda": ridge_lambda,
        "selected_reconstruction": selected,
        "outer_mean_aia": outer_aia,
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
