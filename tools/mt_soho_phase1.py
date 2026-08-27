"""Locked CIFAR-100 train-only feasibility study for MT-SOHO Phase 1A.

The runner fails closed when ``test.pt`` is visible.  Hyperparameters are
selected on inner validation partitions and locked controls are evaluated on
untouched outer validation partitions.  Every candidate result is written
atomically so an interrupted Colab run can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.mt_soho import MTSOHOLearner


OUTER_METHODS = (
    "fixed_wta_ridge",
    "mt_whitened",
    "mt_unwhitened",
    "mt_shuffled",
)


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _environment(device: str) -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
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


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "study_id", "backbone", "dataset", "phase1"}
    if not required.issubset(config):
        raise ValueError(f"config missing fields: {sorted(required - set(config))}")
    if config["schema_version"] != 1:
        raise ValueError("unsupported MT-SOHO Phase-1 schema")
    return config


def _validate_train_cache(cache_dir: Path, config: dict) -> dict:
    if (cache_dir / "test.pt").exists():
        raise RuntimeError("MT-SOHO Phase 1 refuses a visible test.pt")
    metadata_path, train_path = cache_dir / "metadata.json", cache_dir / "train.pt"
    if not metadata_path.is_file() or not train_path.is_file():
        raise FileNotFoundError("Phase 1 requires metadata.json and train.pt")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": config["dataset"]["name"],
        "backbone_model": config["backbone"]["model_name"],
        "checkpoint_sha256": config["backbone"]["checkpoint_sha256"],
        "preprocessing": config["backbone"]["preprocessing"],
    }
    mismatch = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items() if metadata.get(key) != value
    }
    if mismatch:
        raise ValueError(f"feature cache identity mismatch: {mismatch}")
    train = torch.load(train_path, weights_only=True, map_location="cpu")
    features, labels = train.get("features"), train.get("labels")
    dataset, backbone = config["dataset"], config["backbone"]
    if (
        not isinstance(features, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or tuple(features.shape) != (dataset["train_samples"], backbone["feature_dim"])
        or tuple(labels.shape) != (dataset["train_samples"],)
        or not bool(torch.isfinite(features).all())
        or sorted(map(int, torch.unique(labels).tolist()))
        != list(range(dataset["num_classes"]))
    ):
        raise ValueError("invalid train-only feature cache")
    return {"features": features, "labels": labels, "metadata": metadata}


def _nested_parts(
    labels: torch.Tensor,
    class_order: list[int],
    tasks: int,
    split_seed: int,
    outer_fraction: float,
    inner_fraction: float,
) -> tuple[list[torch.Tensor], ...]:
    if len(class_order) % tasks:
        raise ValueError("class count must be divisible by task count")
    per_class = {}
    for class_id in sorted(map(int, torch.unique(labels).tolist())):
        indices = torch.nonzero(labels == class_id).flatten()
        generator = torch.Generator().manual_seed(split_seed * 1000 + class_id)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        outer_count = max(1, round(len(indices) * outer_fraction))
        development, outer = indices[outer_count:], indices[:outer_count]
        inner_count = max(1, round(len(development) * inner_fraction))
        fit, inner = development[inner_count:], development[:inner_count]
        if min(len(fit), len(inner), len(outer)) <= 0:
            raise ValueError(f"empty nested split for class {class_id}")
        per_class[class_id] = fit, inner, development, outer
    classes_per_task = len(class_order) // tasks
    grouped = [[], [], [], []]
    for task in range(tasks):
        task_classes = class_order[task * classes_per_task : (task + 1) * classes_per_task]
        for part in range(4):
            grouped[part].append(torch.cat([per_class[class_id][part] for class_id in task_classes]))
    return tuple(grouped)


def _metrics(matrix: list[list[float]]) -> dict:
    stage_accuracy = [statistics.fmean(row) for row in matrix]
    forgetting = []
    for task in range(max(len(matrix) - 1, 0)):
        best = max(matrix[stage][task] for stage in range(task, len(matrix) - 1))
        forgetting.append(best - matrix[-1][task])
    return {
        "accuracy_matrix": matrix,
        "stage_accuracy": stage_accuracy,
        "final_accuracy": stage_accuracy[-1],
        "average_incremental_accuracy": statistics.fmean(stage_accuracy),
        "forgetting": statistics.fmean(forgetting) if forgetting else 0.0,
    }


def _class_order(classes: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(classes, generator=generator).tolist()


def _learner_kwargs(config: dict, method: str, candidate: dict, projection_seed: int, device: str) -> dict:
    phase, anchor = config["phase1"], config["phase1"]["anchor"]
    return {
        "method": method,
        "feature_dim": config["backbone"]["feature_dim"],
        "expand_dim": phase["expand_dim"],
        "synaptic_degree": anchor["synaptic_degree"],
        "coding_level": anchor["coding_level"],
        "anchor_ridge": float(candidate["anchor_ridge"]),
            "projection_ridge": float(candidate.get("projection_ridge", candidate["anchor_ridge"])),
            "adapted_ridge": float(candidate.get("adapted_ridge", 1.0)),
        "target_rank": int(candidate.get("target_rank", phase["target_grid"]["rank"][0])),
        "shrinkage": float(candidate.get("shrinkage", phase["target_grid"]["shrinkage"][0])),
        "adaptation_weight": float(candidate.get("adaptation_weight", 0.0)),
        "geometry_epsilon": float(phase["geometry_epsilon"]),
        "seed": int(projection_seed),
        "device": device,
        "dtype": torch.float32,
    }


def _evaluate(
    *,
    config: dict,
    stream: dict,
    fit_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    method: str,
    candidate: dict,
    projection_seed: int,
    device: str,
) -> dict:
    learner = MTSOHOLearner(**_learner_kwargs(config, method, candidate, projection_seed, device))
    matrix, update_seconds, inference_seconds = [], 0.0, 0.0
    for task, fit_indices in enumerate(fit_parts):
        started = time.perf_counter()
        learner.update(stream["features"][fit_indices], stream["labels"][fit_indices])
        update_seconds += time.perf_counter() - started
        row = []
        for previous in range(task + 1):
            indices = validation_parts[previous]
            started = time.perf_counter()
            predictions = learner.predict(stream["features"][indices])
            inference_seconds += time.perf_counter() - started
            row.append(float((predictions == stream["labels"][indices]).float().mean().item() * 100.0))
        matrix.append(row)
        print(
            f"TASK method={method} stage={task + 1}/{len(fit_parts)} "
            f"seen_AA={statistics.fmean(row):.4f}", flush=True
        )
    result = _metrics(matrix)
    result.update({
        "method": method,
        "candidate": candidate,
        "projection_seed": int(projection_seed),
        "update_seconds": update_seconds,
        "inference_seconds": inference_seconds,
        "persistent_state_bytes": learner.persistent_state_bytes(),
        "diagnostics": {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in learner.diagnostics.items()
        },
        "exemplar_free": learner.is_exemplar_free,
        "persistent_state_names": sorted(learner.persistent_tensors()),
    })
    learner.assert_exemplar_free_state()
    return result


def _cached_evaluation(path: Path, context: dict, evaluator) -> dict:
    context_hash = _json_hash(context)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("context_hash") != context_hash:
            raise RuntimeError(f"resume context mismatch: {path}")
        print(f"RESTORE {path.stem}", flush=True)
        return payload["result"]
    result = evaluator()
    _atomic_json(path, {"context_hash": context_hash, "context": context, "result": result})
    return result


def _mean_aia(results: list[dict]) -> float:
    return statistics.fmean(item["average_incremental_accuracy"] for item in results)


def _select(results: list[dict], tolerance_pp: float, complexity_key) -> dict:
    best = max(item["mean_inner_aia"] for item in results)
    eligible = [item for item in results if item["mean_inner_aia"] >= best - tolerance_pp]
    return min(eligible, key=lambda item: complexity_key(item["candidate"]))


def _candidate_grid(config: dict, anchor_ridge: float) -> list[dict]:
    grid = config["phase1"]["target_grid"]
    return [
        {
            "anchor_ridge": float(anchor_ridge),
            "projection_ridge": float(anchor_ridge),
            "adapted_ridge": float(adapted_ridge),
            "target_rank": int(rank),
            "shrinkage": float(shrinkage),
            "adaptation_weight": float(weight),
        }
        for rank, shrinkage, adapted_ridge, weight in itertools.product(
            grid["rank"], grid["shrinkage"], grid["adapted_ridge"], grid["adaptation_weight"]
        )
    ]


def run(*, config_path: Path, feature_cache_dir: Path, output_dir: Path, device: str) -> dict:
    config = _read_config(config_path)
    stream = _validate_train_cache(feature_cache_dir, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase, dataset = config["phase1"], config["dataset"]
    config_hash = _json_hash(config)
    partitions = []
    for replicate in phase["development_replicates"]:
        order = _class_order(dataset["num_classes"], replicate["class_order_seed"])
        parts = _nested_parts(
            stream["labels"], order, dataset["num_tasks"], phase["split_seed"],
            phase["outer_validation_fraction"], phase["inner_validation_fraction"],
        )
        partitions.append((replicate, order, parts))

    anchor_search = []
    for ridge in phase["anchor_ridge_grid"]:
        candidate = {"anchor_ridge": float(ridge), "adaptation_weight": 0.0}
        replicate_results = []
        print(f"ANCHOR START lambda={ridge}", flush=True)
        for replicate, order, parts in partitions:
            context = {
                "config_hash": config_hash, "split": "inner", "method": "fixed_wta_ridge",
                "candidate": candidate, "replicate": replicate, "class_order": order,
            }
            ridge_name = str(ridge).replace(".", "p")
            name = f"anchor_lam{ridge_name}_seed{replicate['projection_seed']}.json"
            replicate_results.append(_cached_evaluation(
                output_dir / "inner" / name, context,
                lambda r=replicate, p=parts: _evaluate(
                    config=config, stream=stream, fit_parts=p[0], validation_parts=p[1],
                    method="fixed_wta_ridge", candidate=candidate,
                    projection_seed=r["projection_seed"], device=device,
                ),
            ))
        anchor_search.append({"candidate": candidate, "mean_inner_aia": _mean_aia(replicate_results), "replicates": replicate_results})
        print(f"ANCHOR DONE lambda={ridge} AIA={anchor_search[-1]['mean_inner_aia']:.4f}", flush=True)
    selected_anchor = _select(
        anchor_search, phase["near_tie_tolerance_pp"],
        lambda candidate: -candidate["anchor_ridge"],
    )
    anchor_ridge = selected_anchor["candidate"]["anchor_ridge"]

    mt_search = []
    candidates = _candidate_grid(config, anchor_ridge)
    for index, candidate in enumerate(candidates, 1):
        replicate_results = []
        print(f"START candidate={index}/{len(candidates)} config={candidate}", flush=True)
        for replicate, order, parts in partitions:
            context = {
                "config_hash": config_hash, "split": "inner", "method": "mt_whitened",
                "candidate": candidate, "replicate": replicate, "class_order": order,
            }
            name = f"mt_{index:02d}_seed{replicate['projection_seed']}.json"
            replicate_results.append(_cached_evaluation(
                output_dir / "inner" / name, context,
                lambda r=replicate, p=parts: _evaluate(
                    config=config, stream=stream, fit_parts=p[0], validation_parts=p[1],
                    method="mt_whitened", candidate=candidate,
                    projection_seed=r["projection_seed"], device=device,
                ),
            ))
        mt_search.append({"candidate": candidate, "mean_inner_aia": _mean_aia(replicate_results), "replicates": replicate_results})
        print(f"DONE candidate={index}/{len(candidates)} AIA={mt_search[-1]['mean_inner_aia']:.4f}", flush=True)
    selected_mt = _select(
        mt_search, phase["near_tie_tolerance_pp"],
        lambda c: (c["target_rank"], c["adaptation_weight"], c["shrinkage"], -c["adapted_ridge"]),
    )

    outer = {method: [] for method in OUTER_METHODS}
    for method in OUTER_METHODS:
        candidate = selected_anchor["candidate"] if method == "fixed_wta_ridge" else selected_mt["candidate"]
        for replicate, order, parts in partitions:
            context = {
                "config_hash": config_hash, "split": "outer", "method": method,
                "candidate": candidate, "replicate": replicate, "class_order": order,
            }
            path = output_dir / "outer" / f"{method}_seed{replicate['projection_seed']}.json"
            outer[method].append(_cached_evaluation(
                path, context,
                lambda m=method, c=candidate, r=replicate, p=parts: _evaluate(
                    config=config, stream=stream, fit_parts=p[2], validation_parts=p[3],
                    method=m, candidate=c, projection_seed=r["projection_seed"], device=device,
                ),
            ))

    means = {method: _mean_aia(results) for method, results in outer.items()}
    proposed = means["mt_whitened"]
    gates = {
        "heldout_test_remained_hidden": not (feature_cache_dir / "test.pt").exists(),
        "all_methods_exemplar_free": all(item["exemplar_free"] for values in outer.values() for item in values),
        "numerical_stability": max(
            item["diagnostics"].get("solver_relative_residual", 0.0)
            for values in outer.values() for item in values
        ) <= phase["gates"]["max_solver_relative_residual"],
        "beats_fixed_anchor": proposed - means["fixed_wta_ridge"] >= phase["gates"]["min_fixed_gain_pp"],
        "beats_shuffled_control": proposed - means["mt_shuffled"] >= phase["gates"]["min_shuffled_gain_pp"],
        "whitening_adds_value": proposed - means["mt_unwhitened"] >= phase["gates"]["min_whitening_gain_pp"],
    }
    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "phase1a_pass" if all(gates.values()) else "phase1a_fail",
        "uses_test_set": False,
        "config_hash": config_hash,
        "environment": _environment(device),
        "selected_anchor": selected_anchor,
        "selected_mt": selected_mt,
        "outer_validation": outer,
        "outer_mean_aia": means,
        "gates": gates,
        "next_step": "compare against replay SOHO at matched width" if all(gates.values()) else "stop MT-SOHO accuracy branch",
    }
    _atomic_json(output_dir / "phase1_results.json", summary)
    print(json.dumps({"status": summary["status"], "selected_mt": selected_mt["candidate"], "outer_mean_aia": means, "gates": gates}, indent=2), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(config_path=args.config, feature_cache_dir=args.feature_cache_dir, output_dir=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
