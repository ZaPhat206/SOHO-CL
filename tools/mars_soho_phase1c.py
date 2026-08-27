"""Nested train-only reconstruction-fidelity gate for MARS-SOHO Phase 1C."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.mars_soho.learner import DynamicSOHOMap, _solve_ridge
from methods.mars_soho.reconstruction import SphericalReconstructor
from methods.mars_soho.statistics import SphericalClassMoments
from methods.mars_soho.tangent import TangentClassSketch
from tools import mars_soho_phase1 as base


METHODS = (
    "empirical_replay_oracle",
    "ambient_spherical",
    "tangent_lowrank_uncalibrated",
    "tangent_lowrank_calibrated",
)


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "study_id", "phase1b_artifact", "backbone",
        "phase1c", "datasets",
    }
    if not required.issubset(payload):
        raise ValueError(f"config missing fields: {sorted(required - set(payload))}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported MARS-SOHO Phase-1C schema")
    return payload


def _nested_indices(
    labels: torch.Tensor,
    *,
    split_seed: int,
    outer_fraction: float,
    inner_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = [[], [], [], []]
    for class_id in sorted(map(int, torch.unique(labels).tolist())):
        indices = torch.nonzero(labels == class_id).flatten()
        generator = torch.Generator().manual_seed(split_seed * 1_000 + class_id)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        outer_count = max(1, round(len(indices) * outer_fraction))
        outer, development = indices[:outer_count], indices[outer_count:]
        inner_count = max(1, round(len(development) * inner_fraction))
        inner_validation, inner_fit = development[:inner_count], development[inner_count:]
        for destination, part in zip(
            grouped, (inner_fit, inner_validation, development, outer)
        ):
            destination.append(part)
    return tuple(torch.cat(parts) for parts in grouped)


def _moments(features: torch.Tensor, labels: torch.Tensor) -> SphericalClassMoments:
    result = SphericalClassMoments(
        features.shape[1], device=features.device, dtype=features.dtype
    )
    result.update(features, labels)
    return result


def _encoder(config: dict, dataset_key: str, seed: int, snapshot) -> DynamicSOHOMap:
    phase = config["phase1c"]
    selected = config["datasets"][dataset_key]["locked_soho"]
    encoder = DynamicSOHOMap(
        feature_dim=config["backbone"]["feature_dim"],
        expand_dim=phase["expand_dim"],
        density=selected["density"],
        olda_dim=phase["olda_dim"],
        coding_level=selected["coding_level"],
        use_etf=selected["use_etf"],
        seed=seed,
        device=snapshot.counts.device,
        dtype=snapshot.counts.dtype,
    )
    encoder.update_rotation(snapshot)
    return encoder


def _exact_statistics(
    encoder: DynamicSOHOMap,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_ids: list[int],
    target_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gram = torch.zeros(
        (encoder.expand_dim, encoder.expand_dim),
        device=features.device,
        dtype=features.dtype,
    )
    cross = torch.zeros(
        (encoder.expand_dim, len(class_ids)),
        device=features.device,
        dtype=features.dtype,
    )
    for column, class_id in enumerate(class_ids):
        codes = encoder.encode(features[labels == class_id])
        weight = target_counts[column] / codes.shape[0]
        gram.add_(weight * (codes.T @ codes))
        cross[:, column].copy_(weight * codes.sum(dim=0))
    return gram, cross


def _pseudo_statistics(
    encoder: DynamicSOHOMap,
    generator,
    class_ids: list[int],
    counts: torch.Tensor,
    pseudo_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    gram = torch.zeros(
        (encoder.expand_dim, encoder.expand_dim),
        device=counts.device,
        dtype=counts.dtype,
    )
    cross = torch.zeros(
        (encoder.expand_dim, len(class_ids)),
        device=counts.device,
        dtype=counts.dtype,
    )
    pseudo_features = []
    code_blocks, weights = [], []
    for column, class_id in enumerate(class_ids):
        if isinstance(generator, SphericalReconstructor):
            pseudo = generator.generate(
                class_id, pseudo_per_class, heterogeneous=True
            )
        else:
            pseudo = generator.generate(class_id, pseudo_per_class)
        codes = encoder.encode(pseudo)
        weight = counts[column] / pseudo_per_class
        pseudo_features.append(pseudo)
        code_blocks.append(codes)
        weights.append(weight)
        cross[:, column].copy_(weight * codes.sum(dim=0))
    all_codes = torch.cat(code_blocks)
    row_weights = torch.cat([
        torch.full(
            (pseudo_per_class,), weight,
            device=counts.device,
            dtype=counts.dtype,
        )
        for weight in weights
    ])
    weighted = all_codes * row_weights.sqrt().unsqueeze(1)
    gram.copy_(weighted.T @ weighted)
    return gram, cross, pseudo_features


def _accuracy(
    encoder: DynamicSOHOMap,
    gram: torch.Tensor,
    cross: torch.Tensor,
    ridge_lambda: float,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_ids: list[int],
) -> tuple[float, float]:
    weights, residual = _solve_ridge(gram, cross, ridge_lambda)
    columns = (encoder.encode(features) @ weights).argmax(dim=1).cpu().tolist()
    predictions = torch.tensor([class_ids[index] for index in columns])
    accuracy = float((predictions == labels.cpu()).float().mean().item() * 100)
    return accuracy, residual


def _moment_fidelity(
    pseudo_features: list[torch.Tensor],
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    class_ids: list[int],
) -> dict:
    mean_errors, diagonal_errors, resultant_errors = [], [], []
    for class_id, pseudo in zip(class_ids, pseudo_features):
        target = torch.nn.functional.normalize(
            validation_features[validation_labels == class_id], p=2, dim=1
        )
        target_mean, pseudo_mean = target.mean(dim=0), pseudo.mean(dim=0)
        mean_errors.append(float((pseudo_mean - target_mean).norm().item()))
        target_second, pseudo_second = target.square().mean(0), pseudo.square().mean(0)
        diagonal_errors.append(float(
            ((pseudo_second - target_second).norm()
             / target_second.norm().clamp_min(torch.finfo(target.dtype).eps)).item()
        ))
        resultant_errors.append(float(
            (pseudo_mean.norm() - target_mean.norm()).abs().item()
        ))
    return {
        "mean_l2_error": statistics.fmean(mean_errors),
        "diagonal_second_relative_error": statistics.fmean(diagonal_errors),
        "resultant_length_error": statistics.fmean(resultant_errors),
    }


def _state_audit(moment_state: SphericalClassMoments, generator, method: str) -> dict:
    tensors = dict(moment_state.persistent_tensors())
    if isinstance(generator, TangentClassSketch):
        tensors.update({f"tangent_{name}": value for name, value in generator.persistent_tensors().items()})
    inventory = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": tensor.numel() * tensor.element_size(),
        }
        for name, tensor in tensors.items()
    }
    sample_level_bytes = 0
    if method == "empirical_replay_oracle":
        feature_bytes = moment_state.total_count * moment_state.feature_dim * 4
        label_bytes = moment_state.total_count * 8
        inventory.update({
            "feature_history": {
                "shape": [moment_state.total_count, moment_state.feature_dim],
                "dtype": "torch.float32",
                "bytes": feature_bytes,
                "conceptual_oracle_state": True,
            },
            "label_history": {
                "shape": [moment_state.total_count],
                "dtype": "torch.int64",
                "bytes": label_bytes,
                "conceptual_oracle_state": True,
            },
        })
        sample_level_bytes = feature_bytes + label_bytes
    return {
        "method": method,
        "exemplar_free": method != "empirical_replay_oracle",
        "historical_feature_rows": 0 if method != "empirical_replay_oracle" else moment_state.total_count,
        "sample_level_bytes": sample_level_bytes,
        "persistent_tensor_bytes": sum(item["bytes"] for item in inventory.values()),
        "persistent_tensors": inventory,
    }


def _evaluate(
    *,
    method: str,
    config: dict,
    dataset_key: str,
    stream: dict,
    fit_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    projection_seed: int,
    tangent_rank: int | None,
    device: str,
) -> dict:
    started = time.perf_counter()
    features = stream["features"][fit_indices].to(device=device, dtype=torch.float32)
    labels = stream["labels"][fit_indices].to(device=device, dtype=torch.long)
    validation_features = stream["features"][validation_indices].to(
        device=device, dtype=torch.float32
    )
    validation_labels = stream["labels"][validation_indices].to(
        device=device, dtype=torch.long
    )
    moment_state = _moments(features, labels)
    snapshot = moment_state.snapshot()
    encoder = _encoder(config, dataset_key, projection_seed, snapshot)
    class_ids = list(snapshot.class_ids)
    target_gram, target_cross = _exact_statistics(
        encoder, validation_features, validation_labels, class_ids, snapshot.counts
    )
    empirical_gram, empirical_cross = _exact_statistics(
        encoder, features, labels, class_ids, snapshot.counts
    )
    generator = None
    pseudo_features = []
    if method == "empirical_replay_oracle":
        gram, cross = empirical_gram, empirical_cross
    elif method == "ambient_spherical":
        phase = config["phase1c"]
        generator = SphericalReconstructor(
            snapshot,
            covariance_rank=phase["ambient_covariance_rank"],
            shrinkage=phase["ambient_shrinkage"],
            seed=projection_seed,
        )
        gram, cross, pseudo_features = _pseudo_statistics(
            encoder, generator, class_ids, snapshot.counts,
            phase["pseudo_per_class"],
        )
    else:
        if tangent_rank is None:
            raise ValueError("tangent methods require a rank")
        calibrated = method == "tangent_lowrank_calibrated"
        generator = TangentClassSketch(
            feature_dim=config["backbone"]["feature_dim"],
            rank=tangent_rank,
            calibrated=calibrated,
            seed=projection_seed,
            device=device,
            dtype=torch.float32,
        )
        generator.fit(features, labels, progress=True)
        gram, cross, pseudo_features = _pseudo_statistics(
            encoder, generator, class_ids, snapshot.counts,
            config["phase1c"]["pseudo_per_class"],
        )
    epsilon = torch.finfo(target_gram.dtype).eps
    gram_error = float(
        (gram - target_gram).norm().div(target_gram.norm().clamp_min(epsilon)).item()
    )
    cross_error = float(
        (cross - target_cross).norm().div(target_cross.norm().clamp_min(epsilon)).item()
    )
    accuracy, residual = _accuracy(
        encoder, gram, cross, config["phase1c"]["ridge_lambda"],
        validation_features, validation_labels, class_ids,
    )
    oracle_accuracy, _ = _accuracy(
        encoder, empirical_gram, empirical_cross,
        config["phase1c"]["ridge_lambda"], validation_features,
        validation_labels, class_ids,
    )
    fidelity = _moment_fidelity(
        pseudo_features, validation_features, validation_labels, class_ids
    ) if pseudo_features else None
    audit = _state_audit(moment_state, generator, method)
    if method != "empirical_replay_oracle" and (
        audit["historical_feature_rows"] or audit["sample_level_bytes"]
    ):
        raise AssertionError("exemplar-free fidelity model retained samples")
    return {
        "status": "complete",
        "method": method,
        "uses_test_set": False,
        "tangent_rank": tangent_rank,
        "gram_relative_error": gram_error,
        "cross_relative_error": cross_error,
        "combined_stat_error": (gram_error + cross_error) * 0.5,
        "validation_accuracy": accuracy,
        "empirical_oracle_accuracy": oracle_accuracy,
        "solver_relative_residual": residual,
        "feature_moment_fidelity": fidelity,
        "state_audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _mean(results: list[dict], key: str) -> float:
    return statistics.fmean(result[key] for result in results)


def run(
    *, config_path: Path, dataset_key: str, feature_cache_dir: Path,
    output_root: Path, device: str,
) -> dict:
    config = _read_config(config_path)
    if dataset_key not in config["datasets"]:
        raise ValueError(f"unknown dataset key: {dataset_key}")
    cached = base._validate_train_cache(feature_cache_dir, config, dataset_key)
    stream = {"features": cached["features"], "labels": cached["labels"]}
    phase = config["phase1c"]
    output_dir = output_root / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "config_sha256": base._sha256(config_path),
        "runner_sha256": base._sha256(Path(__file__).resolve()),
        "train_sha256": base._sha256(feature_cache_dir / "train.pt"),
        "phase1b_artifact": config["phase1b_artifact"],
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
                "methods/mars_soho/tangent.py",
            )
        },
    }
    replicates = []
    for index, replicate in enumerate(phase["development_replicates"]):
        parts = _nested_indices(
            stream["labels"],
            split_seed=replicate["split_seed"],
            outer_fraction=phase["outer_validation_fraction"],
            inner_fraction=phase["inner_validation_fraction"],
        )
        replicates.append({"index": index, "replicate": replicate, "parts": parts})
    candidates = []
    for rank in phase["tangent_rank_grid"]:
        results = []
        for item in replicates:
            context = {
                **source, "stage": "inner_rank_selection", "rank": rank,
                **item["replicate"],
            }
            result = base._unit(
                output_dir / "inner" / f"tangent_rank{rank}_r{item['index']}.json",
                context,
                lambda item=item, rank=rank: _evaluate(
                    method="tangent_lowrank_calibrated", config=config,
                    dataset_key=dataset_key, stream=stream,
                    fit_indices=item["parts"][0],
                    validation_indices=item["parts"][1],
                    projection_seed=item["replicate"]["projection_seed"],
                    tangent_rank=int(rank), device=device,
                ),
            )
            results.append(result)
        candidates.append({
            "rank": int(rank),
            "mean_inner_combined_stat_error": _mean(results, "combined_stat_error"),
            "replicates": results,
        })
    best_error = min(item["mean_inner_combined_stat_error"] for item in candidates)
    selected_rank = min(
        item["rank"] for item in candidates
        if item["mean_inner_combined_stat_error"]
        <= best_error + phase["near_tie_tolerance"]
    )
    outer = {method: [] for method in METHODS}
    for method in METHODS:
        for item in replicates:
            rank = selected_rank if method.startswith("tangent_") else None
            context = {
                **source, "stage": "outer_fidelity", "method": method,
                "selected_rank": rank, **item["replicate"],
            }
            result = base._unit(
                output_dir / "outer" / f"{method}_r{item['index']}.json",
                context,
                lambda item=item, method=method, rank=rank: _evaluate(
                    method=method, config=config, dataset_key=dataset_key,
                    stream=stream, fit_indices=item["parts"][2],
                    validation_indices=item["parts"][3],
                    projection_seed=item["replicate"]["projection_seed"],
                    tangent_rank=rank, device=device,
                ),
            )
            outer[method].append(result)
    summary = {
        method: {
            key: _mean(results, key)
            for key in (
                "combined_stat_error", "gram_relative_error",
                "cross_relative_error", "validation_accuracy",
                "elapsed_seconds",
            )
        }
        for method, results in outer.items()
    }
    proposed = summary["tangent_lowrank_calibrated"]
    ambient = summary["ambient_spherical"]
    empirical = summary["empirical_replay_oracle"]
    reduction = (
        ambient["combined_stat_error"] - proposed["combined_stat_error"]
    ) / ambient["combined_stat_error"]
    resultant_error = statistics.fmean(
        result["feature_moment_fidelity"]["resultant_length_error"]
        for result in outer["tangent_lowrank_calibrated"]
    )
    gates = {
        "relative_stat_error_reduction": {
            "threshold": phase["gates"]["minimum_relative_stat_error_reduction"],
            "observed": reduction,
            "pass": reduction
            >= phase["gates"]["minimum_relative_stat_error_reduction"],
        },
        "accuracy_gain_over_ambient_pp": {
            "threshold": phase["gates"]["minimum_accuracy_gain_pp"],
            "observed": proposed["validation_accuracy"] - ambient["validation_accuracy"],
            "pass": proposed["validation_accuracy"] - ambient["validation_accuracy"]
            >= phase["gates"]["minimum_accuracy_gain_pp"],
        },
        "accuracy_gap_to_empirical_oracle_pp": {
            "threshold": phase["gates"]["maximum_accuracy_gap_to_empirical_oracle_pp"],
            "observed": empirical["validation_accuracy"] - proposed["validation_accuracy"],
            "pass": empirical["validation_accuracy"] - proposed["validation_accuracy"]
            <= phase["gates"]["maximum_accuracy_gap_to_empirical_oracle_pp"],
        },
        "resultant_length_error": {
            "threshold": phase["gates"]["maximum_resultant_length_error"],
            "observed": resultant_error,
            "pass": resultant_error
            <= phase["gates"]["maximum_resultant_length_error"],
        },
        "sample_free_state": {
            "pass": all(
                result["state_audit"]["sample_level_bytes"] == 0
                for method in METHODS if method != "empirical_replay_oracle"
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
        "status": "phase1c_pass"
        if all(value["pass"] for value in gates.values())
        else "phase1c_failed",
        "uses_test_set": False,
        "source": source,
        "dataset_key": dataset_key,
        "inner_rank_selection": candidates,
        "selected_tangent_rank": selected_rank,
        "outer_fidelity": outer,
        "outer_summary": summary,
        "gates": gates,
    }
    base._atomic_json(output_dir / "phase1c_results.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_tangent_rank": selected_rank,
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
        config_path=args.config.resolve(), dataset_key=args.dataset_key,
        feature_cache_dir=args.feature_cache_dir.resolve(),
        output_root=args.output_root.resolve(), device=args.device,
    )


if __name__ == "__main__":
    main()
