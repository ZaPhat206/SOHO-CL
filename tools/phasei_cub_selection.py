"""Locked CUB train-only selection for the cross-dataset Schur study.

The manifest is immutable by SHA-256. This runner never opens ``test.pt`` and
stores candidate results independently so an interrupted Colab run can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import crt_gate_runner
from tools.experiment_runner import validate_cache


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_locked_manifest(path: Path, expected_sha256: str) -> tuple[dict, str]:
    observed = sha256(path)
    if observed.lower() != expected_sha256.lower():
        raise ValueError(
            f"Phase I manifest SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "preregistered_not_executed":
        raise ValueError("Phase I manifest is not in its preregistered state")
    return payload, observed


def validate_dataset_audit(path: Path, expected: dict) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "dataset": expected["dataset"],
        "dataset_identity_sha256": expected["dataset_identity_sha256"],
        "class_mapping_sha256": expected["class_mapping_sha256"],
        "cross_split_duplicate_content_count": 0,
    }
    for key, value in checks.items():
        if payload.get(key) != value:
            raise ValueError(f"CUB dataset audit mismatch for {key}")
    if payload.get("train", {}).get("image_count") != expected["train_images"]:
        raise ValueError("CUB train image count mismatch")
    if payload.get("test", {}).get("image_count") != expected["test_images"]:
        raise ValueError("CUB test image count mismatch")
    if payload.get("train", {}).get("content_manifest_sha256") != expected["train_content_manifest_sha256"]:
        raise ValueError("CUB train content manifest mismatch")
    if payload.get("test", {}).get("content_manifest_sha256") != expected["test_content_manifest_sha256"]:
        raise ValueError("CUB test content manifest mismatch")
    return payload, sha256(path)


def runtime_args(cli, manifest: dict) -> SimpleNamespace:
    shared = manifest["shared_protocol"]
    return SimpleNamespace(
        feature_cache_dir=cli.feature_cache_dir,
        gate_cache_dir=cli.gate_cache_dir,
        output_dir=cli.output_dir,
        dataset=manifest["dataset_identity"]["dataset"],
        model_name=shared["backbone"],
        num_classes=manifest["dataset_identity"]["num_classes"],
        num_tasks=shared["num_tasks"],
        validation_fraction=shared["validation_fraction"],
        seed=shared["selection_seed"],
        device=cli.device,
        anchor_dim=shared["anchor_dim"],
        synaptic_degree=shared["synaptic_degree"],
        coding_level=shared["coding_level"],
        scatter_epsilon=shared["scatter_epsilon"],
        statistics_dtype=shared["statistics_dtype"],
        anchor_batch_size=shared["anchor_batch_size"],
    )


def validate_feature_metadata(args, manifest: dict) -> tuple[dict, dict]:
    train, _, metadata = validate_cache(args.feature_cache_dir, args, load_test=False)
    shared = manifest["shared_protocol"]
    identity = manifest["dataset_identity"]
    expected = {
        "dataset": identity["dataset"],
        "dataset_version": "processed-imagefolder",
        "backbone_model": shared["backbone"],
        "checkpoint_sha256": shared["checkpoint_sha256"],
        "preprocessing": shared["preprocessing"],
        "feature_dim": shared["feature_dim"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"feature cache metadata mismatch for {key}")
    if metadata.get("split_sizes") != {
        "train": identity["train_images"], "test": identity["test_images"]
    }:
        raise ValueError("feature cache split-size mismatch")
    if train["features"].shape != (identity["train_images"], shared["feature_dim"]):
        raise ValueError("feature cache train tensor shape mismatch")
    observed_classes = sorted(int(value) for value in torch.unique(train["labels"]).tolist())
    if observed_classes != list(range(identity["num_classes"])):
        raise ValueError("feature cache class mapping is not the complete contiguous CUB mapping")
    return train, metadata


def candidate_key(candidate: dict) -> str:
    encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def evaluate_cached(cache_dir: Path, candidate: dict, context: dict, evaluator) -> dict:
    path = cache_dir / f"{candidate_key(candidate)}.json"
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("candidate_config") != candidate:
            raise ValueError(f"candidate cache identity mismatch: {path}")
        if result.get("candidate_context") != context:
            raise ValueError(f"candidate cache source-context mismatch: {path}")
        if result.get("uses_test_set") is not False:
            raise ValueError(f"candidate cache does not certify train-only evaluation: {path}")
        print(
            f"RESUME {candidate['method']} rank={candidate.get('rank', 0)} "
            f"AA={result['validation_average_incremental_accuracy']:.4f}",
            flush=True,
        )
        return result
    result = evaluator()
    result["candidate_config"] = candidate
    result["candidate_context"] = context
    dump(path, result)
    return result


def best(results: list[dict]) -> dict:
    # max preserves manifest grid order for exact ties.
    return max(results, key=lambda item: item["validation_average_incremental_accuracy"])


def run(cli) -> dict:
    manifest, manifest_hash = load_locked_manifest(Path(cli.manifest), cli.manifest_sha256)
    dataset_audit, dataset_audit_hash = validate_dataset_audit(
        Path(cli.dataset_audit), manifest["dataset_identity"]
    )
    args = runtime_args(cli, manifest)
    train, source_metadata = validate_feature_metadata(args, manifest)
    source_train = Path(args.feature_cache_dir) / "train.pt"
    source_train_hash = sha256(source_train)
    candidate_context = {
        "locked_manifest_sha256": manifest_hash,
        "dataset_identity_sha256": dataset_audit["dataset_identity_sha256"],
        "source_train_sha256": source_train_hash,
    }

    crt_gate_runner.prepare_cache(args)
    gate_manifest = crt_gate_runner.validate_gate_cache(args, train, source_metadata)
    gate_dir = Path(args.gate_cache_dir)
    projection = torch.load(gate_dir / "anchor_projection.pt", weights_only=True)
    snapshots = [
        torch.load(gate_dir / f"statistics_task_{task:02d}.pt", weights_only=True)
        for task in range(args.num_tasks)
    ]
    validation = [
        torch.load(gate_dir / f"validation_task_{task:02d}.pt", weights_only=True)
        for task in range(args.num_tasks)
    ]
    candidate_dir = Path(args.output_dir) / "candidate_cache"
    search = manifest["equal_budget_train_only_search"]
    started = time.perf_counter()

    def raw(ridge):
        config = {"method": "raw_ridge", "ridge_lambda": ridge}
        return evaluate_cached(candidate_dir, config, candidate_context, lambda: crt_gate_runner._evaluate_raw_ridge(
            args, train, validation, snapshots, ridge
        ))

    def structured(method, anchor, residual=1.0, complement=1.0, rank=1):
        config = crt_gate_runner._candidate(method, anchor, residual, complement, rank, 1.0)
        return evaluate_cached(candidate_dir, config, candidate_context, lambda: crt_gate_runner._evaluate_candidate(
            args, train, projection, validation, snapshots, config
        ))

    raw_results = [raw(value) for value in search["raw_ridges"]]
    anchor_results = [structured("anchor_only", value) for value in search["anchor_ridges"]]
    full_results = [
        structured("full_raw_residual", anchor, residual, complement, args.raw_dim)
        for anchor in search["anchor_ridges"]
        for residual in search["residual_ridges"]
        for complement in search["complement_ridges"]
    ]
    # Low-rank methods receive exactly the same Cartesian budget. Anchor Ridge
    # is selected once by the anchor-only control, then frozen for all methods.
    anchor_best = best(anchor_results)
    locked_anchor = anchor_best["anchor_ridge"]
    low_rank_results = {}
    for method in search["low_rank_methods"]:
        values = [
            structured(method, locked_anchor, residual, complement, rank)
            for rank in search["ranks"]
            for residual in search["residual_ridges"]
            for complement in search["complement_ridges"]
        ]
        if len(values) != search["low_rank_candidates_per_method"]:
            raise AssertionError("low-rank candidate budget differs from manifest")
        low_rank_results[method] = values

    selected = {
        "raw_ridge": best(raw_results),
        "anchor_only": anchor_best,
        "full_raw_residual": best(full_results),
        **{method: best(values) for method, values in low_rank_results.items()},
    }
    fixed = manifest["fixed_transfer_from_cifar100"]
    fixed_results = {
        "raw_ridge": raw(fixed["raw_ridge"]["ridge_lambda"]),
        "anchor_only": structured("anchor_only", fixed["anchor_only"]["anchor_ridge"]),
    }
    for method in ("full_raw_residual", "schur_residual", "fisher_residual", "random_residual"):
        config = fixed[method]
        fixed_results[method] = structured(
            method, config["anchor_ridge"], config["residual_ridge"],
            config["complement_ridge"], config.get("rank", args.raw_dim)
        )

    schur = selected["schur_residual"]
    full = selected["full_raw_residual"]
    raw_selected = selected["raw_ridge"]
    strongest_control = best([selected["fisher_residual"], selected["random_residual"]])
    observed_residual = max(
        result["solver_relative_residual_max"]
        for result in [*raw_results, *anchor_results, *full_results]
        + [item for values in low_rank_results.values() for item in values]
    )
    thresholds = manifest["selection_gates"]
    gates = {
        "numerical_stability": {
            "pass": observed_residual <= thresholds["maximum_relative_solver_residual"],
            "observed": observed_residual,
        },
        "full_adds_information": {
            "pass": full["validation_average_incremental_accuracy"] - anchor_best["validation_average_incremental_accuracy"] >= thresholds["minimum_full_gain_over_anchor_percentage_points"],
            "gain": full["validation_average_incremental_accuracy"] - anchor_best["validation_average_incremental_accuracy"],
        },
        "schur_approaches_full": {
            "pass": full["validation_average_incremental_accuracy"] - schur["validation_average_incremental_accuracy"] <= thresholds["maximum_schur_gap_to_full_percentage_points"],
            "gap": full["validation_average_incremental_accuracy"] - schur["validation_average_incremental_accuracy"],
        },
        "schur_beats_raw": {
            "pass": schur["validation_average_incremental_accuracy"] - raw_selected["validation_average_incremental_accuracy"] >= thresholds["minimum_schur_gain_over_raw_percentage_points"],
            "gain": schur["validation_average_incremental_accuracy"] - raw_selected["validation_average_incremental_accuracy"],
        },
        "schur_beats_low_rank_controls": {
            "pass": schur["validation_average_incremental_accuracy"] - strongest_control["validation_average_incremental_accuracy"] >= thresholds["minimum_schur_gain_over_strongest_low_rank_control_percentage_points"],
            "gain": schur["validation_average_incremental_accuracy"] - strongest_control["validation_average_incremental_accuracy"],
            "strongest_control": strongest_control["method"],
        },
    }
    authorized = all(item["pass"] for item in gates.values())
    try:
        runner_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        runner_commit = None
    report = {
        "schema_version": 1,
        "study_id": manifest["study_id"],
        "status": "train_only_selection_passed" if authorized else "train_only_selection_failed",
        "held_out_test_authorized": authorized,
        "test_cache_opened": False,
        "hyperparameter_search_on_test": False,
        "locked_manifest_sha256": manifest_hash,
        "dataset_audit_sha256": dataset_audit_hash,
        "dataset_identity_sha256": dataset_audit["dataset_identity_sha256"],
        "source_train": {"bytes": source_train.stat().st_size, "sha256": source_train_hash},
        "source_feature_metadata": source_metadata,
        "source_gate_cache": gate_manifest,
        "runner_git_commit": runner_commit,
        "selected_equal_budget": selected,
        "fixed_transfer_validation": fixed_results,
        "candidates": [*raw_results, *anchor_results, *full_results]
        + [item for values in low_rank_results.values() for item in values],
        "candidate_counts": {
            "raw_ridge": len(raw_results),
            "anchor_only": len(anchor_results),
            "full_raw_residual": len(full_results),
            **{method: len(values) for method, values in low_rank_results.items()},
        },
        "gates": gates,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
        },
        "total_seconds": time.perf_counter() - started,
    }
    dump(Path(args.output_dir) / "selection_results.json", report)
    print(json.dumps({
        "status": report["status"],
        "selected": {
            method: {
                "validation_AA": result["validation_average_incremental_accuracy"],
                "config": result["candidate_config"],
            }
            for method, result in selected.items()
        },
        "gates": gates,
    }, indent=2), flush=True)
    print(
        "PASS: stop and return selection_results.json for review."
        if authorized else
        "STOP: a train-only gate failed; held-out CUB test remains forbidden.",
        flush=True,
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--gate-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
