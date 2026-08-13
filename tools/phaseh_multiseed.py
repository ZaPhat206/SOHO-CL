"""Locked, resumable Phase H-B multi-seed CIFAR-100 study runner."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.cached_replay_baselines import (
    CachedFlyCLFidelity,
    CachedSOHOReplayFidelity,
)
from methods.crt_soho import create_learner as create_crt_learner
from methods.sft_cl import create_learner as create_sft_learner
from tools.experiment_runner import forgetting_from_matrix, split, validate_cache


EXPECTED_MANIFEST_SHA256 = "4dc4740611b7e8dffffd33204b9d7b0ccc77b3d970cd2f8387f8b55a31392d66"
METHODS = (
    "raw_ridge",
    "anchor_only",
    "full_raw_residual",
    "schur_residual",
    "fisher_residual",
    "random_residual",
    "flycl",
    "soho_replay",
)
REFERENCE_METHOD = "flycl"
T_CRITICAL_95_DF4 = 2.7764451051977987


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_dump(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_copy(source: str | Path, destination: str | Path) -> None:
    """Copy a small locked input without ever exposing a partial artifact."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_locked_manifest(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Phase H manifest does not exist: {path}")
    digest = sha256(path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Phase H manifest SHA-256 mismatch: expected "
            f"{EXPECTED_MANIFEST_SHA256}, observed {digest}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    shared = manifest.get("shared_protocol", {})
    rules = manifest.get("stopping_rules", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("study_id") != "phaseh_schur_matched_multiseed_cifar100"
        or manifest.get("status") != "preregistered_not_executed"
        or tuple(manifest.get("methods", {})) != METHODS
        or shared.get("seeds") != [1993, 2025, 3407, 4421, 5501]
        or shared.get("test_time_hyperparameter_search") is not False
        or rules.get("fly_reference_gate_seed") != 1993
        or rules.get("fly_reference_gate_metric") != "average_incremental_accuracy"
        or rules.get("stop_on_fly_discrepancy") is not True
    ):
        raise ValueError("Phase H manifest contract mismatch")
    return manifest, digest


def authorize_phase_g_evidence(zip_path: str | Path, manifest: dict) -> dict:
    """Verify the immutable Phase G result before any new cache is opened."""
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Phase G evidence ZIP does not exist: {zip_path}")
    expected = manifest["phase_g_evidence"]
    observed_zip_sha = sha256(zip_path)
    if observed_zip_sha != expected["heldout_zip_sha256"]:
        raise ValueError("Phase G evidence ZIP SHA-256 mismatch")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Phase G evidence ZIP is corrupt")
        names = archive.namelist()

        def exactly_one(suffix: str) -> str:
            matches = [name for name in names if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Phase G ZIP must contain one {suffix}")
            return matches[0]

        result = json.loads(archive.read(exactly_one("heldout_results.json")))
        lock = json.loads(archive.read(exactly_one("locked_manifest.json")))
        gate_bytes = archive.read(exactly_one("authorized_gate_results.json"))
    gate_sha = hashlib.sha256(gate_bytes).hexdigest()
    if (
        gate_sha != expected["gate_results_sha256"]
        or result.get("lock") != lock
        or lock.get("gate_results_sha256") != gate_sha
        or result.get("hyperparameter_search_performed") is not False
        or result.get("test_cache_opened") is not True
        or result.get("full_training_total_count") != 50000
    ):
        raise ValueError("Phase G evidence lock mismatch")
    proposal = lock.get("selected_proposal", {})
    full = lock.get("selected_full_raw_residual", {})
    raw = lock.get("selected_raw_ridge", {})
    methods = manifest["methods"]
    if (
        proposal.get("method") != "schur_residual"
        or proposal.get("rank") != methods["schur_residual"]["rank"]
        or proposal.get("anchor_ridge") != methods["schur_residual"]["anchor_ridge"]
        or proposal.get("residual_ridge") != methods["schur_residual"]["residual_ridge"]
        or proposal.get("complement_ridge") != methods["schur_residual"]["complement_ridge"]
        or full.get("anchor_ridge") != methods["full_raw_residual"]["anchor_ridge"]
        or raw.get("ridge_lambda") != methods["raw_ridge"]["ridge_lambda"]
    ):
        raise ValueError("Phase G selected configuration differs from Phase H lock")
    return {
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": observed_zip_sha,
        "gate_results_sha256": gate_sha,
        "phase_g_environment": result.get("environment"),
        "source_gate_cache": lock.get("source_gate_cache"),
    }


def _load_exact_baseline_config(path: Path, expected: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != expected:
        raise ValueError(f"baseline config mismatch: {path}")
    return payload


def load_baseline_configs(manifest: dict) -> dict:
    fly_expected = {
        "method": "cached_flycl_fidelity", "fly_expand_dim": 10000,
        "fly_synaptic_degree": 300, "fly_coding_level": 0.3,
        "fly_ridge_lower": 6, "fly_ridge_upper": 10, "seed": 1993,
        "dataset": "CIFAR-100", "model_name": "vit_base_patch16_224",
        "num_classes": 100, "num_tasks": 10, "device": "cuda",
    }
    soho_expected = {
        "method": "cached_soho_replay_fidelity", "soho_expand_dim": 10000,
        "soho_density": 0.1, "soho_olda_dim": 768,
        "soho_coding_level": 0.25, "soho_no_etf": False,
        "soho_ridge_lower": -2, "soho_ridge_upper": 10,
        "soho_replay_chunk_size": 2000, "soho_gcv_sample_size": 3000,
        "seed": 1993, "dataset": "CIFAR-100",
        "model_name": "vit_base_patch16_224", "num_classes": 100,
        "num_tasks": 10, "device": "cuda",
    }
    return {
        "flycl": _load_exact_baseline_config(
            ROOT / manifest["methods"]["flycl"]["config"], fly_expected
        ),
        "soho_replay": _load_exact_baseline_config(
            ROOT / manifest["methods"]["soho_replay"]["config"], soho_expected
        ),
    }


def validate_study_cache(cache_dir: str | Path, manifest: dict, evidence: dict):
    shared = manifest["shared_protocol"]
    args = SimpleNamespace(
        dataset=shared["dataset"], model_name=shared["backbone"]
    )
    train, test, metadata = validate_cache(cache_dir, args, load_test=True)
    expected_metadata = {
        "dataset": shared["dataset"],
        "backbone_model": shared["backbone"],
        "checkpoint_sha256": shared["checkpoint_sha256"],
        "preprocessing": shared["preprocessing"],
        "feature_dim": shared["feature_dim"],
        "finite": True,
        "train_shape": [50000, shared["feature_dim"]],
        "test_shape": [10000, shared["feature_dim"]],
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Phase H feature-cache mismatch for {key}")
    train_path = Path(cache_dir) / "train.pt"
    source_train = evidence["source_gate_cache"]["source_train"]
    if (
        train_path.stat().st_size != source_train["bytes"]
        or sha256(train_path) != source_train["sha256"]
    ):
        raise ValueError("Phase H train cache differs from Phase G source")
    return train, test, metadata


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_learner(method: str, manifest: dict, baselines: dict, seed: int,
                   raw_dim: int, device: str):
    methods = manifest["methods"]
    dtype = torch.float32
    if method == "raw_ridge":
        return create_sft_learner(
            method="raw_ridge", feature_dim=raw_dim,
            ridge_lambda=methods[method]["ridge_lambda"], requested_rank=1,
            seed=seed, device=device, dtype=dtype,
        )
    if method in {
        "anchor_only", "full_raw_residual", "schur_residual",
        "fisher_residual", "random_residual",
    }:
        anchor = methods["anchor_only"]
        config = methods[method]
        return create_crt_learner(
            method=method, raw_dim=raw_dim, anchor_dim=anchor["anchor_dim"],
            synaptic_degree=anchor["synaptic_degree"],
            coding_level=anchor["coding_level"],
            anchor_ridge=config["anchor_ridge"],
            residual_ridge=config.get("residual_ridge", 1.0),
            complement_ridge=config.get("complement_ridge", 1.0),
            requested_rank=(
                raw_dim if method == "full_raw_residual"
                else config.get("rank", 1)
            ),
            seed=seed, device=device, dtype=dtype,
        )
    if method == "flycl":
        config = baselines[method]
        return CachedFlyCLFidelity(
            raw_dim, config["fly_expand_dim"], config["fly_synaptic_degree"],
            config["fly_coding_level"], manifest["shared_protocol"]["num_classes"],
            config["fly_ridge_lower"], config["fly_ridge_upper"],
            seed=seed, device=device, dtype=dtype,
        )
    if method == "soho_replay":
        config = baselines[method]
        return CachedSOHOReplayFidelity(
            raw_dim, config["soho_expand_dim"], config["soho_density"],
            config["soho_olda_dim"], not config["soho_no_etf"],
            config["soho_coding_level"], manifest["shared_protocol"]["num_classes"],
            config["soho_ridge_lower"], config["soho_ridge_upper"],
            seed=seed, device=device, dtype=dtype,
            replay_chunk_size=config["soho_replay_chunk_size"],
            gcv_sample_size=config["soho_gcv_sample_size"],
        )
    raise ValueError(f"unsupported Phase H method: {method}")


def _metric_summary(matrix: list[list[float]]) -> dict:
    stage_means = [sum(row) / len(row) for row in matrix]
    return {
        "accuracy_matrix": matrix,
        "accuracy_after_each_task": stage_means,
        "final_accuracy": sum(matrix[-1]) / len(matrix[-1]),
        "average_incremental_accuracy": sum(stage_means) / len(stage_means),
        "forgetting": forgetting_from_matrix(matrix),
    }


class Progress:
    def __init__(self, total_units: int, already_complete: int = 0,
                 device: str = "cpu"):
        self.total_units = total_units
        self.completed_units = already_complete
        self.device = device
        self.started = time.perf_counter()
        self.unit_seconds: list[float] = []

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds):
            return "?"
        seconds = max(int(seconds), 0)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def task(self, seed_index, seed_total, method_index, method_total,
             task_index, task_total, unit_started):
        elapsed = time.perf_counter() - unit_started
        remaining_tasks = task_total - task_index
        unit_eta = elapsed / task_index * remaining_tasks if task_index else None
        mean_unit = statistics.fmean(self.unit_seconds) if self.unit_seconds else None
        remaining_units = self.total_units - self.completed_units - 1
        study_eta = None if mean_unit is None else mean_unit * remaining_units + (unit_eta or 0)
        print(
            f"[seed {seed_index}/{seed_total} | method {method_index}/{method_total} "
            f"| task {task_index}/{task_total}] elapsed={self._duration(elapsed)} "
            f"unit_eta={self._duration(unit_eta)} study_eta={self._duration(study_eta)}",
            flush=True,
        )

    def complete(self, seed, method, result, seconds):
        self.completed_units += 1
        self.unit_seconds.append(seconds)
        print(
            f"[done {self.completed_units}/{self.total_units}] seed={seed} "
            f"method={method} AA={result['average_incremental_accuracy']:.4f} "
            f"final={result['final_accuracy']:.4f} time={self._duration(seconds)}",
            flush=True,
        )


def evaluate_method(method: str, manifest: dict, baselines: dict, seed: int,
                    train: dict, test: dict, train_indices, test_indices,
                    progress: Progress, seed_index: int, method_index: int) -> dict:
    # The cache remains on CPU; learner/device is explicit in the manifest run.
    learner_device = progress.device
    learner = _build_learner(
        method, manifest, baselines, seed, int(train["features"].shape[1]),
        learner_device,
    )
    torch_device = torch.device(learner_device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    matrix, update_seconds, inference_seconds, diagnostics = [], [], [], []
    unit_started = time.perf_counter()
    for task in range(manifest["shared_protocol"]["num_tasks"]):
        _sync(torch_device)
        started = time.perf_counter()
        learner.update(
            train["features"][train_indices[task]], train["labels"][train_indices[task]]
        )
        _sync(torch_device)
        update_seconds.append(time.perf_counter() - started)
        row = []
        for previous in range(task + 1):
            _sync(torch_device)
            started = time.perf_counter()
            predictions = learner.predict(test["features"][test_indices[previous]])
            _sync(torch_device)
            inference_seconds.append(time.perf_counter() - started)
            labels = test["labels"][test_indices[previous]].cpu()
            row.append(float((predictions.cpu() == labels).float().mean().item() * 100))
        matrix.append(row)
        diagnostics.append({
            "task": task,
            "selected_ridge": learner.diagnostics.get("selected_ridge"),
            "ridge_policy": learner.diagnostics.get("ridge_policy"),
            "effective_rank": learner.diagnostics.get("effective_rank"),
            "retained_sample_count": learner.diagnostics.get("retained_sample_count"),
            "retained_correction_energy": learner.diagnostics.get("retained_correction_energy"),
            "solver_relative_residual_max": learner.diagnostics.get("solver_relative_residual_max"),
        })
        progress.task(
            seed_index, len(manifest["shared_protocol"]["seeds"]), method_index,
            len(METHODS), task + 1, manifest["shared_protocol"]["num_tasks"],
            unit_started,
        )
    result = {
        "seed": seed,
        "method": method,
        **_metric_summary(matrix),
        "persistent_state_bytes": learner.persistent_state_bytes(),
        "exemplar_free": bool(getattr(learner, "is_exemplar_free", True)),
        "total_update_seconds": sum(update_seconds),
        "total_inference_seconds": sum(inference_seconds),
        "peak_runtime_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch_device))
            if torch_device.type == "cuda" else None
        ),
        "diagnostics_by_task": diagnostics,
    }
    del learner
    gc.collect()
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _result_path(output_dir: Path, seed: int, method: str) -> Path:
    return output_dir / "runs" / f"seed_{seed}" / method / "result.json"


def _valid_completed_result(path: Path, seed: int, method: str,
                            manifest_sha: str, cache_sha: str) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = (
        "final_accuracy", "average_incremental_accuracy", "forgetting",
        "persistent_state_bytes", "total_update_seconds", "total_inference_seconds",
    )
    expected_order = random.Random(seed).sample(list(range(100)), 100)
    matrix = payload.get("accuracy_matrix")
    stage_accuracies = payload.get("accuracy_after_each_task")
    matrix_is_complete = (
        isinstance(matrix, list) and len(matrix) == 10
        and all(isinstance(row, list) and len(row) == stage + 1
                for stage, row in enumerate(matrix))
        and all(_finite_number(value) for row in matrix for value in row)
    )
    if (
        payload.get("seed") != seed or payload.get("method") != method
        or payload.get("manifest_sha256") != manifest_sha
        or payload.get("train_cache_sha256") != cache_sha
        or payload.get("completed") is not True
        or payload.get("class_order") != expected_order
        or payload.get("exemplar_free") is not (method != "soho_replay")
        or not matrix_is_complete
        or not isinstance(stage_accuracies, list) or len(stage_accuracies) != 10
        or not all(_finite_number(value) for value in stage_accuracies)
        or not all(_finite_number(payload.get(field)) for field in metrics)
    ):
        raise ValueError(f"invalid resume result: {path}")
    return payload


def _aggregate(results: list[dict], manifest: dict) -> dict:
    by_method = {method: [] for method in METHODS}
    for result in results:
        by_method[result["method"]].append(result)
    seeds = manifest["shared_protocol"]["seeds"]
    if any(sorted(item["seed"] for item in values) != sorted(seeds) for values in by_method.values()):
        raise ValueError("cannot aggregate an incomplete seed/method grid")
    summaries = []
    for method, values in by_method.items():
        row = {"method": method, "seeds": seeds}
        for metric in (
            "final_accuracy", "average_incremental_accuracy", "forgetting",
            "persistent_state_bytes", "total_update_seconds", "total_inference_seconds",
            "peak_runtime_memory_bytes",
        ):
            raw_samples = [item.get(metric) for item in values]
            if all(_finite_number(value) for value in raw_samples):
                samples = [float(value) for value in raw_samples]
                row[f"{metric}_mean"] = statistics.fmean(samples)
                row[f"{metric}_std"] = statistics.stdev(samples)
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
        row["exemplar_free"] = all(item["exemplar_free"] for item in values)
        summaries.append(row)
    indexed = {(item["seed"], item["method"]): item for item in results}
    paired = []
    for expression in manifest["reporting"]["paired_differences"]:
        left, right = expression.split("-", 1)
        for metric in ("average_incremental_accuracy", "final_accuracy", "forgetting"):
            differences = [
                indexed[(seed, left)][metric] - indexed[(seed, right)][metric]
                for seed in seeds
            ]
            mean = statistics.fmean(differences)
            std = statistics.stdev(differences)
            half_width = T_CRITICAL_95_DF4 * std / math.sqrt(len(differences))
            paired.append({
                "comparison": expression, "metric": metric,
                "differences_by_seed": dict(zip(map(str, seeds), differences)),
                "mean": mean, "std": std,
                "confidence_interval_95": [mean - half_width, mean + half_width],
            })
    return {"method_summaries": summaries, "paired_differences": paired}


def run(args) -> dict:
    manifest, manifest_sha = load_locked_manifest(args.manifest)
    evidence = authorize_phase_g_evidence(args.phase_g_evidence_zip, manifest)
    baselines = load_baseline_configs(manifest)
    train, test, cache_metadata = validate_study_cache(
        args.feature_cache_dir, manifest, evidence
    )
    train_sha = sha256(Path(args.feature_cache_dir) / "train.pt")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_identity = {
        "manifest_sha256": manifest_sha,
        "phase_g_evidence": evidence,
        "train_cache_sha256": train_sha,
        "feature_cache_metadata": cache_metadata,
        "device": args.device,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    identity_path = output_dir / "study_identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != study_identity:
            raise ValueError("resume study identity mismatch")
    else:
        _atomic_dump(identity_path, study_identity)
    locked_manifest_copy = output_dir / "locked_phaseh_manifest.json"
    locked_evidence_copy = output_dir / "phaseg_evidence.zip"
    if locked_manifest_copy.is_file() and sha256(locked_manifest_copy) != manifest_sha:
        raise ValueError("stored Phase H manifest copy mismatch")
    if locked_evidence_copy.is_file() and sha256(locked_evidence_copy) != evidence["zip_sha256"]:
        raise ValueError("stored Phase G evidence copy mismatch")
    if not locked_manifest_copy.is_file():
        _atomic_copy(args.manifest, locked_manifest_copy)
    if not locked_evidence_copy.is_file():
        _atomic_copy(args.phase_g_evidence_zip, locked_evidence_copy)

    seeds = manifest["shared_protocol"]["seeds"]
    total_units = len(seeds) * len(METHODS)
    completed = []
    for seed in seeds:
        for method in METHODS:
            result = _valid_completed_result(
                _result_path(output_dir, seed, method), seed, method,
                manifest_sha, train_sha,
            )
            if result is not None:
                completed.append(result)
    progress = Progress(total_units, len(completed), args.device)
    if completed:
        print(f"[resume] validated {len(completed)}/{total_units} completed units", flush=True)

    reference_seed = manifest["stopping_rules"]["fly_reference_gate_seed"]
    work = [(reference_seed, REFERENCE_METHOD)] + [
        (seed, method) for seed in seeds for method in METHODS
        if (seed, method) != (reference_seed, REFERENCE_METHOD)
    ]
    indexed_results = {(item["seed"], item["method"]): item for item in completed}
    for seed, method in work:
        seed_index = seeds.index(seed) + 1
        method_index = METHODS.index(method) + 1
        if (seed, method) not in indexed_results:
            order = random.Random(seed).sample(
                list(range(manifest["shared_protocol"]["num_classes"])),
                manifest["shared_protocol"]["num_classes"],
            )
            train_indices = split(train["labels"], order, manifest["shared_protocol"]["num_tasks"])
            test_indices = split(test["labels"], order, manifest["shared_protocol"]["num_tasks"])
            unit_started = time.perf_counter()
            result = evaluate_method(
                method, manifest, baselines, seed, train, test,
                train_indices, test_indices, progress, seed_index, method_index,
            )
            result.update(
                completed=True, manifest_sha256=manifest_sha,
                train_cache_sha256=train_sha, class_order=order,
                elapsed_seconds=time.perf_counter() - unit_started,
            )
            _atomic_dump(_result_path(output_dir, seed, method), result)
            indexed_results[(seed, method)] = result
            progress.complete(seed, method, result, result["elapsed_seconds"])
        if (seed, method) == (reference_seed, REFERENCE_METHOD):
            result = indexed_results[(seed, method)]
            rules = manifest["stopping_rules"]
            difference = abs(
                result[rules["fly_reference_gate_metric"]]
                - rules["fly_reference_average_incremental_accuracy"]
            )
            gate = {
                "seed": seed, "method": method,
                "metric": rules["fly_reference_gate_metric"],
                "observed": result[rules["fly_reference_gate_metric"]],
                "reference": rules["fly_reference_average_incremental_accuracy"],
                "absolute_difference_percentage_points": difference,
                "tolerance_percentage_points": rules["fly_reference_tolerance_percentage_points"],
                "pass": difference <= rules["fly_reference_tolerance_percentage_points"],
            }
            _atomic_dump(output_dir / "fly_reference_gate.json", gate)
            print(
                f"[FLY gate] observed={gate['observed']:.4f} "
                f"reference={gate['reference']:.4f} diff={difference:.4f} "
                f"status={'PASS' if gate['pass'] else 'FAIL'}",
                flush=True,
            )
            if not gate["pass"]:
                stopped = {
                    "status": "stopped_fly_reference_discrepancy",
                    "fly_reference_gate": gate,
                    "completed_units": len(indexed_results),
                    "test_cache_opened": True,
                    "hyperparameter_search_performed": False,
                }
                _atomic_dump(output_dir / "STOPPED_FLY_DISCREPANCY.json", stopped)
                return stopped

    results = [indexed_results[(seed, method)] for seed in seeds for method in METHODS]
    aggregate = _aggregate(results, manifest)
    final = {
        "status": "complete",
        "manifest_sha256": manifest_sha,
        "phase_g_evidence": evidence,
        "test_cache_opened": True,
        "hyperparameter_search_performed": False,
        "seeds": seeds,
        "methods": list(METHODS),
        **aggregate,
    }
    _atomic_dump(output_dir / "phaseh_summary.json", final)
    print("[complete] Phase H-B 5-seed grid finished", flush=True)
    return final


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=str(ROOT / "configs" / "phaseh_cifar100_multiseed.json")
    )
    parser.add_argument("--phase-g-evidence-zip", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
