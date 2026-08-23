"""Synthetic correctness/timing gate for opt-in SRQ-FLY update backends."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly import SquareRootFLYLearner as LockedLearner
from methods.srq_fly_optimized import SquareRootFLYLearner as OptimizedLearner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate(config: dict) -> None:
    required = {
        "study", "seed", "feature_dim", "expand_dim", "synaptic_degree",
        "coding_level", "ridge_lambda", "block_size", "group_size",
        "num_tasks", "rows_per_task", "num_classes", "probe_rows",
        "solver_tolerance", "maximum_relative_logit_drift",
    }
    if set(config) != required:
        raise ValueError(f"optimization config fields mismatch: {sorted(set(config) ^ required)}")
    if config["seed"] != 2025:
        raise ValueError("new SRQ optimization protocols must use seed 2025")
    if min(
        config["feature_dim"], config["expand_dim"], config["synaptic_degree"],
        config["block_size"], config["group_size"], config["num_tasks"],
        config["rows_per_task"], config["num_classes"], config["probe_rows"],
    ) <= 0:
        raise ValueError("dimensions and counts must be positive")
    if config["synaptic_degree"] > config["feature_dim"]:
        raise ValueError("synaptic degree exceeds feature dimension")
    if not 0 < config["coding_level"] <= 1:
        raise ValueError("coding level must be in (0, 1]")
    if config["ridge_lambda"] <= 0 or config["solver_tolerance"] <= 0:
        raise ValueError("ridge and tolerance must be positive")


def _codes(config: dict, generator: torch.Generator) -> torch.Tensor:
    rows, dimension = config["rows_per_task"], config["expand_dim"]
    active = max(1, int(config["coding_level"] * dimension))
    dense = torch.zeros(rows, dimension, dtype=torch.float32)
    scores = torch.randn(rows, dimension, generator=generator)
    indices = scores.topk(active, dim=1).indices
    values = torch.randn(rows, active, generator=generator)
    dense.scatter_(1, indices, values)
    return dense


def _learner_kwargs(config: dict, device: torch.device) -> dict:
    return {
        "feature_dim": config["feature_dim"],
        "expand_dim": config["expand_dim"],
        "synaptic_degree": config["synaptic_degree"],
        "coding_level": config["coding_level"],
        "ridge_lambda": config["ridge_lambda"],
        "block_size": config["block_size"],
        "group_size": config["group_size"],
        "seed": config["seed"],
        "device": device,
        "statistics_dtype": torch.float32,
        "solver_dtype": torch.float32,
    }


def _timed_update(learner, codes: torch.Tensor, labels: torch.Tensor, device: torch.device) -> float:
    _sync(device)
    started = time.perf_counter()
    learner.update_codes(codes, labels)
    _sync(device)
    return time.perf_counter() - started


def run(config_path: Path, output: Path, device_name: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate(config)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    kwargs = _learner_kwargs(config, device)
    locked = LockedLearner(storage_mode="int8", **kwargs)
    optimized = OptimizedLearner(
        storage_mode="int8", update_backend="gram_cholesky",
        profile_updates=False, **kwargs,
    )
    qr = OptimizedLearner(
        storage_mode="int8", update_backend="stacked_qr",
        profile_updates=False, **kwargs,
    )
    generator = torch.Generator().manual_seed(config["seed"] + 17)
    update_seconds = {"locked": [], "optimized_gram": [], "optimized_qr": []}
    stream = []
    last_codes = None
    for task in range(config["num_tasks"]):
        codes = _codes(config, generator).to(device)
        labels = (
            torch.arange(config["rows_per_task"], device=device)
            + task * config["rows_per_task"]
        ) % config["num_classes"]
        stream.append((codes, labels))
        update_seconds["locked"].append(_timed_update(locked, codes, labels, device))
        update_seconds["optimized_gram"].append(
            _timed_update(optimized, codes, labels, device)
        )
        update_seconds["optimized_qr"].append(_timed_update(qr, codes, labels, device))
        last_codes = codes
        print(
            f"TASK {task + 1}/{config['num_tasks']} "
            f"locked={update_seconds['locked'][-1]:.4f}s "
            f"optimized_gram={update_seconds['optimized_gram'][-1]:.4f}s "
            f"optimized_qr={update_seconds['optimized_qr'][-1]:.4f}s",
            flush=True,
        )
    if last_codes is None:
        raise RuntimeError("empty optimization stream")
    probe = last_codes[: config["probe_rows"]]
    locked_logits = locked.predict_logits_from_codes(probe)
    optimized_logits = optimized.predict_logits_from_codes(probe)
    qr_logits = qr.predict_logits_from_codes(probe)

    # Profiling inserts synchronization at every stage, so it must be kept out
    # of the fair end-to-end timings above.  Replay the same synthetic stream
    # in separate learners and report only their final-stage breakdown.
    profiled_gram = OptimizedLearner(
        storage_mode="int8", update_backend="gram_cholesky",
        profile_updates=True, **kwargs,
    )
    profiled_qr = OptimizedLearner(
        storage_mode="int8", update_backend="stacked_qr",
        profile_updates=True, **kwargs,
    )
    for codes, labels in stream:
        profiled_gram.update_codes(codes, labels)
        profiled_qr.update_codes(codes, labels)
    denominator = max(float(torch.linalg.vector_norm(locked_logits).item()), 1.0)
    gram_drift = float(torch.linalg.vector_norm(optimized_logits - locked_logits).item()) / denominator
    qr_drift = float(torch.linalg.vector_norm(qr_logits - locked_logits).item()) / denominator
    gates = {
        "optimized_gram_matches_locked": gram_drift == 0.0,
        "optimized_qr_within_logit_tolerance": qr_drift
        <= config["maximum_relative_logit_drift"],
        "optimized_gram_state_bytes_unchanged": optimized.persistent_state_bytes()
        == locked.persistent_state_bytes(),
        "optimized_qr_state_bytes_unchanged": qr.persistent_state_bytes()
        == locked.persistent_state_bytes(),
        "optimized_gram_solver_stable": optimized.diagnostics["solver_relative_residual"]
        <= config["solver_tolerance"],
        "optimized_qr_solver_stable": qr.diagnostics["solver_relative_residual"]
        <= config["solver_tolerance"],
    }
    core_gate_names = {
        "optimized_gram_matches_locked",
        "optimized_gram_state_bytes_unchanged",
        "optimized_gram_solver_stable",
    }
    qr_gate_names = {
        "optimized_qr_within_logit_tolerance",
        "optimized_qr_state_bytes_unchanged",
        "optimized_qr_solver_stable",
    }
    core_pass = all(gates[name] for name in core_gate_names)
    qr_pass = all(gates[name] for name in qr_gate_names)
    totals = {name: sum(values) for name, values in update_seconds.items()}
    result = {
        "study": config["study"],
        "status": "pass" if core_pass else "fail",
        "backend_decisions": {
            "optimized_gram_eligible_for_dataset_benchmark": core_pass,
            "stacked_qr_eligible_for_dataset_benchmark": qr_pass,
        },
        "uses_test_set": False,
        "synthetic_only": True,
        "config_sha256": _sha256(config_path),
        "source_identity": {
            "runner": _sha256(Path(__file__).resolve()),
            "locked_learner": _sha256(ROOT / "methods/srq_fly/learner.py"),
            "optimized_learner": _sha256(ROOT / "methods/srq_fly_optimized/learner.py"),
            "optimized_storage": _sha256(ROOT / "methods/srq_fly_optimized/storage.py"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "update_seconds": update_seconds,
        "total_update_seconds": totals,
        "speedup_over_locked": {
            "optimized_gram": totals["locked"] / totals["optimized_gram"],
            "optimized_qr": totals["locked"] / totals["optimized_qr"],
        },
        "last_stage_seconds": {
            "optimized_gram": profiled_gram.diagnostics["last_update_stage_seconds"],
            "optimized_qr": profiled_qr.diagnostics["last_update_stage_seconds"],
        },
        "relative_logit_drift": {
            "optimized_gram": gram_drift,
            "optimized_qr": qr_drift,
        },
        "persistent_state_bytes": {
            "locked": locked.persistent_state_bytes(),
            "optimized_gram": optimized.persistent_state_bytes(),
            "optimized_qr": qr.persistent_state_bytes(),
        },
        "gates": gates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise RuntimeError("SRQ update optimization correctness gate failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/srq_fly_update_optimization_smoke.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.device)


if __name__ == "__main__":
    main()
