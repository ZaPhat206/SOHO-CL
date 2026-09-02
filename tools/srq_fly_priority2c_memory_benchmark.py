"""Repeated isolated gate for implicit-Ridge first-update QR.

The study is synthetic-only.  It compares the Priority-2B batch-64 backend
with a backend that initializes sqrt(lambda)*I and applies blocked QR on task
one, avoiding the dense first-task Gram and Cholesky path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import srq_fly_priority2b_memory_benchmark as p2b


WORKER_FIELDS = p2b.WORKER_FIELDS
PROTOCOL_FIELDS = {
    "schema_version", "warmup_repetitions", "measured_repetitions",
    "maximum_median_update_ratio_to_priority2b",
    "maximum_median_peak_ratio_to_priority2b",
}
LABELS = ("exact_fly", "priority2b_batch64", "implicit_ridge_batch64")
METHODS = {
    "exact_fly": "exact_fly_dense",
    "priority2b_batch64": "optimized_streaming_quant_blocked_qr_srq_int8",
    "implicit_ridge_batch64": (
        "optimized_implicit_ridge_streaming_blocked_qr_srq_int8"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = WORKER_FIELDS | PROTOCOL_FIELDS
    if set(config) != required:
        raise ValueError(
            f"Priority-2C config keys mismatch: missing={sorted(required-set(config))}, "
            f"unknown={sorted(set(config)-required)}"
        )
    p2b.update_benchmark._validate(
        {key: value for key, value in config.items() if key in WORKER_FIELDS}
    )
    if config["schema_version"] != 1:
        raise ValueError("unsupported Priority-2C schema")
    if int(config["quantization_batch_blocks"]) != 64:
        raise ValueError("Priority-2C must inherit locked Priority-2B batch 64")
    if int(config["measured_repetitions"]) < 3:
        raise ValueError("Priority-2C requires at least three measured repetitions")
    for field in (
        "maximum_median_update_ratio_to_priority2b",
        "maximum_median_peak_ratio_to_priority2b",
    ):
        if float(config[field]) <= 0:
            raise ValueError(f"invalid gate: {field}")
    return config


def _worker_source() -> dict[str, str]:
    return p2b._expected_worker_source()


def _run_one(
    *, worker_config: Path, label: str, repetition: int, output_dir: Path,
    device: str, profile_stages: bool,
) -> tuple[dict, torch.Tensor]:
    result_path = output_dir / f"rep_{repetition:02d}_{label}.json"
    probe_path = output_dir / f"rep_{repetition:02d}_{label}.probe.pt"
    if result_path.is_file() and probe_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "complete"
            and cached.get("method") == METHODS[label]
            and cached.get("config_sha256") == _sha256(worker_config)
            and cached.get("source_identity") == _worker_source()
            and (cached.get("profiled_task_stage_seconds") is not None)
            == profile_stages
        ):
            print(f"RESUME repetition={repetition} label={label}", flush=True)
            return cached, torch.load(probe_path, weights_only=True, map_location="cpu")
    command = [
        sys.executable, "-u", str(ROOT / "tools/srq_fly_system_benchmark.py"),
        "worker", "--config", str(worker_config), "--method", METHODS[label],
        "--output", str(result_path), "--probe-output", str(probe_path),
        "--device", device,
    ]
    if not profile_stages:
        command.append("--skip-stage-profile")
    print(f"START repetition={repetition} label={label} profile={profile_stages}", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise RuntimeError(f"Priority-2C worker failed: {label} repetition {repetition}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    probe = torch.load(probe_path, weights_only=True, map_location="cpu")
    print(
        f"DONE repetition={repetition} label={label} "
        f"update={result['total_update_seconds']:.4f}s "
        f"peak={result['peak_cuda_allocated_bytes']}", flush=True,
    )
    return result, probe


def run(
    *, config_path: Path, output_dir: Path, device: str,
    require_clean_git: bool = False,
) -> dict:
    config_path = config_path.resolve()
    config = _read_config(config_path)
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Priority-2C requires CUDA")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if require_clean_git and dirty:
        raise RuntimeError(f"repository must be clean before Priority-2C:\n{dirty}")
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_config = output_dir / "locked_worker_config.json"
    worker_config.write_text(
        json.dumps({key: config[key] for key in WORKER_FIELDS}, indent=2),
        encoding="utf-8",
    )
    total = int(config["warmup_repetitions"]) + int(config["measured_repetitions"])
    measured = {label: [] for label in LABELS}
    probes = {}
    for repetition in range(total):
        warmup = repetition < int(config["warmup_repetitions"])
        shift = repetition % len(LABELS)
        ordered = LABELS[shift:] + LABELS[:shift]
        current = {}
        for label in ordered:
            row, probe = _run_one(
                worker_config=worker_config, label=label, repetition=repetition,
                output_dir=output_dir, device=device,
                profile_stages=(not warmup and repetition == int(config["warmup_repetitions"])
                                and label != "exact_fly"),
            )
            current[label] = probe
            if not warmup:
                measured[label].append(row)
        if not warmup:
            probes[repetition] = current

    baseline = measured["priority2b_batch64"]
    candidate = measured["implicit_ridge_batch64"]
    time_ratios, peak_ratios, drifts = [], [], []
    for offset, repetition in enumerate(sorted(probes)):
        time_ratios.append(
            float(candidate[offset]["total_update_seconds"])
            / float(baseline[offset]["total_update_seconds"])
        )
        peak_ratios.append(
            float(candidate[offset]["peak_cuda_allocated_bytes"])
            / float(baseline[offset]["peak_cuda_allocated_bytes"])
        )
        reference = probes[repetition]["priority2b_batch64"]
        denominator = max(float(torch.linalg.vector_norm(reference)), 1.0)
        drifts.append(float(torch.linalg.vector_norm(
            probes[repetition]["implicit_ridge_batch64"] - reference
        )) / denominator)
    state = int(baseline[0]["persistent_state_bytes"])
    gates = {
        "predictor_within_tolerance": max(drifts)
        <= float(config["maximum_relative_logit_drift"]),
        "persistent_state_unchanged": all(
            int(row["persistent_state_bytes"]) == state for row in candidate
        ),
        "solver_stable": max(float(row["solver_relative_residual"]) for row in candidate)
        <= float(config["solver_tolerance"]),
        "median_update_ratio_to_priority2b": statistics.median(time_ratios)
        <= float(config["maximum_median_update_ratio_to_priority2b"]),
        "median_peak_ratio_to_priority2b": statistics.median(peak_ratios)
        <= float(config["maximum_median_peak_ratio_to_priority2b"]),
    }
    summaries = []
    for label in LABELS:
        rows = measured[label]
        summaries.append({
            "label": label,
            "worker_method": METHODS[label],
            "update_seconds": p2b._statistics(
                [float(row["total_update_seconds"]) for row in rows]
            ),
            "peak_allocated_bytes": p2b._statistics(
                [float(row["peak_cuda_allocated_bytes"]) for row in rows]
            ),
            "peak_reserved_bytes": p2b._statistics(
                [float(row["peak_cuda_reserved_bytes"]) for row in rows]
            ),
            "persistent_state_bytes": int(rows[0]["persistent_state_bytes"]),
            "maximum_solver_relative_residual": max(
                float(row["solver_relative_residual"]) for row in rows
            ),
            "profiled_task_stage_seconds": next((
                row["profiled_task_stage_seconds"] for row in rows
                if row["profiled_task_stage_seconds"] is not None
            ), None),
            "profiled_task_stage_cuda_memory": next((
                row["profiled_task_stage_cuda_memory"] for row in rows
                if row["profiled_task_stage_cuda_memory"] is not None
            ), None),
        })
    result = {
        "schema_version": 1,
        "study": config["study"],
        "status": "PASS_REVIEW_PRIORITY2C" if all(gates.values()) else "STOP_PRIORITY2C",
        "uses_test_set": False,
        "synthetic_only": True,
        "config_sha256": _sha256(config_path),
        "source_identity": {
            "runner": _sha256(Path(__file__).resolve()),
            "system_worker": _worker_source()["runner"],
            "optimized_learner": _worker_source()["optimized_learner"],
            "optimized_storage": _worker_source()["optimized_storage"],
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(dirty),
        "locked_priority2b_backend": {
            "quantization_backend": "streaming",
            "quantization_batch_blocks": 64,
            "first_update_backend": "gram_cholesky",
        },
        "candidate_backend": {
            "quantization_backend": "streaming",
            "quantization_batch_blocks": 64,
            "first_update_backend": "implicit_ridge_qr",
        },
        "paired_update_ratio_to_priority2b": p2b._statistics(time_ratios),
        "paired_peak_ratio_to_priority2b": p2b._statistics(peak_ratios),
        "maximum_relative_logit_drift": max(drifts),
        "summaries": summaries,
        "gates": gates,
    }
    (output_dir / "priority2c_memory_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()
    run(
        config_path=args.config,
        output_dir=args.output_dir.resolve(),
        device=args.device,
        require_clean_git=args.require_clean_git,
    )


if __name__ == "__main__":
    main()
