"""Repeated, isolated Priority-2A benchmark for chunked blocked SRQ updates.

This study is synthetic and never accesses a dataset cache.  Every measured
method/repetition runs in a fresh process.  Chunk sizes are fixed in the input
config, and selection minimizes median peak CUDA allocation subject to locked
speed, numerical, state, and predictor-fidelity gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import srq_fly_system_benchmark as system_benchmark
from tools import srq_fly_update_benchmark as update_benchmark


BASELINES = {
    "exact_fly": "exact_fly_dense",
    "unchunked_blocked_qr": "optimized_blocked_qr_srq_int8",
}
WORKER_FIELDS = {
    "study", "seed", "feature_dim", "expand_dim", "synaptic_degree",
    "coding_level", "ridge_lambda", "block_size", "group_size",
    "update_panel_size", "update_trailing_chunk_size", "num_tasks",
    "rows_per_task", "num_classes", "probe_rows", "solver_tolerance",
    "maximum_relative_logit_drift", "maximum_update_ratio_to_exact_fly",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    update_benchmark._validate(
        {key: value for key, value in config.items() if key in WORKER_FIELDS}
    )
    required = {
        "schema_version", "study", "warmup_repetitions", "measured_repetitions",
        "trailing_chunk_sizes", "maximum_median_update_ratio_to_exact",
        "maximum_median_peak_allocated_ratio_to_exact",
    }
    if set(config) - (WORKER_FIELDS | {
        "schema_version",
        "warmup_repetitions", "measured_repetitions", "trailing_chunk_sizes",
        "maximum_median_update_ratio_to_exact",
        "maximum_median_peak_allocated_ratio_to_exact",
    }):
        raise ValueError("Priority-2A config contains unknown fields")
    missing = required - set(config)
    if missing:
        raise ValueError(f"Priority-2A config missing fields: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("unsupported Priority-2A schema")
    if int(config["warmup_repetitions"]) < 0 or int(config["measured_repetitions"]) < 3:
        raise ValueError("Priority-2A requires at least three measured repetitions")
    chunks = [int(value) for value in config["trailing_chunk_sizes"]]
    if not chunks or len(chunks) != len(set(chunks)) or any(value <= 0 for value in chunks):
        raise ValueError("trailing chunk sizes must be unique positive integers")
    if chunks != sorted(chunks):
        raise ValueError("trailing chunk sizes must be sorted")
    if float(config["maximum_median_update_ratio_to_exact"]) <= 0:
        raise ValueError("invalid update-ratio gate")
    if float(config["maximum_median_peak_allocated_ratio_to_exact"]) <= 0:
        raise ValueError("invalid peak-memory gate")
    return config


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _statistics(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty measurement list")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
        "minimum": min(values),
        "maximum": max(values),
    }


def _candidate_label(chunk_size: int) -> str:
    return f"chunked_blocked_qr_{chunk_size}"


def _write_candidate_config(base: dict, chunk_size: int, path: Path) -> None:
    payload = dict(base)
    payload["study"] = f"{base['study']}-chunk-{chunk_size}"
    payload["update_trailing_chunk_size"] = int(chunk_size)
    # The shared worker validates only its update-level contract.
    for key in (
        "schema_version", "warmup_repetitions", "measured_repetitions",
        "trailing_chunk_sizes", "maximum_median_update_ratio_to_exact",
        "maximum_median_peak_allocated_ratio_to_exact",
    ):
        payload.pop(key, None)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_one(
    *, worker_config: Path, worker_method: str, label: str, repetition: int,
    output_dir: Path, device: str, profile_stages: bool,
) -> tuple[dict, torch.Tensor]:
    stem = f"rep_{repetition:02d}_{label}"
    result_path = output_dir / f"{stem}.json"
    probe_path = output_dir / f"{stem}.probe.pt"
    expected_source = {
        "runner": _sha256(ROOT / "tools/srq_fly_system_benchmark.py"),
        "optimized_learner": _sha256(
            ROOT / "methods/srq_fly_optimized/learner.py"
        ),
        "optimized_storage": _sha256(
            ROOT / "methods/srq_fly_optimized/storage.py"
        ),
    }
    if result_path.is_file() and probe_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        profile_matches = (
            cached.get("profiled_task_stage_seconds") is not None
        ) == profile_stages
        if (
            cached.get("status") == "complete"
            and cached.get("method") == worker_method
            and cached.get("config_sha256") == _sha256(worker_config)
            and cached.get("source_identity") == expected_source
            and profile_matches
        ):
            print(
                f"RESUME repetition={repetition} label={label}", flush=True
            )
            return cached, torch.load(
                probe_path, weights_only=True, map_location="cpu"
            )
    command = [
        sys.executable, "-u", str(ROOT / "tools/srq_fly_system_benchmark.py"),
        "worker", "--config", str(worker_config), "--method", worker_method,
        "--output", str(result_path), "--probe-output", str(probe_path),
        "--device", device,
    ]
    if not profile_stages:
        command.append("--skip-stage-profile")
    print(
        f"START repetition={repetition} label={label} profile={profile_stages}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Priority-2A worker failed: {label} repetition {repetition}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    probe = torch.load(probe_path, weights_only=True, map_location="cpu")
    print(
        f"DONE repetition={repetition} label={label} "
        f"update={result['total_update_seconds']:.4f}s "
        f"peak={result['peak_cuda_allocated_bytes']}",
        flush=True,
    )
    return result, probe


def run(
    *, config_path: Path, output_dir: Path, device: str,
    require_clean_git: bool = False,
) -> dict:
    config_path = config_path.resolve()
    config = _read_config(config_path)
    if device != "cuda":
        raise ValueError("Priority-2A peak-memory selection requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if require_clean_git and dirty:
        raise RuntimeError(f"repository must be clean before Priority-2A:\n{dirty}")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "locked_candidate_configs"
    config_dir.mkdir(exist_ok=True)
    base_worker_config = config_dir / "base.json"
    base_payload = dict(config)
    for key in (
        "schema_version", "warmup_repetitions", "measured_repetitions",
        "trailing_chunk_sizes", "maximum_median_update_ratio_to_exact",
        "maximum_median_peak_allocated_ratio_to_exact",
    ):
        base_payload.pop(key, None)
    base_worker_config.write_text(json.dumps(base_payload, indent=2), encoding="utf-8")
    candidate_configs = {}
    for chunk_size in map(int, config["trailing_chunk_sizes"]):
        candidate_path = config_dir / f"chunk_{chunk_size}.json"
        _write_candidate_config(config, chunk_size, candidate_path)
        candidate_configs[chunk_size] = candidate_path

    labels = list(BASELINES) + [
        _candidate_label(size) for size in map(int, config["trailing_chunk_sizes"])
    ]
    total_repetitions = int(config["warmup_repetitions"]) + int(
        config["measured_repetitions"]
    )
    measured: dict[str, list[dict]] = {label: [] for label in labels}
    probes: dict[int, dict[str, torch.Tensor]] = {}
    for repetition in range(total_repetitions):
        is_warmup = repetition < int(config["warmup_repetitions"])
        # Rotate the order to reduce systematic thermal/clock bias.
        shift = repetition % len(labels)
        ordered = labels[shift:] + labels[:shift]
        current_probes = {}
        for label in ordered:
            if label == "exact_fly":
                worker_method = BASELINES[label]
                worker_config = base_worker_config
            elif label == "unchunked_blocked_qr":
                worker_method = BASELINES[label]
                worker_config = base_worker_config
            else:
                chunk_size = int(label.rsplit("_", 1)[1])
                worker_method = "optimized_chunked_blocked_qr_srq_int8"
                worker_config = candidate_configs[chunk_size]
            result, probe = _run_one(
                worker_config=worker_config,
                worker_method=worker_method,
                label=label,
                repetition=repetition,
                output_dir=output_dir,
                device=device,
                profile_stages=(not is_warmup and repetition == int(config["warmup_repetitions"])),
            )
            current_probes[label] = probe
            if not is_warmup:
                measured[label].append(result)
        if not is_warmup:
            probes[repetition] = current_probes

    reference_state = measured["unchunked_blocked_qr"][0]["persistent_state_bytes"]
    tolerance = float(config["maximum_relative_logit_drift"])
    solver_tolerance = float(config["solver_tolerance"])
    summaries = {}
    selected_candidates = []
    for label in labels:
        rows = measured[label]
        times = [float(row["total_update_seconds"]) for row in rows]
        peaks = [float(row["peak_cuda_allocated_bytes"]) for row in rows]
        reserved = [float(row["peak_cuda_reserved_bytes"]) for row in rows]
        summary = {
            "label": label,
            "worker_method": rows[0]["method"],
            "update_seconds": _statistics(times),
            "peak_allocated_bytes": _statistics(peaks),
            "peak_reserved_bytes": _statistics(reserved),
            "persistent_state_bytes": int(rows[0]["persistent_state_bytes"]),
            "serialized_checkpoint_bytes": int(rows[0]["serialized_checkpoint_bytes"]),
            "maximum_solver_relative_residual": max(
                float(row["solver_relative_residual"]) for row in rows
            ),
            "profiled_task_stage_seconds": next(
                (row["profiled_task_stage_seconds"] for row in rows
                 if row["profiled_task_stage_seconds"] is not None),
                None,
            ),
            "profiled_task_stage_cuda_memory": next(
                (row["profiled_task_stage_cuda_memory"] for row in rows
                 if row["profiled_task_stage_cuda_memory"] is not None),
                None,
            ),
        }
        if label.startswith("chunked_blocked_qr_"):
            chunk_size = int(label.rsplit("_", 1)[1])
            paired_time_ratios = []
            paired_peak_ratios = []
            drifts = []
            for offset, repetition in enumerate(sorted(probes)):
                exact = measured["exact_fly"][offset]
                paired_time_ratios.append(
                    float(rows[offset]["total_update_seconds"])
                    / float(exact["total_update_seconds"])
                )
                paired_peak_ratios.append(
                    float(rows[offset]["peak_cuda_allocated_bytes"])
                    / float(exact["peak_cuda_allocated_bytes"])
                )
                reference = probes[repetition]["unchunked_blocked_qr"]
                denominator = max(float(torch.linalg.vector_norm(reference)), 1.0)
                drifts.append(
                    float(torch.linalg.vector_norm(probes[repetition][label] - reference))
                    / denominator
                )
            gates = {
                "predictor_within_tolerance": max(drifts) <= tolerance,
                "persistent_state_unchanged": all(
                    int(row["persistent_state_bytes"]) == reference_state for row in rows
                ),
                "solver_stable": summary["maximum_solver_relative_residual"]
                <= solver_tolerance,
                "median_update_ratio_to_exact": statistics.median(paired_time_ratios)
                <= float(config["maximum_median_update_ratio_to_exact"]),
                "median_peak_allocated_ratio_to_exact": statistics.median(
                    paired_peak_ratios
                ) <= float(config["maximum_median_peak_allocated_ratio_to_exact"]),
            }
            summary.update(
                chunk_size=chunk_size,
                paired_update_ratio_to_exact=_statistics(paired_time_ratios),
                paired_peak_allocated_ratio_to_exact=_statistics(paired_peak_ratios),
                maximum_relative_logit_drift_from_unchunked=max(drifts),
                gates=gates,
            )
            if all(gates.values()):
                selected_candidates.append(summary)
        summaries[label] = summary

    selected = None
    if selected_candidates:
        selected = min(
            selected_candidates,
            key=lambda row: (
                row["paired_peak_allocated_ratio_to_exact"]["median"],
                row["paired_update_ratio_to_exact"]["median"],
                row["chunk_size"],
            ),
        )
    status = "PASS_REVIEW_PRIORITY2A" if selected is not None else "STOP_MEMORY_GATE"
    result = {
        "schema_version": 1,
        "study": config["study"],
        "status": status,
        "uses_test_set": False,
        "synthetic_only": True,
        "config_sha256": _sha256(config_path),
        "source_identity": {
            "runner": _sha256(Path(__file__).resolve()),
            "system_worker": _sha256(ROOT / "tools/srq_fly_system_benchmark.py"),
            "optimized_learner": _sha256(ROOT / "methods/srq_fly_optimized/learner.py"),
            "optimized_storage": _sha256(ROOT / "methods/srq_fly_optimized/storage.py"),
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(dirty),
        "device": device,
        "warmup_repetitions": int(config["warmup_repetitions"]),
        "measured_repetitions": int(config["measured_repetitions"]),
        "selection_rule": (
            "minimum median paired peak allocated ratio among candidates passing "
            "locked speed, peak, fidelity, state, and solver gates"
        ),
        "summaries": [summaries[label] for label in labels],
        "selected_candidate": None if selected is None else {
            "label": selected["label"],
            "chunk_size": selected["chunk_size"],
            "paired_update_ratio_to_exact": selected[
                "paired_update_ratio_to_exact"
            ],
            "paired_peak_allocated_ratio_to_exact": selected[
                "paired_peak_allocated_ratio_to_exact"
            ],
            "maximum_relative_logit_drift_from_unchunked": selected[
                "maximum_relative_logit_drift_from_unchunked"
            ],
        },
    }
    output_path = output_dir / "priority2a_memory_results.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
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
