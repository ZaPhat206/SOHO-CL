"""Repeated isolated benchmark for streaming SRQ factor quantization.

Priority 2B is synthetic-only.  It compares the eager Priority-2A encoder with
bounded lazy block batches.  Every method/repetition runs in a fresh process;
the locked rule selects memory first subject to predictor, state, numerical,
and timing gates.
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
    "eager_quant": "optimized_eager_quant_blocked_qr_srq_int8",
}
WORKER_FIELDS = {
    "study", "seed", "feature_dim", "expand_dim", "synaptic_degree",
    "coding_level", "ridge_lambda", "block_size", "group_size",
    "update_panel_size", "quantization_batch_blocks", "num_tasks",
    "rows_per_task", "num_classes", "probe_rows", "solver_tolerance",
    "maximum_relative_logit_drift", "maximum_update_ratio_to_exact_fly",
}
PROTOCOL_FIELDS = {
    "schema_version", "warmup_repetitions", "measured_repetitions",
    "quantization_batch_grid", "maximum_median_update_ratio_to_eager",
    "maximum_median_peak_allocated_ratio_to_exact",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(config) - WORKER_FIELDS - PROTOCOL_FIELDS
    if unknown:
        raise ValueError(f"Priority-2B config contains unknown fields: {sorted(unknown)}")
    required = WORKER_FIELDS - {"quantization_batch_blocks"}
    required |= PROTOCOL_FIELDS
    missing = required - set(config)
    if missing:
        raise ValueError(f"Priority-2B config missing fields: {sorted(missing)}")
    update_benchmark._validate(
        {key: value for key, value in config.items() if key in WORKER_FIELDS}
    )
    if config["schema_version"] != 1:
        raise ValueError("unsupported Priority-2B schema")
    if int(config["warmup_repetitions"]) < 0 or int(
        config["measured_repetitions"]
    ) < 3:
        raise ValueError("Priority-2B requires at least three measured repetitions")
    batches = [int(value) for value in config["quantization_batch_grid"]]
    if (
        not batches
        or batches != sorted(batches)
        or len(batches) != len(set(batches))
        or any(value <= 0 for value in batches)
    ):
        raise ValueError("quantization batch grid must be sorted unique positive integers")
    if float(config["maximum_median_update_ratio_to_eager"]) <= 0:
        raise ValueError("invalid eager timing gate")
    if float(config["maximum_median_peak_allocated_ratio_to_exact"]) <= 0:
        raise ValueError("invalid exact peak-memory gate")
    return config


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _statistics(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty measurements")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_standard_deviation": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
        "minimum": min(values),
        "maximum": max(values),
    }


def _candidate_label(batch_blocks: int) -> str:
    return f"streaming_quant_batch_{batch_blocks}"


def _worker_payload(config: dict, *, batch_blocks: int = 16) -> dict:
    payload = {key: value for key, value in config.items() if key in WORKER_FIELDS}
    payload["quantization_batch_blocks"] = int(batch_blocks)
    return payload


def _expected_worker_source() -> dict[str, str]:
    return {
        "runner": _sha256(ROOT / "tools/srq_fly_system_benchmark.py"),
        "optimized_learner": _sha256(
            ROOT / "methods/srq_fly_optimized/learner.py"
        ),
        "optimized_storage": _sha256(
            ROOT / "methods/srq_fly_optimized/storage.py"
        ),
    }


def _run_one(
    *, worker_config: Path, worker_method: str, label: str, repetition: int,
    output_dir: Path, device: str, profile_stages: bool,
) -> tuple[dict, torch.Tensor]:
    stem = f"rep_{repetition:02d}_{label}"
    result_path = output_dir / f"{stem}.json"
    probe_path = output_dir / f"{stem}.probe.pt"
    if result_path.is_file() and probe_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        profile_matches = (
            cached.get("profiled_task_stage_seconds") is not None
        ) == profile_stages
        if (
            cached.get("status") == "complete"
            and cached.get("method") == worker_method
            and cached.get("config_sha256") == _sha256(worker_config)
            and cached.get("source_identity") == _expected_worker_source()
            and profile_matches
        ):
            print(f"RESUME repetition={repetition} label={label}", flush=True)
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
        raise RuntimeError(
            f"Priority-2B worker failed: {label} repetition {repetition}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    probe = torch.load(probe_path, weights_only=True, map_location="cpu")
    print(
        f"DONE repetition={repetition} label={label} "
        f"update={result['total_update_seconds']:.4f}s "
        f"peak={result['peak_cuda_allocated_bytes']}",
        flush=True,
    )
    return result, probe


def _quantization_stage_increment(row: dict) -> int | None:
    tasks = row.get("profiled_task_stage_cuda_memory")
    if not tasks:
        return None
    increments = []
    for task in tasks:
        stage = task.get("factor_quantization")
        if stage and stage["peak_allocated_bytes"] is not None:
            increments.append(
                int(stage["peak_allocated_bytes"])
                - int(stage["before_allocated_bytes"])
            )
    return max(increments) if increments else None


def run(
    *, config_path: Path, output_dir: Path, device: str,
    require_clean_git: bool = False,
) -> dict:
    config_path = config_path.resolve()
    config = _read_config(config_path)
    if device != "cuda":
        raise ValueError("Priority-2B peak-memory selection requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if require_clean_git and dirty:
        raise RuntimeError(f"repository must be clean before Priority-2B:\n{dirty}")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "locked_candidate_configs"
    config_dir.mkdir(exist_ok=True)
    base_config = config_dir / "base_eager.json"
    base_config.write_text(
        json.dumps(_worker_payload(config), indent=2), encoding="utf-8"
    )
    candidate_configs = {}
    for batch in map(int, config["quantization_batch_grid"]):
        path = config_dir / f"streaming_batch_{batch}.json"
        path.write_text(
            json.dumps(_worker_payload(config, batch_blocks=batch), indent=2),
            encoding="utf-8",
        )
        candidate_configs[batch] = path

    labels = list(BASELINES) + [
        _candidate_label(batch) for batch in map(int, config["quantization_batch_grid"])
    ]
    total_repetitions = int(config["warmup_repetitions"]) + int(
        config["measured_repetitions"]
    )
    measured: dict[str, list[dict]] = {label: [] for label in labels}
    probes: dict[int, dict[str, torch.Tensor]] = {}
    for repetition in range(total_repetitions):
        warmup = repetition < int(config["warmup_repetitions"])
        shift = repetition % len(labels)
        ordered = labels[shift:] + labels[:shift]
        current_probes = {}
        for label in ordered:
            if label in BASELINES:
                worker_method = BASELINES[label]
                worker_config = base_config
            else:
                batch = int(label.rsplit("_", 1)[1])
                worker_method = "optimized_streaming_quant_blocked_qr_srq_int8"
                worker_config = candidate_configs[batch]
            result, probe = _run_one(
                worker_config=worker_config,
                worker_method=worker_method,
                label=label,
                repetition=repetition,
                output_dir=output_dir,
                device=device,
                profile_stages=(
                    not warmup
                    and repetition == int(config["warmup_repetitions"])
                    and label != "exact_fly"
                ),
            )
            current_probes[label] = probe
            if not warmup:
                measured[label].append(result)
        if not warmup:
            probes[repetition] = current_probes

    eager_state = int(measured["eager_quant"][0]["persistent_state_bytes"])
    tolerance = float(config["maximum_relative_logit_drift"])
    solver_tolerance = float(config["solver_tolerance"])
    summaries = {}
    eligible = []
    for label in labels:
        rows = measured[label]
        summary = {
            "label": label,
            "worker_method": rows[0]["method"],
            "update_seconds": _statistics(
                [float(row["total_update_seconds"]) for row in rows]
            ),
            "peak_allocated_bytes": _statistics(
                [float(row["peak_cuda_allocated_bytes"]) for row in rows]
            ),
            "peak_reserved_bytes": _statistics(
                [float(row["peak_cuda_reserved_bytes"]) for row in rows]
            ),
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
            "maximum_profiled_quantization_increment_bytes": next(
                (value for value in map(_quantization_stage_increment, rows)
                 if value is not None),
                None,
            ),
        }
        if label == "eager_quant" or label.startswith("streaming_quant_batch_"):
            streaming = label.startswith("streaming_quant_batch_")
            batch = (
                int(label.rsplit("_", 1)[1])
                if streaming
                else int(_worker_payload(config)["quantization_batch_blocks"])
            )
            time_ratios, peak_ratios, drifts = [], [], []
            for offset, repetition in enumerate(sorted(probes)):
                eager = measured["eager_quant"][offset]
                exact = measured["exact_fly"][offset]
                time_ratios.append(
                    float(rows[offset]["total_update_seconds"])
                    / float(eager["total_update_seconds"])
                )
                peak_ratios.append(
                    float(rows[offset]["peak_cuda_allocated_bytes"])
                    / float(exact["peak_cuda_allocated_bytes"])
                )
                if streaming:
                    reference = probes[repetition]["eager_quant"]
                    denominator = max(
                        float(torch.linalg.vector_norm(reference)), 1.0
                    )
                    drifts.append(
                        float(
                            torch.linalg.vector_norm(
                                probes[repetition][label] - reference
                            )
                        ) / denominator
                    )
                else:
                    drifts.append(0.0)
            gates = {
                "predictor_within_tolerance": max(drifts) <= tolerance,
                "persistent_state_unchanged": all(
                    int(row["persistent_state_bytes"]) == eager_state for row in rows
                ),
                "solver_stable": summary["maximum_solver_relative_residual"]
                <= solver_tolerance,
                "median_update_ratio_to_eager": statistics.median(time_ratios)
                <= float(config["maximum_median_update_ratio_to_eager"]),
                "median_peak_allocated_ratio_to_exact": statistics.median(peak_ratios)
                <= float(config["maximum_median_peak_allocated_ratio_to_exact"]),
            }
            summary.update(
                quantization_backend="streaming" if streaming else "eager",
                quantization_batch_blocks=batch,
                paired_update_ratio_to_eager=_statistics(time_ratios),
                paired_peak_allocated_ratio_to_exact=_statistics(peak_ratios),
                maximum_relative_logit_drift_from_eager=max(drifts),
                gates=gates,
            )
            if all(gates.values()):
                eligible.append(summary)
        summaries[label] = summary

    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                row["paired_peak_allocated_ratio_to_exact"]["median"],
                row["paired_update_ratio_to_eager"]["median"],
                0 if row["quantization_backend"] == "eager" else 1,
                row["quantization_batch_blocks"],
            ),
        )
    result = {
        "schema_version": 1,
        "study": config["study"],
        "status": "PASS_REVIEW_PRIORITY2B" if selected else "STOP_QUANTIZATION_GATE",
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
            "minimum median paired peak ratio among streaming encoders passing "
            "locked eager-timing, exact-peak, fidelity, state, and solver gates"
        ),
        "summaries": [summaries[label] for label in labels],
        "selected_candidate": None if selected is None else {
            "label": selected["label"],
            "quantization_backend": selected["quantization_backend"],
            "quantization_batch_blocks": selected["quantization_batch_blocks"],
            "paired_update_ratio_to_eager": selected["paired_update_ratio_to_eager"],
            "paired_peak_allocated_ratio_to_exact": selected[
                "paired_peak_allocated_ratio_to_exact"
            ],
            "maximum_relative_logit_drift_from_eager": selected[
                "maximum_relative_logit_drift_from_eager"
            ],
        },
    }
    output_path = output_dir / "priority2b_memory_results.json"
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
