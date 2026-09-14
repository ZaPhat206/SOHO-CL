"""M19: synthetic, test-free T4 panel-size benchmark for SRQ.

The benchmark uses the repository's real version-1 P2B factor codec and
blocked-QR update.  It times four synchronized stages independently:
dequantization, QR, requantization, and the two triangular solves.  Each
width/panel pair is an atomic resumable unit.  No dataset, feature cache,
labelled example, validation accuracy, or test split is opened.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable, TypeVar

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge.accounting import persistent_tensor_bytes  # noqa: E402
from methods.analytic_ridge.compressed_upper import (  # noqa: E402
    CompressedUpper,
    UpperBlock,
)
from methods.analytic_ridge.qr import blocked_qr_rank_update  # noqa: E402


STAGES = (
    "factor_dequantization",
    "blocked_qr",
    "factor_requantization",
    "triangular_solve",
)
STATUS_PASS = "PASS_M19_PANEL_SYSTEM_BENCHMARK_T4"
STATUS_WARNING = "COMPLETE_M19_PANEL_SYSTEM_BENCHMARK_WITH_WARNINGS"
T = TypeVar("T")


def _sha256(path: str | Path, *, source: bool = False) -> str:
    payload = Path(path).read_bytes()
    if source:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _lock_precision() -> dict:
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:  # pragma: no cover - only old PyTorch releases
        pass
    return {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": bool(
            getattr(getattr(torch.backends, "cuda", object()), "matmul", object()).allow_tf32
        ) if hasattr(getattr(torch.backends, "cuda", object()), "matmul") else None,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)
        if hasattr(torch.backends, "cudnn") else None,
    }


def _validate_config(config: dict) -> dict:
    required = {
        "schema_version", "study_id", "seed", "uses_test_set",
        "synthetic_only", "required_device_name_substring", "widths",
        "panel_sizes", "rows_per_update", "num_classes", "probe_rows",
        "warmup_repetitions", "measured_repetitions", "ridge_lambda",
        "synthetic_update_standard_deviation",
        "prior_factor_off_diagonal_scale", "storage", "update",
        "timed_stages", "selection", "gates",
    }
    if set(config) != required:
        raise ValueError(
            f"M19 config keys mismatch: missing={sorted(required-set(config))}, "
            f"unexpected={sorted(set(config)-required)}"
        )
    if config["schema_version"] != 1 or int(config["seed"]) != 2025:
        raise ValueError("M19 requires schema 1 and protocol seed 2025")
    if config["uses_test_set"] or not config["synthetic_only"]:
        raise ValueError("M19 must remain synthetic and test-free")
    widths = list(map(int, config["widths"]))
    panels = list(map(int, config["panel_sizes"]))
    if len(widths) != len(set(widths)) or len(panels) != len(set(panels)):
        raise ValueError("M19 widths and panels must be unique")
    if min(widths + panels) <= 0 or any(panel > max(widths) for panel in panels):
        raise ValueError("M19 widths/panels must be positive and compatible")
    counts = (
        int(config["rows_per_update"]), int(config["num_classes"]),
        int(config["probe_rows"]), int(config["measured_repetitions"]),
    )
    if min(counts) <= 0 or int(config["warmup_repetitions"]) < 0:
        raise ValueError("M19 counts and measured repetitions must be positive")
    if config["probe_rows"] > config["rows_per_update"]:
        raise ValueError("M19 probe cannot exceed the synthetic update")
    if float(config["ridge_lambda"]) <= 0:
        raise ValueError("M19 Ridge must be positive")
    if float(config["synthetic_update_standard_deviation"]) <= 0:
        raise ValueError("M19 synthetic update scale must be positive")
    if float(config["prior_factor_off_diagonal_scale"]) <= 0:
        raise ValueError("M19 prior-factor scale must be positive")
    storage = config["storage"]
    if set(storage) != {
        "mode", "block_size", "group_size", "quantization_batch_blocks"
    } or storage["mode"] != "int8" or min(
        int(storage["block_size"]), int(storage["group_size"]),
        int(storage["quantization_batch_blocks"]),
    ) <= 0:
        raise ValueError("M19 storage contract is invalid")
    update = config["update"]
    if set(update) != {
        "backend", "trailing_chunk_size", "preserve_update_rows"
    } or update["backend"] != "blocked_qr":
        raise ValueError("M19 must benchmark the blocked-QR backend")
    if update["trailing_chunk_size"] is not None and int(
        update["trailing_chunk_size"]
    ) <= 0:
        raise ValueError("M19 trailing chunk must be null or positive")
    if not bool(update["preserve_update_rows"]):
        raise ValueError("M19 locks the public non-consuming update contract")
    if tuple(config["timed_stages"]) != STAGES:
        raise ValueError("M19 stage list/order is locked")
    selection = config["selection"]
    if int(selection["reference_panel_size"]) not in panels:
        raise ValueError("M19 reference panel is absent")
    if selection["uses_accuracy"]:
        raise ValueError("M19 cannot select a panel from accuracy")
    if selection["rule"] != (
        "minimum_geometric_mean_total_stage_time_ratio_across_widths"
    ) or selection["tie_break"] != "smaller_panel_size":
        raise ValueError("M19 selection rule is not the preregistered rule")
    return config


def _read_config(path: str | Path) -> dict:
    return _validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_stage(
    name: str, operation: Callable[[], T], device: torch.device
) -> tuple[T, float, int | None]:
    if name not in STAGES:
        raise ValueError(f"unknown M19 stage {name}")
    _sync(device)
    baseline = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = int(torch.cuda.memory_allocated(device))
    started = time.perf_counter()
    result = operation()
    _sync(device)
    elapsed = time.perf_counter() - started
    peak_delta = None
    if device.type == "cuda" and baseline is not None:
        peak_delta = max(0, int(torch.cuda.max_memory_allocated(device)) - baseline)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError(f"invalid timing for {name}: {elapsed}")
    return result, elapsed, peak_delta


def _block_entries(dimension: int, block_size: int, row: int, column: int) -> int:
    rows = min(block_size, dimension - row * block_size)
    columns = min(block_size, dimension - column * block_size)
    return rows * columns if row != column else rows * (rows - 1) // 2


def _synthetic_prior_factor(config: dict, dimension: int, device: torch.device):
    """Create a stable, nonzero version-1 P2B state without a dense setup matrix."""
    storage = config["storage"]
    block_size = int(storage["block_size"])
    group_size = int(storage["group_size"])
    scale = float(config["prior_factor_off_diagonal_scale"])
    diagonal = torch.full(
        (dimension,), math.sqrt(float(config["ridge_lambda"])),
        dtype=torch.float32, device=device,
    )
    count = math.ceil(dimension / block_size)
    blocks = []
    block_index = 0
    for row in range(count):
        for column in range(row, count):
            entries = _block_entries(dimension, block_size, row, column)
            # A nonzero deterministic payload makes decode/re-encode exercise
            # the real INT8 path while diagonal dominance keeps the solve tame.
            quantized_value = block_index % 17 - 8
            values = torch.full(
                (entries,), quantized_value, dtype=torch.int8, device=device
            )
            scales = torch.full(
                (math.ceil(entries / group_size),), scale,
                dtype=torch.float32, device=device,
            )
            blocks.append(UpperBlock(row, column, values, scales))
            block_index += 1
    return CompressedUpper(
        dimension=dimension,
        block_size=block_size,
        group_size=group_size,
        mode="int8",
        diagonal=diagonal,
        blocks=blocks,
        validate_values=False,
    )


def _synthetic_inputs(config: dict, dimension: int, device: torch.device):
    seed = int(config["seed"]) + dimension * 13
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = int(config["rows_per_update"])
    classes = int(config["num_classes"])
    update = torch.randn(
        (rows, dimension), generator=generator, device=device, dtype=torch.float32
    ).mul_(float(config["synthetic_update_standard_deviation"]))
    labels = torch.arange(rows, device=device) % classes
    targets = torch.nn.functional.one_hot(labels, num_classes=classes).float()
    cross = update.T @ targets
    counts = targets.sum(0)
    probe = update[: int(config["probe_rows"])].clone()
    del labels, targets
    return update, cross, counts, probe


def _one_trial(*, config, prior, update_rows, cross, counts, probe, panel, device):
    timings: dict[str, float] = {}
    peaks: dict[str, int | None] = {}

    upper, timings[STAGES[0]], peaks[STAGES[0]] = _timed_stage(
        STAGES[0], lambda: prior.reconstruct_upper(dtype=torch.float32), device
    )
    upper, timings[STAGES[1]], peaks[STAGES[1]] = _timed_stage(
        STAGES[1],
        lambda: blocked_qr_rank_update(
            upper,
            update_rows,
            panel_size=int(panel),
            trailing_chunk_size=config["update"]["trailing_chunk_size"],
            preserve_update_rows=True,
        ),
        device,
    )

    def requantize():
        return CompressedUpper.from_upper_inplace_streaming(
            upper,
            block_size=int(config["storage"]["block_size"]),
            group_size=int(config["storage"]["group_size"]),
            mode="int8",
            maximum_batched_blocks=int(
                config["storage"]["quantization_batch_blocks"]
            ),
        )

    quantized, timings[STAGES[2]], peaks[STAGES[2]] = _timed_stage(
        STAGES[2], requantize, device
    )
    compressed, relative_factor_error = quantized

    def solve():
        intermediate = torch.linalg.solve_triangular(
            upper.T, cross, upper=False
        )
        return torch.linalg.solve_triangular(upper, intermediate, upper=True)

    weights, timings[STAGES[3]], peaks[STAGES[3]] = _timed_stage(
        STAGES[3], solve, device
    )
    residual_numerator = torch.linalg.vector_norm(
        upper.T @ (upper @ weights) - cross
    )
    residual_denominator = torch.clamp(torch.linalg.vector_norm(cross), min=1e-12)
    residual = float((residual_numerator / residual_denominator).item())
    logits = (probe @ weights).detach().cpu()
    predictions = logits.argmax(1)
    factor_bytes = persistent_tensor_bytes(compressed.persistent_tensors("factor"))
    total_state_bytes = factor_bytes + sum(
        tensor.numel() * tensor.element_size()
        for tensor in (cross, counts, weights)
    )
    record = {
        "stage_seconds": timings,
        "total_stage_seconds": sum(timings.values()),
        "stage_peak_incremental_allocated_bytes": peaks,
        "factor_persistent_tensor_bytes": factor_bytes,
        "analytic_persistent_tensor_bytes": total_state_bytes,
        "solver_relative_residual": residual,
        "relative_local_factor_error": float(relative_factor_error),
        "predictions": predictions.tolist(),
        "_probe_logits": logits,
    }
    del upper, compressed, weights, predictions, quantized
    return record


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(map(float, values))
    return {
        "median": float(statistics.median(ordered)),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _unit_path(output_dir: Path, width: int, panel: int) -> Path:
    return output_dir / "units" / f"width_{width}_panel_{panel}.json"


def run_unit(config: dict, *, width: int, panel: int, output_dir: Path, device):
    if width not in config["widths"] or panel not in config["panel_sizes"]:
        raise ValueError("M19 unit is outside the locked grid")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("M19 requested CUDA but CUDA is unavailable")
    precision = _lock_precision()
    torch.manual_seed(int(config["seed"]) + width)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]) + width)
    prior = _synthetic_prior_factor(config, width, device)
    update_rows, cross, counts, probe = _synthetic_inputs(config, width, device)

    warmups = int(config["warmup_repetitions"])
    measured = int(config["measured_repetitions"])
    for warmup in range(warmups):
        print(
            f"M19 WARMUP width={width} panel={panel} {warmup+1}/{warmups}",
            flush=True,
        )
        _one_trial(
            config=config, prior=prior, update_rows=update_rows, cross=cross,
            counts=counts, probe=probe, panel=panel, device=device,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    records = []
    repetition_logits = []
    for repetition in range(measured):
        record = _one_trial(
            config=config, prior=prior, update_rows=update_rows, cross=cross,
            counts=counts, probe=probe, panel=panel, device=device,
        )
        repetition_logits.append(record.pop("_probe_logits"))
        records.append(record)
        print(
            f"M19 width={width} panel={panel} repeat={repetition+1}/{measured} "
            + " ".join(f"{stage}={record['stage_seconds'][stage]:.4f}s" for stage in STAGES),
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    medians = {
        stage: _distribution([row["stage_seconds"][stage] for row in records])
        for stage in STAGES
    }
    totals = _distribution([row["total_stage_seconds"] for row in records])
    predictions_repeat_identical = all(
        row["predictions"] == records[0]["predictions"] for row in records
    )
    reference_repeat_logits = repetition_logits[0]
    repeat_logit_drifts = [
        float(torch.linalg.vector_norm(current - reference_repeat_logits).item())
        / max(float(torch.linalg.vector_norm(reference_repeat_logits).item()), 1e-12)
        for current in repetition_logits
    ]
    factor_bytes = {row["factor_persistent_tensor_bytes"] for row in records}
    total_bytes = {row["analytic_persistent_tensor_bytes"] for row in records}
    result = {
        "schema_version": 1,
        "status": "complete",
        "study_id": config["study_id"],
        "uses_test_set": False,
        "synthetic_only": True,
        "width": width,
        "panel_size": panel,
        "rows_per_update": int(config["rows_per_update"]),
        "warmup_repetitions": warmups,
        "measured_repetitions": measured,
        "stage_seconds": medians,
        "total_stage_seconds": totals,
        "raw_repetitions": records,
        "factor_persistent_tensor_bytes": records[-1]["factor_persistent_tensor_bytes"],
        "analytic_persistent_tensor_bytes": records[-1]["analytic_persistent_tensor_bytes"],
        "state_bytes_identical_across_repetitions": len(factor_bytes) == len(total_bytes) == 1,
        "predictions_identical_across_repetitions": predictions_repeat_identical,
        "maximum_relative_logit_drift_across_repetitions": max(repeat_logit_drifts),
        "predictions": records[-1]["predictions"],
        "probe_logits": repetition_logits[-1].tolist(),
        "maximum_solver_relative_residual": max(
            row["solver_relative_residual"] for row in records
        ),
        "precision_lock": precision,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda" else platform.processor(),
            "total_device_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda" else None,
        },
    }
    path = _unit_path(output_dir, width, panel)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return result


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return float("inf")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize(config: dict, units: list[dict], *, config_path: Path) -> dict:
    expected = {(width, panel) for width in config["widths"] for panel in config["panel_sizes"]}
    observed = {(row["width"], row["panel_size"]) for row in units if row.get("status") == "complete"}
    complete = observed == expected and len(units) == len(expected)
    by_key = {(row["width"], row["panel_size"]): row for row in units}
    reference_panel = int(config["selection"]["reference_panel_size"])
    comparisons = {}
    all_prediction_equal = True
    maximum_relative_logit_drift = 0.0
    for width in config["widths"]:
        if (width, reference_panel) not in by_key:
            continue
        reference = by_key[(width, reference_panel)]["predictions"]
        reference_logits = torch.tensor(
            by_key[(width, reference_panel)]["probe_logits"], dtype=torch.float32
        )
        for panel in config["panel_sizes"]:
            if (width, panel) not in by_key:
                continue
            current = by_key[(width, panel)]["predictions"]
            current_logits = torch.tensor(
                by_key[(width, panel)]["probe_logits"], dtype=torch.float32
            )
            agreement = sum(a == b for a, b in zip(current, reference)) / len(reference)
            relative_logit_drift = float(
                torch.linalg.vector_norm(current_logits - reference_logits).item()
            ) / max(float(torch.linalg.vector_norm(reference_logits).item()), 1e-12)
            comparisons[f"{width}:{panel}"] = {
                "prediction_agreement_with_reference_panel": agreement,
                "prediction_changes": sum(a != b for a, b in zip(current, reference)),
                "relative_logit_drift_with_reference_panel": relative_logit_drift,
            }
            all_prediction_equal &= agreement == 1.0
            maximum_relative_logit_drift = max(
                maximum_relative_logit_drift, relative_logit_drift
            )

    state_values = {row["analytic_persistent_tensor_bytes"] for row in units}
    state_by_width = {
        str(width): {
            row["analytic_persistent_tensor_bytes"]
            for row in units if row["width"] == width
        }
        for width in config["widths"]
    }
    state_invariant = all(len(values) == 1 for values in state_by_width.values())
    residual_limit = float(config["gates"]["maximum_solver_relative_residual"])
    residual_ok = all(row["maximum_solver_relative_residual"] <= residual_limit for row in units)
    finite_ok = all(
        math.isfinite(row["stage_seconds"][stage]["median"])
        and row["stage_seconds"][stage]["median"] > 0
        for row in units for stage in STAGES
    )
    repeat_ok = all(
        row["predictions_identical_across_repetitions"]
        and row["state_bytes_identical_across_repetitions"]
        and row["maximum_relative_logit_drift_across_repetitions"] == 0.0
        for row in units
    )
    logit_drift_ok = maximum_relative_logit_drift <= float(
        config["gates"]["maximum_relative_logit_drift_from_reference_panel"]
    )
    required_device = str(config["required_device_name_substring"] or "")
    device_ok = all(
        required_device.lower() in row["environment"]["device_name"].lower()
        for row in units
    ) if required_device else True

    ratios_by_panel = {}
    for panel in config["panel_sizes"]:
        ratios = []
        for width in config["widths"]:
            candidate = by_key.get((width, panel))
            reference = by_key.get((width, reference_panel))
            if candidate is None or reference is None:
                continue
            ratios.append(
                candidate["total_stage_seconds"]["median"]
                / reference["total_stage_seconds"]["median"]
            )
        ratios_by_panel[str(panel)] = _geometric_mean(ratios)
    selected_panel = None
    finite_candidates = [
        (ratio, int(panel)) for panel, ratio in ratios_by_panel.items()
        if math.isfinite(ratio)
    ]
    if finite_candidates:
        selected_panel = min(finite_candidates, key=lambda item: (item[0], item[1]))[1]

    stage_totals = {
        stage: sum(row["stage_seconds"][stage]["median"] for row in units)
        for stage in STAGES
    }
    all_stage_time = sum(stage_totals.values())
    shares = {
        stage: value / all_stage_time if all_stage_time else float("nan")
        for stage, value in stage_totals.items()
    }
    bottleneck = max(shares, key=shares.get) if shares else None
    gates = {
        "all_width_panel_units_complete": complete,
        "required_t4_device": device_ok,
        "all_stage_timings_finite": finite_ok,
        "solver_residual_within_tolerance": residual_ok,
        "state_bytes_identical_across_panels_at_each_width": state_invariant,
        "repetitions_reproduce_state_and_predictions": repeat_ok,
        "predictions_identical_across_panels": all_prediction_equal,
        "relative_logit_drift_within_tolerance": logit_drift_ok,
    }
    result = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": STATUS_PASS if all(gates.values()) else STATUS_WARNING,
        "uses_test_set": False,
        "synthetic_only": True,
        "config_sha256": _sha256(config_path, source=True),
        "runner_sha256": _sha256(Path(__file__), source=True),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "grid": {
            "widths": config["widths"],
            "panel_sizes": config["panel_sizes"],
            "rows_per_update": config["rows_per_update"],
            "unit_count": len(units),
        },
        "selection": {
            "rule": config["selection"]["rule"],
            "reference_panel_size": reference_panel,
            "geometric_mean_total_time_ratio_by_panel": ratios_by_panel,
            "selected_panel_size": selected_panel,
        },
        "bottleneck": {
            "stage": bottleneck,
            "aggregate_median_seconds": stage_totals,
            "aggregate_time_fraction": shares,
        },
        "cross_panel_prediction_checks": comparisons,
        "maximum_relative_logit_drift_across_panels": maximum_relative_logit_drift,
        "analytic_state_bytes_observed_across_all_widths": sorted(state_values),
        "gates": gates,
        "units": units,
    }
    return result


def run(config_path: Path, output_dir: Path, *, device: str, max_new_units: int | None):
    config = _read_config(config_path)
    unit_dir = output_dir / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        (width, panel)
        for width in config["widths"]
        for panel in config["panel_sizes"]
        if not _unit_path(output_dir, width, panel).is_file()
    ]
    if max_new_units is not None:
        pending = pending[:max_new_units]
    for width, panel in pending:
        print(f"M19 START UNIT width={width} panel={panel}", flush=True)
        run_unit(
            config, width=width, panel=panel, output_dir=output_dir,
            device=device,
        )
        print(f"M19 CHECKPOINT width={width} panel={panel}: COMPLETE", flush=True)

    units = []
    for width in config["widths"]:
        for panel in config["panel_sizes"]:
            path = _unit_path(output_dir, width, panel)
            if path.is_file():
                units.append(json.loads(path.read_text(encoding="utf-8")))
    progress = {
        "study_id": config["study_id"],
        "completed_units": len(units),
        "total_units": len(config["widths"]) * len(config["panel_sizes"]),
    }
    (output_dir / "m19_progress.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
    )
    if progress["completed_units"] == progress["total_units"]:
        result = summarize(config, units, config_path=config_path)
        (output_dir / "m19_results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print("M19 STATUS:", result["status"], flush=True)
        print("M19 SELECTED PANEL:", result["selection"]["selected_panel_size"], flush=True)
        print("M19 BOTTLENECK:", result["bottleneck"]["stage"], flush=True)
        print("M19 GATES:", json.dumps(result["gates"], indent=2), flush=True)
    else:
        print(
            f"M19 PROGRESS: {progress['completed_units']}/{progress['total_units']} units",
            flush=True,
        )
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-units", type=int)
    args = parser.parse_args()
    if args.max_new_units is not None and args.max_new_units <= 0:
        raise ValueError("--max-new-units must be positive")
    run(
        args.config.resolve(), args.output_dir.resolve(), device=args.device,
        max_new_units=args.max_new_units,
    )


if __name__ == "__main__":
    main()
