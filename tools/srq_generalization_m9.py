"""Repeated train-only whole-process systems audit for Exact FLY and SRQ.

Four paired repetitions alternate method order. Every method executes in a
fresh process from frozen-backbone loading through feature extraction,
analytic updates, a fixed probe, and temporary checkpoint serialization. The
parent attributes NVML memory to the worker PID. No test dataset is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import srq_fly_priority5_memory as priority5


METHODS = priority5.METHODS
TOP_KEYS = {
    "schema_version", "study_id", "source_priority5_config",
    "source_priority5_config_sha256", "repetitions", "method_orders",
    "checkpoint_measurement", "aggregation", "gates",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_config(path: Path) -> tuple[dict, Path, dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config["schema_version"] != 1:
        raise ValueError("M9 config keys/schema mismatch")
    repetitions = int(config["repetitions"])
    if repetitions < int(config["gates"]["minimum_repetitions"]):
        raise ValueError("M9 requires at least the locked minimum repetitions")
    orders = config["method_orders"]
    if len(orders) != repetitions or any(
        sorted(order) != sorted(METHODS) or len(order) != len(METHODS)
        for order in orders
    ):
        raise ValueError("each M9 repetition must contain both methods once")
    first_counts = {method: sum(order[0] == method for order in orders) for method in METHODS}
    if config["gates"]["require_balanced_first_method"] and len(set(first_counts.values())) != 1:
        raise ValueError("M9 method-first schedule must be balanced")
    if set(config["gates"]) != {
        "minimum_repetitions", "require_every_priority5_gate",
        "require_checkpoint_bytes", "require_balanced_first_method",
    }:
        raise ValueError("M9 gate keys mismatch")
    source_path = (ROOT / config["source_priority5_config"]).resolve()
    if source_path != (
        ROOT / "configs/srq_fly_priority5_cifar100_whole_process_memory.json"
    ).resolve():
        raise ValueError("M9 source Priority-5 config path is not locked")
    if _sha256(source_path) != config["source_priority5_config_sha256"]:
        raise ValueError("M9 source Priority-5 config identity mismatch")
    source = priority5._read_config(source_path)
    return config, source_path, source


def _source_identity() -> dict[str, object]:
    return {
        "m9_runner": _sha256(ROOT / "tools/srq_generalization_m9.py"),
        "priority5_runner": _sha256(ROOT / "tools/srq_fly_priority5_memory.py"),
        "priority5_dependencies": priority5._source_identity(),
    }


def _serialize_checkpoint(torch, learner, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(learner.state_dict(), path)
    size = int(path.stat().st_size)
    path.unlink()
    if size <= 0:
        raise RuntimeError("serialized checkpoint is empty")
    return size


def run_worker(args) -> dict:
    import torch

    from models.backbone import load_model
    from tools import srq_fly_system_benchmark as system_benchmark
    from utils.train_utils import random_initialization

    m9_path = Path(args.config).resolve()
    m9, source_path, config = _read_config(m9_path)
    if args.method not in METHODS or args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M9 worker requires a locked method and CUDA")
    checkpoint = Path(args.backbone_checkpoint).resolve()
    if checkpoint.stat().st_size != int(config["checkpoint_size"]):
        raise RuntimeError("checkpoint size mismatch")
    if _sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    output = Path(args.output).resolve()
    marker = Path(args.stage_marker).resolve()
    scratch = Path(args.scratch_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    random_initialization(int(config["seed"]))
    device = torch.device("cuda")
    records: dict[str, dict] = {}
    priority5._set_stage(marker, "dependency_import")

    with priority5._torch_stage(torch, device, marker, "backbone_load", records):
        loader = priority5._build_train_loader(
            config, Path(args.root), scratch / "dataset_view"
        )
        backbone = load_model(
            config["model_name"], checkpoint_path=str(checkpoint),
            expected_checkpoint_size=int(config["checkpoint_size"]),
            expected_checkpoint_sha256=config["checkpoint_sha256"],
        ).eval().to(device)

    with priority5._torch_stage(torch, device, marker, "feature_extraction", records):
        train_features, train_labels = priority5._extract_train_features_to_cpu(
            torch, backbone, loader, device, int(config["feature_dim"])
        )
    del backbone, loader
    torch.cuda.empty_cache()
    if tuple(train_features.shape) != (config["train_samples"], config["feature_dim"]):
        raise RuntimeError("training feature shape mismatch")
    if not bool(torch.isfinite(train_features).all()):
        raise RuntimeError("non-finite training features")
    class_order, parts = priority5._task_indices(torch, train_labels, config)

    worker_config = priority5._worker_config(config)
    method_name = (
        "exact_fly_dense" if args.method == "exact_fly_10000"
        else "optimized_streaming_quant_blocked_qr_srq_int8"
    )
    with priority5._torch_stage(torch, device, marker, "analytic_update", records):
        learner = system_benchmark._learner(method_name, worker_config, device)
        projection_sha = priority5._projection_sha256(
            torch, learner.flyhash.projection_matrix
        )
        task_seconds = []
        for task, indices in enumerate(parts):
            started = time.perf_counter()
            values = train_features[indices].to(device)
            codes = learner.flyhash(
                values, float(config["representation"]["coding_level"]),
                absolute_wta=False,
            ).to(torch.float32)
            del values
            task_labels = train_labels[indices]
            if args.method == "srq_fly_p2b_10000":
                learner.update_codes_consuming(codes, task_labels)
            else:
                learner.update_codes(codes, task_labels)
            del codes
            torch.cuda.synchronize(device)
            task_seconds.append(time.perf_counter() - started)
            print(
                f"TASK rep={args.repetition} method={args.method} "
                f"{task + 1}/{len(parts)} seconds={task_seconds[-1]:.3f}",
                flush=True,
            )

    with priority5._torch_stage(torch, device, marker, "final_probe", records):
        probe_rows = min(512, len(train_features))
        probe_codes = learner.flyhash(
            train_features[:probe_rows].to(device),
            float(config["representation"]["coding_level"]),
            absolute_wta=False,
        ).to(torch.float32)
        logits = learner.predict_logits_from_codes(probe_codes)
        predictions = torch.tensor(
            [learner.class_ids[index] for index in logits.argmax(1).cpu().tolist()]
        )
        probe_logits = logits.detach().cpu()
        del probe_codes, logits

    with priority5._torch_stage(
        torch, device, marker, "checkpoint_serialization", records
    ):
        checkpoint_bytes = _serialize_checkpoint(
            torch, learner, scratch / "checkpoint.pt"
        )

    result = {
        "schema_version": 1,
        "study_id": m9["study_id"],
        "repetition": int(args.repetition),
        "method": args.method,
        "status": "complete",
        "uses_test_set": False,
        "test_features_materialized": False,
        "train_samples": len(train_features),
        "class_order": class_order,
        "projection_sha256": projection_sha,
        "persistent_state_bytes": int(learner.persistent_state_bytes()),
        "serialized_checkpoint_bytes": checkpoint_bytes,
        "solver_relative_residual": float(
            learner.diagnostics["solver_relative_residual"]
        ),
        "task_update_seconds": task_seconds,
        "torch_cuda_stages": records,
        "whole_process_torch_peak_allocated_bytes": max(
            row["peak_allocated_bytes"] for row in records.values()
        ),
        "whole_process_torch_peak_reserved_bytes": max(
            row["peak_reserved_bytes"] for row in records.values()
        ),
        "probe_predictions": predictions.tolist(),
        "probe_logits": probe_logits.tolist(),
        "m9_config_sha256": _sha256(m9_path),
        "source_priority5_config_sha256": _sha256(source_path),
        "source_identity": _source_identity(),
    }
    _atomic_json(output, result)
    priority5._set_stage(marker, "complete")
    return result


def _sample_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _method_metrics(result: dict, monitor: dict) -> dict[str, float]:
    stages = result["torch_cuda_stages"]
    analytic_nvml = monitor["stage_peaks"]["analytic_update"]["process_bytes"]
    return {
        "persistent_state_bytes": float(result["persistent_state_bytes"]),
        "serialized_checkpoint_bytes": float(result["serialized_checkpoint_bytes"]),
        "torch_analytic_peak_allocated_bytes": float(
            stages["analytic_update"]["peak_allocated_bytes"]
        ),
        "torch_analytic_peak_reserved_bytes": float(
            stages["analytic_update"]["peak_reserved_bytes"]
        ),
        "nvml_analytic_peak_process_bytes": float(analytic_nvml),
        "nvml_whole_process_peak_process_bytes": float(
            monitor["peak_worker_process_bytes"]
        ),
        "backbone_load_seconds": float(stages["backbone_load"]["seconds"]),
        "feature_extraction_seconds": float(
            stages["feature_extraction"]["seconds"]
        ),
        "analytic_stage_seconds": float(stages["analytic_update"]["seconds"]),
        "final_probe_seconds": float(stages["final_probe"]["seconds"]),
        "checkpoint_serialization_seconds": float(
            stages["checkpoint_serialization"]["seconds"]
        ),
        "total_measured_stage_seconds": float(
            sum(row["seconds"] for row in stages.values())
        ),
    }


def _write_repetition_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, summary: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "metric", "mean", "sample_std", "minimum", "maximum"],
        )
        writer.writeheader()
        for method in METHODS:
            for metric, values in summary[method].items():
                writer.writerow({"method": method, "metric": metric, **values})


def _svg_bars(path: Path, title: str, panels: list[tuple[str, dict]], unit: str) -> None:
    width, height = 280 * len(panels), 350
    colors = {"exact_fly_10000": "#4C78A8", "srq_fly_p2b_10000": "#F58518"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="16">{title}</text>',
    ]
    for panel_index, (label, values) in enumerate(panels):
        left = panel_index * 280 + 35
        baseline, chart_height = 285, 210
        maximum = max(
            values[method]["mean"] + values[method]["sample_std"] for method in METHODS
        ) or 1.0
        parts.append(
            f'<text x="{left + 105}" y="52" text-anchor="middle" font-size="12">{label}</text>'
        )
        parts.append(
            f'<line x1="{left}" y1="{baseline}" x2="{left + 210}" y2="{baseline}" stroke="black"/>'
        )
        for index, method in enumerate(METHODS):
            mean = values[method]["mean"]
            std = values[method]["sample_std"]
            bar_height = chart_height * mean / maximum
            x = left + 35 + index * 95
            y = baseline - bar_height
            error = chart_height * std / maximum
            parts.extend([
                f'<rect x="{x}" y="{y}" width="55" height="{bar_height}" fill="{colors[method]}"/>',
                f'<line x1="{x + 27.5}" y1="{y - error}" x2="{x + 27.5}" y2="{y + error}" stroke="black"/>',
                f'<line x1="{x + 20}" y1="{y - error}" x2="{x + 35}" y2="{y - error}" stroke="black"/>',
                f'<text x="{x + 27.5}" y="{y - error - 5}" text-anchor="middle" font-size="10">{mean:.1f}</text>',
                f'<text x="{x + 27.5}" y="{baseline + 17}" text-anchor="middle" font-size="10">{"Exact" if index == 0 else "SRQ"}</text>',
            ])
        parts.append(
            f'<text x="{left + 105}" y="325" text-anchor="middle" font-size="10">mean +/- sample SD ({unit}), n=4</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_figures(output_dir: Path, aggregate: dict) -> None:
    memory_metrics = (
        ("Persistent state", "persistent_state_bytes"),
        ("Serialized checkpoint", "serialized_checkpoint_bytes"),
        ("PyTorch analytic peak", "torch_analytic_peak_allocated_bytes"),
        ("NVML process peak", "nvml_whole_process_peak_process_bytes"),
    )
    memory_panels = []
    for label, metric in memory_metrics:
        memory_panels.append((label, {
            method: {
                name: value / 2**20
                for name, value in aggregate[method][metric].items()
            }
            for method in METHODS
        }))
    _svg_bars(
        output_dir / "m9_memory_summary.svg",
        "Repeated whole-process memory audit",
        memory_panels,
        "MiB",
    )
    time_metrics = (
        ("Backbone load", "backbone_load_seconds"),
        ("Feature extraction", "feature_extraction_seconds"),
        ("Analytic update", "analytic_stage_seconds"),
        ("Checkpoint save", "checkpoint_serialization_seconds"),
        ("Total stages", "total_measured_stage_seconds"),
    )
    time_panels = [
        (label, {method: aggregate[method][metric] for method in METHODS})
        for label, metric in time_metrics
    ]
    _svg_bars(
        output_dir / "m9_stage_time_summary.svg",
        "Repeated stage-time audit",
        time_panels,
        "seconds",
    )


def _ratio(exact: dict, srq: dict, metric: str) -> float:
    denominator = exact[metric]
    return srq[metric] / denominator if denominator else float("inf")


def run_driver(args) -> dict:
    config_path = Path(args.config).resolve()
    m9, source_path, source = _read_config(config_path)
    if args.device != "cuda":
        raise ValueError("M9 requires CUDA")
    if args.require_clean_git:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
        if dirty:
            raise RuntimeError(f"repository must be clean before M9:\n{dirty}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    all_monitors: list[dict] = []
    repetitions: list[dict] = []
    sampler = priority5._NVMLSampler(source["nvml"]["device_index"])
    try:
        for repetition, order in enumerate(m9["method_orders"], start=1):
            pair_results, pair_monitors = [], []
            for position, method in enumerate(order, start=1):
                rep_dir = output_dir / f"rep_{repetition:02d}"
                result_path = rep_dir / f"{method}.json"
                marker = rep_dir / f"{method}.stage.json"
                worker_scratch = (
                    Path(args.scratch_dir).resolve()
                    / f"rep_{repetition:02d}" / method
                )
                command = [
                    sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                    "--config", str(config_path), "--repetition", str(repetition),
                    "--method", method, "--root", args.root,
                    "--backbone-checkpoint", args.backbone_checkpoint,
                    "--output", str(result_path), "--stage-marker", str(marker),
                    "--scratch-dir", str(worker_scratch), "--device", args.device,
                ]
                print(
                    f"START repetition={repetition}/{m9['repetitions']} "
                    f"position={position}/2 method={method}", flush=True,
                )
                process = subprocess.Popen(command, cwd=ROOT)
                monitor = priority5._monitor_worker(
                    process, marker, sampler,
                    float(source["nvml"]["poll_interval_seconds"]),
                )
                return_code = process.wait()
                if return_code != 0 or not result_path.is_file():
                    raise RuntimeError(
                        f"M9 worker failed: repetition={repetition}, method={method}"
                    )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                monitor.update(
                    repetition=repetition,
                    order_position=position,
                    method=method,
                    nvml_device_name=sampler.device_name,
                )
                pair_results.append(result)
                pair_monitors.append(monitor)
                all_results.append(result)
                all_monitors.append(monitor)
                print(
                    f"DONE repetition={repetition} method={method} "
                    f"process_peak={monitor['peak_worker_process_bytes']} "
                    f"checkpoint={result['serialized_checkpoint_bytes']}",
                    flush=True,
                )
            pair_summary = priority5._summarize(source, pair_results, pair_monitors)
            repetitions.append({
                "repetition": repetition,
                "method_order": list(order),
                "priority5_status": pair_summary["status"],
                "priority5_gates": pair_summary["gates"],
                "priority5_comparisons": pair_summary["comparisons"],
            })
    finally:
        sampler.close()

    rows = []
    metrics_by_repetition: dict[int, dict[str, dict[str, float]]] = {}
    for result in all_results:
        repetition = int(result["repetition"])
        monitor = next(
            row for row in all_monitors
            if row["repetition"] == repetition and row["method"] == result["method"]
        )
        metrics = _method_metrics(result, monitor)
        metrics_by_repetition.setdefault(repetition, {})[result["method"]] = metrics
        rows.append({
            "repetition": repetition,
            "method": result["method"],
            "order_position": monitor["order_position"],
            **metrics,
            "solver_relative_residual": result["solver_relative_residual"],
        })

    aggregate = {
        method: {
            metric: _sample_stats([
                metrics_by_repetition[index][method][metric]
                for index in sorted(metrics_by_repetition)
            ])
            for metric in next(iter(metrics_by_repetition.values()))[method]
        }
        for method in METHODS
    }
    ratio_metrics = (
        "persistent_state_bytes", "serialized_checkpoint_bytes",
        "torch_analytic_peak_allocated_bytes",
        "torch_analytic_peak_reserved_bytes",
        "nvml_analytic_peak_process_bytes",
        "nvml_whole_process_peak_process_bytes",
        "analytic_stage_seconds", "total_measured_stage_seconds",
    )
    paired_ratios = {
        metric: _sample_stats([
            _ratio(
                metrics_by_repetition[index]["exact_fly_10000"],
                metrics_by_repetition[index]["srq_fly_p2b_10000"],
                metric,
            )
            for index in sorted(metrics_by_repetition)
        ])
        for metric in ratio_metrics
    }
    source_identities = [result["source_identity"] for result in all_results]
    checks = {
        "minimum_repetitions": len(repetitions) >= int(
            m9["gates"]["minimum_repetitions"]
        ),
        "all_workers_complete": len(all_results) == 2 * int(m9["repetitions"])
        and all(result["status"] == "complete" for result in all_results),
        "all_priority5_gates": all(
            row["priority5_status"] == "PASS_PRIORITY5_MEMORY"
            and all(row["priority5_gates"].values()) for row in repetitions
        ),
        "train_only_boundary": all(
            not result["uses_test_set"] and not result["test_features_materialized"]
            for result in all_results
        ),
        "checkpoint_bytes_recorded": all(
            result["serialized_checkpoint_bytes"] > 0 for result in all_results
        ),
        "source_identity_stable": all(
            identity == source_identities[0] for identity in source_identities
        ),
        "persistent_state_deterministic": all(
            len({
                result["persistent_state_bytes"] for result in all_results
                if result["method"] == method
            }) == 1 for method in METHODS
        ),
        "balanced_locked_order": [row["method_order"] for row in repetitions]
        == m9["method_orders"],
        "nvml_worker_observed": all(
            monitor["worker_sample_count"] >= source["nvml"]["minimum_worker_samples"]
            and monitor["peak_worker_process_bytes"] > 0
            for monitor in all_monitors
        ),
    }
    status = (
        "PASS_M9_REPEATED_SYSTEMS_TRAIN_ONLY"
        if all(checks.values()) else "FAIL_M9_REPEATED_SYSTEMS_TRAIN_ONLY"
    )
    payload = {
        "schema_version": 1,
        "study_id": m9["study_id"],
        "status": status,
        "uses_test_set": False,
        "repetitions": repetitions,
        "methods": all_results,
        "nvml": all_monitors,
        "aggregate": aggregate,
        "paired_srq_over_exact_ratios": paired_ratios,
        "gates": checks,
        "config_sha256": _sha256(config_path),
        "source_priority5_config_sha256": _sha256(source_path),
        "source_identity": _source_identity(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "measurement_note": (
            "Process-attributed NVML is the primary whole-process metric. "
            "PyTorch allocated/reserved metrics cover only its allocator. "
            "Checkpoint bytes are temporary torch.save file sizes. Performance "
            "ratios are reported with variability and are not universal bounds."
        ),
    }
    _write_repetition_csv(output_dir / "m9_repetitions.csv", rows)
    _write_summary_csv(output_dir / "m9_summary.csv", aggregate)
    _write_figures(output_dir, aggregate)
    _atomic_json(output_dir / "m9_results.json", payload)
    print(json.dumps({
        "status": status,
        "paired_srq_over_exact_ratios": paired_ratios,
        "gates": checks,
    }, indent=2))
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", required=True)
    worker.add_argument("--repetition", type=int, required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--root", required=True)
    worker.add_argument("--backbone-checkpoint", required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument("--stage-marker", required=True)
    worker.add_argument("--scratch-dir", required=True)
    worker.add_argument("--device", default="cuda")
    driver = subparsers.add_parser("run")
    driver.add_argument("--config", required=True)
    driver.add_argument("--root", required=True)
    driver.add_argument("--backbone-checkpoint", required=True)
    driver.add_argument("--output-dir", required=True)
    driver.add_argument("--scratch-dir", required=True)
    driver.add_argument("--device", default="cuda")
    driver.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "worker":
        run_worker(args)
    else:
        result = run_driver(args)
        if result["status"] != "PASS_M9_REPEATED_SYSTEMS_TRAIN_ONLY":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
