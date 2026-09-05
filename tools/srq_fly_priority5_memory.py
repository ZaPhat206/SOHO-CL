"""Whole-process train-only CUDA/NVML memory audit for Exact FLY and SRQ-FLY.

Each method runs in a fresh process and repeats the complete deployed path:
load the frozen ViT, extract CIFAR-100 training features, release the backbone,
then execute ten analytic updates.  The parent samples NVML while the worker
also records PyTorch allocator peaks by stage.  No test dataset is opened and
no per-sample cache is written to the evidence bundle.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ("exact_fly_10000", "srq_fly_p2b_10000")
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_size", "checkpoint_sha256", "feature_dim", "seed",
    "num_classes", "num_tasks", "train_samples", "batch_size",
    "num_workers", "representation", "ridge_lambda", "storage",
    "p2b_backend", "methods", "nvml", "gates",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config["schema_version"] != 1:
        raise ValueError("Priority-5 config keys/schema mismatch")
    if config["dataset"] != "CIFAR-100" or config["methods"] != list(METHODS):
        raise ValueError("Priority 5 is locked to CIFAR-100 and paired methods")
    if config["seed"] != 2025 or config["feature_dim"] != 768:
        raise ValueError("locked seed/feature dimension mismatch")
    if config["num_classes"] % config["num_tasks"]:
        raise ValueError("classes must divide evenly into tasks")
    if min(config[key] for key in (
        "checkpoint_size", "num_classes", "num_tasks", "train_samples",
        "batch_size", "num_workers", "ridge_lambda",
    )) <= 0:
        raise ValueError("invalid positive scalar")
    representation = config["representation"]
    if set(representation) != {"expand_dim", "synaptic_degree", "coding_level"}:
        raise ValueError("representation keys mismatch")
    if representation["expand_dim"] != 10000 or not 0 < representation["coding_level"] <= 1:
        raise ValueError("invalid locked representation")
    if not 0 < representation["synaptic_degree"] <= config["feature_dim"]:
        raise ValueError("invalid synaptic degree")
    if set(config["storage"]) != {"block_size", "group_size"} or min(config["storage"].values()) <= 0:
        raise ValueError("storage keys/values mismatch")
    expected_backend = {
        "storage_mode": "int8", "update_backend": "blocked_qr",
        "update_panel_size": 128, "first_update_backend": "gram_cholesky",
        "quantization_backend": "streaming", "quantization_batch_blocks": 64,
    }
    if config["p2b_backend"] != expected_backend:
        raise ValueError("Priority 5 must measure the locked P2B backend")
    if set(config["nvml"]) != {
        "device_index", "poll_interval_seconds", "minimum_worker_samples"
    }:
        raise ValueError("NVML keys mismatch")
    if not 0 < config["nvml"]["poll_interval_seconds"] <= 0.25:
        raise ValueError("NVML polling interval must be in (0, 0.25]")
    if config["nvml"]["minimum_worker_samples"] <= 0:
        raise ValueError("invalid NVML sample gate")
    if set(config["gates"]) != {
        "minimum_prediction_agreement", "maximum_solver_relative_residual",
        "maximum_srq_state_fraction_of_exact",
        "maximum_srq_analytic_torch_peak_ratio",
        "maximum_srq_analytic_nvml_peak_ratio",
    } or any(float(value) <= 0 for value in config["gates"].values()):
        raise ValueError("gate keys/values mismatch")
    return config


def _source_identity() -> dict[str, str]:
    paths = {
        "runner": ROOT / "tools/srq_fly_priority5_memory.py",
        "model_loader": ROOT / "models/backbone.py",
        "flyhash": ROOT / "models/flyhash.py",
        "data_utils": ROOT / "utils/data_utils.py",
        "train_utils": ROOT / "utils/train_utils.py",
        "optimized_learner": ROOT / "methods/srq_fly_optimized/learner.py",
        "optimized_storage": ROOT / "methods/srq_fly_optimized/storage.py",
        "exact_helper": ROOT / "tools/srq_fly_system_benchmark.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _set_stage(path: Path, stage: str) -> None:
    _atomic_json(path, {"pid": os.getpid(), "stage": stage, "time": time.time()})
    print(f"STAGE {stage}", flush=True)


class _NVMLSampler:
    """Small wrapper kept mockable for unit tests."""

    def __init__(self, device_index: int):
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
        self.device_name = pynvml.nvmlDeviceGetName(self._handle)
        if isinstance(self.device_name, bytes):
            self.device_name = self.device_name.decode("utf-8")

    def sample(self, pid: int) -> tuple[int, int | None]:
        device_bytes = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        processes = list(self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle))
        graphics = getattr(self._nvml, "nvmlDeviceGetGraphicsRunningProcesses", None)
        if graphics is not None:
            try:
                processes += list(graphics(self._handle))
            except self._nvml.NVMLError:
                # Compute-process attribution remains valid on devices that do
                # not expose a separate graphics-process query.
                pass
        used = [int(row.usedGpuMemory) for row in processes
                if int(row.pid) == int(pid) and row.usedGpuMemory is not None
                and int(row.usedGpuMemory) < 2**63]
        return device_bytes, (max(used) if used else None)

    def close(self) -> None:
        self._nvml.nvmlShutdown()


def _monitor_worker(process, marker: Path, sampler, interval: float) -> dict:
    baseline_device, _ = sampler.sample(process.pid)
    device_peak = baseline_device
    process_peak = 0
    samples = worker_samples = 0
    stage_peaks: dict[str, dict[str, int]] = {}
    observed_stages: set[str] = set()
    while process.poll() is None:
        stage = "process_start"
        if marker.is_file():
            try:
                stage = json.loads(marker.read_text(encoding="utf-8"))["stage"]
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        device_bytes, process_bytes = sampler.sample(process.pid)
        samples += 1
        device_peak = max(device_peak, device_bytes)
        observed_stages.add(stage)
        row = stage_peaks.setdefault(stage, {"device_bytes": 0, "process_bytes": 0})
        row["device_bytes"] = max(row["device_bytes"], device_bytes)
        if process_bytes is not None:
            worker_samples += 1
            process_peak = max(process_peak, process_bytes)
            row["process_bytes"] = max(row["process_bytes"], process_bytes)
        time.sleep(interval)
    # Include one terminal sample when the context is still visible.
    try:
        device_bytes, process_bytes = sampler.sample(process.pid)
        device_peak = max(device_peak, device_bytes)
        if process_bytes is not None:
            worker_samples += 1
            process_peak = max(process_peak, process_bytes)
    except Exception:
        pass
    return {
        "baseline_device_bytes": baseline_device,
        "peak_device_bytes": device_peak,
        "baseline_adjusted_peak_device_bytes": max(0, device_peak - baseline_device),
        "peak_worker_process_bytes": process_peak,
        "sample_count": samples,
        "worker_sample_count": worker_samples,
        "observed_stages": sorted(observed_stages),
        "stage_peaks": stage_peaks,
    }


@contextmanager
def _torch_stage(torch, device, marker: Path, stage: str, records: dict):
    _set_stage(marker, stage)
    torch.cuda.synchronize(device)
    before_allocated = int(torch.cuda.memory_allocated(device))
    before_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    yield
    torch.cuda.synchronize(device)
    records[stage] = {
        "seconds": time.perf_counter() - started,
        "before_allocated_bytes": before_allocated,
        "after_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "before_reserved_bytes": before_reserved,
        "after_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _projection_sha256(torch, projection) -> str:
    digest = hashlib.sha256()
    for tensor in (projection.ccol_indices(), projection.row_indices(), projection.values()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(bytes(str(tuple(value.shape)), "ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _build_train_loader(config: dict, root: Path, view_root: Path):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from utils.data_utils import build_transform, resolve_cifar100_directory

    source = Path(resolve_cifar100_directory(str(root))).resolve()
    for name in ("meta", "train"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"CIFAR training source missing {name}: {source}")
    if view_root.exists():
        shutil.rmtree(view_root)
    view_root.mkdir(parents=True)
    os.symlink(source, view_root / "cifar-100-python", target_is_directory=True)
    dataset = datasets.CIFAR100(
        root=str(view_root), train=True, download=False,
        # ``build_transform`` returns the repository's ordered transform list;
        # match ``load_dataset`` by converting it into one callable pipeline.
        transform=transforms.Compose([
            *build_transform(is_cifar=True, data_augmentation="vit")
        ]),
    )
    if len(dataset) != int(config["train_samples"]):
        raise RuntimeError("unexpected CIFAR training size")
    return DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config["num_workers"]), pin_memory=True,
    )


def _extract_train_features_to_cpu(torch, model, loader, device, expected_dim: int):
    """Extract frozen features without retaining ViT token storage on CUDA.

    A pooled CLS output can be a small view backed by the full token tensor.
    Keeping those views in a GPU list therefore retains roughly
    ``batch * tokens * dimension`` values per batch.  An explicit CPU copy
    breaks that storage alias while preserving the exact pooled values.
    """
    from tqdm.auto import tqdm

    feature_parts = []
    label_parts = []
    with torch.no_grad():
        for data, labels in tqdm(
            loader, desc="Extracting train features to CPU", leave=False
        ):
            data = data.to(device, non_blocking=True)
            embedding = model(data)
            if embedding.ndim != 2 or embedding.shape[1] != int(expected_dim):
                raise RuntimeError(
                    f"backbone output must have shape (B, {expected_dim}); "
                    f"got {tuple(embedding.shape)}"
                )
            feature_parts.append(
                embedding.detach().to(device="cpu", copy=True).contiguous()
            )
            label_parts.append(labels.detach().to(device="cpu", copy=True))
            del data, embedding
    return torch.cat(feature_parts, dim=0), torch.cat(label_parts, dim=0)


def _task_indices(torch, labels, config: dict):
    order = random.Random(int(config["seed"])).sample(
        list(range(int(config["num_classes"]))), int(config["num_classes"])
    )
    per_task = len(order) // int(config["num_tasks"])
    return order, [
        torch.isin(labels, torch.tensor(order[start:start + per_task])).nonzero().flatten()
        for start in range(0, len(order), per_task)
    ]


def _worker_config(config: dict) -> dict:
    rep = config["representation"]
    storage = config["storage"]
    return {
        "study": config["study_id"], "seed": config["seed"],
        "feature_dim": config["feature_dim"], "expand_dim": rep["expand_dim"],
        "synaptic_degree": rep["synaptic_degree"],
        "coding_level": rep["coding_level"], "ridge_lambda": config["ridge_lambda"],
        "block_size": storage["block_size"], "group_size": storage["group_size"],
        "num_tasks": config["num_tasks"], "rows_per_task": 1,
        "num_classes": config["num_classes"], "probe_rows": 256,
        "solver_tolerance": config["gates"]["maximum_solver_relative_residual"],
        "maximum_relative_logit_drift": 1.0,
        "update_panel_size": config["p2b_backend"]["update_panel_size"],
        "quantization_batch_blocks": config["p2b_backend"]["quantization_batch_blocks"],
    }


def run_worker(args) -> dict:
    import torch

    from models.backbone import load_model
    from tools import srq_fly_system_benchmark as system_benchmark
    from utils.train_utils import random_initialization

    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    if args.method not in METHODS or args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Priority-5 worker requires a locked method and CUDA")
    checkpoint = Path(args.backbone_checkpoint).resolve()
    if checkpoint.stat().st_size != int(config["checkpoint_size"]):
        raise RuntimeError("checkpoint size mismatch")
    if _sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    output = Path(args.output).resolve()
    marker = Path(args.stage_marker).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    random_initialization(int(config["seed"]))
    device = torch.device("cuda")
    records = {}
    _set_stage(marker, "dependency_import")

    with _torch_stage(torch, device, marker, "backbone_load", records):
        loader = _build_train_loader(
            config, Path(args.root),
            Path(args.scratch_dir).resolve() / args.method / "dataset_view",
        )
        backbone = load_model(
            config["model_name"], checkpoint_path=str(checkpoint),
            expected_checkpoint_size=int(config["checkpoint_size"]),
            expected_checkpoint_sha256=config["checkpoint_sha256"],
        ).eval().to(device)

    with _torch_stage(torch, device, marker, "feature_extraction", records):
        train_features, train_labels = _extract_train_features_to_cpu(
            torch, backbone, loader, device, int(config["feature_dim"])
        )
    del backbone, loader
    torch.cuda.empty_cache()
    if tuple(train_features.shape) != (config["train_samples"], config["feature_dim"]):
        raise RuntimeError("training feature shape mismatch")
    if not bool(torch.isfinite(train_features).all()):
        raise RuntimeError("non-finite training features")
    class_order, parts = _task_indices(torch, train_labels, config)

    worker_config = _worker_config(config)
    method_name = (
        "exact_fly_dense" if args.method == "exact_fly_10000"
        else "optimized_streaming_quant_blocked_qr_srq_int8"
    )
    with _torch_stage(torch, device, marker, "analytic_update", records):
        learner = system_benchmark._learner(method_name, worker_config, device)
        projection_sha = _projection_sha256(torch, learner.flyhash.projection_matrix)
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
                f"TASK {args.method} {task + 1}/{len(parts)} "
                f"seconds={task_seconds[-1]:.3f} "
                f"state={learner.persistent_state_bytes()}", flush=True,
            )

    with _torch_stage(torch, device, marker, "final_probe", records):
        probe_rows = min(512, len(train_features))
        probe_codes = learner.flyhash(
            train_features[:probe_rows].to(device),
            float(config["representation"]["coding_level"]), absolute_wta=False,
        ).to(torch.float32)
        logits = learner.predict_logits_from_codes(probe_codes)
        predictions = torch.tensor(
            [learner.class_ids[index] for index in logits.argmax(1).cpu().tolist()]
        )
        probe_logits = logits.detach().cpu()
        del probe_codes, logits

    result = {
        "schema_version": 1, "study_id": config["study_id"],
        "method": args.method, "status": "complete", "uses_test_set": False,
        "test_features_materialized": False, "train_samples": len(train_features),
        "class_order": class_order, "projection_sha256": projection_sha,
        "persistent_state_bytes": int(learner.persistent_state_bytes()),
        "solver_relative_residual": float(learner.diagnostics["solver_relative_residual"]),
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
        "config_sha256": _sha256(config_path),
        "source_identity": _source_identity(),
    }
    _atomic_json(output, result)
    _set_stage(marker, "complete")
    return result


def _summarize(config: dict, results: list[dict], monitors: list[dict]) -> dict:
    by_method = {row["method"]: row for row in results}
    monitor_by_method = {row["method"]: row for row in monitors}
    exact = by_method["exact_fly_10000"]
    srq = by_method["srq_fly_p2b_10000"]
    exact_logits = exact["probe_logits"]
    srq_logits = srq["probe_logits"]
    exact_predictions = exact["probe_predictions"]
    srq_predictions = srq["probe_predictions"]
    agreement = sum(a == b for a, b in zip(exact_predictions, srq_predictions)) / len(exact_predictions)
    numerator = sum((a - b) ** 2 for row_a, row_b in zip(exact_logits, srq_logits)
                    for a, b in zip(row_a, row_b)) ** 0.5
    denominator = max(sum(a * a for row in exact_logits for a in row) ** 0.5, 1.0)
    state_fraction = srq["persistent_state_bytes"] / exact["persistent_state_bytes"]
    torch_ratio = (
        srq["torch_cuda_stages"]["analytic_update"]["peak_allocated_bytes"]
        / exact["torch_cuda_stages"]["analytic_update"]["peak_allocated_bytes"]
    )
    exact_analytic_nvml = monitor_by_method["exact_fly_10000"].get(
        "stage_peaks", {}
    ).get("analytic_update", {}).get("process_bytes", 0)
    srq_analytic_nvml = monitor_by_method["srq_fly_p2b_10000"].get(
        "stage_peaks", {}
    ).get("analytic_update", {}).get("process_bytes", 0)
    nvml_ratio = (
        srq_analytic_nvml / exact_analytic_nvml
        if exact_analytic_nvml > 0 and srq_analytic_nvml > 0 else float("inf")
    )
    gates = config["gates"]
    checks = {
        "both_methods_complete": set(by_method) == set(METHODS)
        and all(row["status"] == "complete" for row in results),
        "train_only_boundary": all(not row["uses_test_set"] and not row["test_features_materialized"] for row in results),
        "paired_data_and_projection": exact["class_order"] == srq["class_order"]
        and exact["projection_sha256"] == srq["projection_sha256"]
        and exact["train_samples"] == srq["train_samples"] == config["train_samples"],
        "nvml_worker_observed": all(
            row["worker_sample_count"] >= config["nvml"]["minimum_worker_samples"]
            and row["peak_worker_process_bytes"] > 0 for row in monitors
        ),
        "required_stages_observed": all(
            {"backbone_load", "feature_extraction", "analytic_update", "final_probe"}
            <= set(row["observed_stages"]) for row in monitors
        ),
        "prediction_agreement": agreement >= gates["minimum_prediction_agreement"],
        "solver_stable": max(exact["solver_relative_residual"], srq["solver_relative_residual"])
        <= gates["maximum_solver_relative_residual"],
        "persistent_state_reduced": state_fraction <= gates["maximum_srq_state_fraction_of_exact"],
        "analytic_torch_peak_reduced": torch_ratio <= gates["maximum_srq_analytic_torch_peak_ratio"],
        "analytic_nvml_peak_reduced": nvml_ratio <= gates["maximum_srq_analytic_nvml_peak_ratio"],
    }
    return {
        "schema_version": 1, "study_id": config["study_id"],
        "status": "PASS_PRIORITY5_MEMORY" if all(checks.values()) else "STOP_PRIORITY5_MEMORY",
        "uses_test_set": False, "methods": results, "nvml": monitors,
        "comparisons": {
            "prediction_agreement": agreement,
            "relative_probe_logit_drift": numerator / denominator,
            "srq_state_fraction_of_exact": state_fraction,
            "srq_analytic_torch_peak_ratio": torch_ratio,
            "srq_analytic_nvml_process_peak_ratio": nvml_ratio,
            "whole_process_nvml_process_peak_ratio": (
                monitor_by_method["srq_fly_p2b_10000"]["peak_worker_process_bytes"]
                / monitor_by_method["exact_fly_10000"]["peak_worker_process_bytes"]
            ),
        },
        "gates": checks,
    }


def run_driver(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    if args.device != "cuda":
        raise ValueError("Priority 5 requires CUDA")
    if args.require_clean_git:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
        if dirty:
            raise RuntimeError(f"repository must be clean before Priority 5:\n{dirty}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results, monitors = [], []
    sampler = _NVMLSampler(config["nvml"]["device_index"])
    try:
        for method in METHODS:
            result_path = output_dir / f"{method}.json"
            marker = output_dir / f"{method}.stage.json"
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                "--config", str(config_path), "--method", method,
                "--root", args.root, "--backbone-checkpoint", args.backbone_checkpoint,
                "--output", str(result_path), "--stage-marker", str(marker),
                "--scratch-dir", str(Path(args.scratch_dir).resolve()),
                "--device", args.device,
            ]
            print(f"START isolated whole-process method={method}", flush=True)
            process = subprocess.Popen(command, cwd=ROOT)
            monitor = _monitor_worker(
                process, marker, sampler,
                float(config["nvml"]["poll_interval_seconds"]),
            )
            return_code = process.wait()
            if return_code != 0 or not result_path.is_file():
                raise RuntimeError(f"Priority-5 worker failed: {method}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            monitor.update(method=method, nvml_device_name=sampler.device_name)
            results.append(result)
            monitors.append(monitor)
            print(
                f"DONE {method} process_peak={monitor['peak_worker_process_bytes']} "
                f"state={result['persistent_state_bytes']}", flush=True,
            )
    finally:
        sampler.close()
    summary = _summarize(config, results, monitors)
    summary.update(
        config_sha256=_sha256(config_path), source_identity=_source_identity(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        measurement_note=(
            "NVML process peak is the primary whole-process metric. Device-wide "
            "and baseline-adjusted peaks are diagnostic because other Colab "
            "processes may share the GPU. PyTorch peaks are allocator-local."
        ),
    )
    _atomic_json(output_dir / "priority5_memory_results.json", summary)
    print(json.dumps({"status": summary["status"], **summary["comparisons"]}, indent=2))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", required=True)
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
        run_driver(args)


if __name__ == "__main__":
    main()
