"""Isolated SRQ-FLY update-time and CUDA-memory benchmark.

Each implementation runs in a fresh Python process so CUDA allocator history
from one method cannot contaminate another method's peak.  The stream is
synthetic and contains no dataset test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.srq_fly import SquareRootFLYLearner as LockedLearner
from methods.srq_fly_optimized import SquareRootFLYLearner as OptimizedLearner
from models.flyhash import FlyHash
from tools import srq_fly_update_benchmark as update_benchmark


METHODS = (
    "exact_fly_dense",
    "locked_srq_int8",
    "optimized_gram_srq_int8",
    "optimized_direct_srq_int8",
    "optimized_blocked_qr_srq_int8",
    "optimized_chunked_blocked_qr_srq_int8",
    "optimized_qr_srq_int8",
)
EXPERIMENTAL_WORKER_METHODS = (
    "optimized_eager_quant_blocked_qr_srq_int8",
    "optimized_streaming_quant_blocked_qr_srq_int8",
)
WORKER_METHODS = METHODS + EXPERIMENTAL_WORKER_METHODS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    return (
        tensor.values().numel() * tensor.values().element_size()
        + tensor.ccol_indices().numel() * tensor.ccol_indices().element_size()
        + tensor.row_indices().numel() * tensor.row_indices().element_size()
    )


class _ExactSyntheticFLY:
    """Dense sufficient-statistic control on a materialized WTA stream."""

    def __init__(self, config: dict, device: torch.device) -> None:
        self.device = device
        self.dimension = int(config["expand_dim"])
        self.ridge_lambda = float(config["ridge_lambda"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config["seed"]))
            self.flyhash = FlyHash(
                int(config["feature_dim"]), self.dimension,
                int(config["synaptic_degree"]),
            ).to(device)
        if self.flyhash.projection_matrix.layout != torch.sparse_csc:
            self.flyhash.to_sparse()
        self.gram = torch.zeros((self.dimension, self.dimension), device=device)
        self.Q = torch.zeros((self.dimension, 0), device=device)
        self.counts = torch.zeros(0, device=device)
        self.class_ids: list[int] = []
        self.weights = None
        self.diagnostics = {"solver_relative_residual": 0.0}

    def update_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        updated_ids = sorted(set(self.class_ids) | set(map(int, labels.cpu().tolist())))
        columns = {value: index for index, value in enumerate(updated_ids)}
        cross = torch.zeros((self.dimension, len(updated_ids)), device=self.device)
        counts = torch.zeros(len(updated_ids), device=self.device)
        for old_column, class_id in enumerate(self.class_ids):
            cross[:, columns[class_id]] = self.Q[:, old_column]
            counts[columns[class_id]] = self.counts[old_column]
        target_columns = torch.tensor(
            [columns[int(value)] for value in labels.cpu().tolist()],
            device=self.device,
        )
        targets = torch.nn.functional.one_hot(
            target_columns, num_classes=len(updated_ids)
        ).to(torch.float32)
        self.gram.addmm_(codes.T, codes)
        cross.addmm_(codes.T, targets)
        counts.add_(targets.sum(0))
        system = self.gram.clone()
        system.diagonal().add_(self.ridge_lambda)
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info.max().item()) != 0:
            raise RuntimeError("exact synthetic FLY Cholesky failed")
        self.weights = torch.cholesky_solve(cross, factor)
        residual = torch.linalg.vector_norm(system @ self.weights - cross) / max(
            float(torch.linalg.vector_norm(cross)), 1.0
        )
        self.diagnostics["solver_relative_residual"] = float(residual)
        self.class_ids, self.Q, self.counts = updated_ids, cross, counts

    def predict_logits_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return codes @ self.weights

    def persistent_state_bytes(self) -> int:
        tensors = [
            self.flyhash.projection_matrix, self.gram, self.Q, self.counts,
            self.weights,
        ]
        return sum(_tensor_bytes(value) for value in tensors if value is not None)

    def state_dict(self) -> dict:
        return {
            "method": "exact_fly_dense", "projection": self.flyhash.projection_matrix,
            "gram": self.gram, "Q": self.Q, "counts": self.counts,
            "class_ids": self.class_ids, "weights": self.weights,
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _learner(
    method: str, config: dict, device: torch.device, *, profile_updates: bool = False
):
    if method == "exact_fly_dense":
        return _ExactSyntheticFLY(config, device)
    kwargs = update_benchmark._learner_kwargs(config, device)
    if method == "locked_srq_int8":
        return LockedLearner(storage_mode="int8", **kwargs)
    backends = {
        "optimized_gram_srq_int8": "gram_cholesky",
        "optimized_direct_srq_int8": "gram_cholesky_direct",
        "optimized_blocked_qr_srq_int8": "blocked_qr",
        "optimized_chunked_blocked_qr_srq_int8": "blocked_qr",
        "optimized_qr_srq_int8": "stacked_qr",
        "optimized_eager_quant_blocked_qr_srq_int8": "blocked_qr",
        "optimized_streaming_quant_blocked_qr_srq_int8": "blocked_qr",
    }
    if method not in backends:
        raise ValueError(f"unknown method: {method}")
    return OptimizedLearner(
        storage_mode="int8",
        update_backend=backends[method],
        update_panel_size=int(config.get("update_panel_size", 128)),
        update_trailing_chunk_size=(
            int(config.get("update_trailing_chunk_size", 1024))
            if method == "optimized_chunked_blocked_qr_srq_int8"
            else None
        ),
        quantization_backend=(
            "streaming"
            if method == "optimized_streaming_quant_blocked_qr_srq_int8"
            else "eager"
        ),
        quantization_batch_blocks=int(config.get("quantization_batch_blocks", 16)),
        profile_updates=profile_updates,
        **kwargs,
    )


def _memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_worker(
    *, config_path: Path, method: str, output: Path, probe_output: Path,
    device_name: str, profile_stages: bool = True,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    update_benchmark._validate(config)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    learner = _learner(method, config, device)
    generator = torch.Generator().manual_seed(int(config["seed"]) + 17)
    task_seconds: list[float] = []
    task_memory: list[dict] = []
    last_codes = None
    profile_stream: list[tuple[torch.Tensor, torch.Tensor]] | None = (
        []
        if profile_stages and method in {
            "optimized_blocked_qr_srq_int8",
            "optimized_chunked_blocked_qr_srq_int8",
            "optimized_eager_quant_blocked_qr_srq_int8",
            "optimized_streaming_quant_blocked_qr_srq_int8",
        }
        else None
    )

    if device.type == "cuda":
        # Initialize CUDA libraries before resetting the measured peaks.
        warm = torch.ones((8, 8), device=device)
        _ = warm @ warm
        _sync(device)
        del warm
        torch.cuda.empty_cache()
    baseline = _memory_snapshot(device)

    for task in range(int(config["num_tasks"])):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        generated_codes = update_benchmark._codes(config, generator)
        generated_labels = (
            torch.arange(int(config["rows_per_task"]))
            + task * int(config["rows_per_task"])
        ) % int(config["num_classes"])
        if profile_stream is not None:
            profile_stream.append((generated_codes, generated_labels))
        codes = generated_codes.to(device)
        labels = generated_labels.to(device)
        pending_probe = (
            generated_codes[: int(config["probe_rows"])].detach().clone()
            if task == int(config["num_tasks"]) - 1
            else None
        )
        _sync(device)
        started = time.perf_counter()
        if method in {
            "optimized_chunked_blocked_qr_srq_int8",
            "optimized_eager_quant_blocked_qr_srq_int8",
            "optimized_streaming_quant_blocked_qr_srq_int8",
        }:
            learner.update_codes_consuming(codes, labels)
        else:
            learner.update_codes(codes, labels)
        _sync(device)
        elapsed = time.perf_counter() - started
        task_seconds.append(elapsed)
        memory = _memory_snapshot(device)
        memory.update(task=task + 1)
        task_memory.append(memory)
        if pending_probe is not None:
            last_codes = pending_probe.to(device)
        print(
            f"TASK method={method} {task + 1}/{config['num_tasks']} "
            f"update={elapsed:.4f}s "
            f"peak_allocated={memory['peak_allocated_bytes']}",
            flush=True,
        )
        del generated_codes, generated_labels, codes, labels

    if last_codes is None:
        raise RuntimeError("empty synthetic stream")
    logits = learner.predict_logits_from_codes(last_codes).detach().cpu()
    probe_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits, probe_output)
    checkpoint_path = output.with_suffix(".checkpoint.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(learner.state_dict(), checkpoint_path)
    checkpoint_bytes = checkpoint_path.stat().st_size
    checkpoint_path.unlink()
    persistent_state_bytes = learner.persistent_state_bytes()
    solver_relative_residual = learner.diagnostics["solver_relative_residual"]
    peak_allocated = [row["peak_allocated_bytes"] for row in task_memory]
    peak_reserved = [row["peak_reserved_bytes"] for row in task_memory]
    profiled_stage_seconds = None
    profiled_stage_cuda_memory = None
    if profile_stream is not None:
        # The timed learner must not coexist with the diagnostic learner.
        # Otherwise absolute per-stage peaks include an unrelated model/state.
        del learner, last_codes
        if device.type == "cuda":
            _sync(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        profiled = _learner(method, config, device, profile_updates=True)
        profiled_stage_seconds = []
        profiled_stage_cuda_memory = []
        for profile_codes, profile_labels in profile_stream:
            values = profile_codes.to(device)
            labels = profile_labels.to(device)
            if method in {
                "optimized_chunked_blocked_qr_srq_int8",
                "optimized_eager_quant_blocked_qr_srq_int8",
                "optimized_streaming_quant_blocked_qr_srq_int8",
            }:
                profiled.update_codes_consuming(values, labels)
            else:
                profiled.update_codes(values, labels)
            profiled_stage_seconds.append(
                dict(profiled.diagnostics["last_update_stage_seconds"])
            )
            profiled_stage_cuda_memory.append(
                dict(profiled.diagnostics["last_update_stage_cuda_memory"])
            )
        del profiled

    result = {
        "schema_version": 1,
        "status": "complete",
        "method": method,
        "uses_test_set": False,
        "synthetic_only": True,
        "config_sha256": _sha256(config_path),
        "source_identity": {
            "runner": _sha256(Path(__file__).resolve()),
            "optimized_learner": _sha256(
                ROOT / "methods/srq_fly_optimized/learner.py"
            ),
            "optimized_storage": _sha256(
                ROOT / "methods/srq_fly_optimized/storage.py"
            ),
        },
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "device_total_memory_bytes": (
            int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda" else None
        ),
        "task_update_seconds": task_seconds,
        "profiled_task_stage_seconds": profiled_stage_seconds,
        "profiled_task_stage_cuda_memory": profiled_stage_cuda_memory,
        "total_update_seconds": sum(task_seconds),
        "baseline_cuda_memory": baseline,
        "task_cuda_memory": task_memory,
        "peak_cuda_allocated_bytes": (
            max(peak_allocated) if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            max(peak_reserved) if device.type == "cuda" else None
        ),
        "persistent_state_bytes": persistent_state_bytes,
        "serialized_checkpoint_bytes": checkpoint_bytes,
        "solver_relative_residual": solver_relative_residual,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_isolated(
    *, config_path: Path, output_dir: Path, device_name: str,
    methods: tuple[str, ...] = METHODS,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    update_benchmark._validate(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    probes = {}
    for method in methods:
        result_path = output_dir / f"{method}.json"
        probe_path = output_dir / f"{method}.probe.pt"
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()), "worker",
            "--config", str(config_path), "--method", method,
            "--output", str(result_path), "--probe-output", str(probe_path),
            "--device", device_name,
        ]
        print(f"START isolated method={method}", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"isolated worker failed: {method}")
        results.append(json.loads(result_path.read_text(encoding="utf-8")))
        probes[method] = torch.load(probe_path, weights_only=True, map_location="cpu")
        probe_path.unlink()
        print(f"DONE isolated method={method}", flush=True)

    reference = probes["locked_srq_int8"]
    denominator = max(float(torch.linalg.vector_norm(reference)), 1.0)
    drift = {
        method: float(torch.linalg.vector_norm(values - reference)) / denominator
        for method, values in probes.items() if method != "exact_fly_dense"
    }
    by_method = {row["method"]: row for row in results}
    tolerance = float(config["maximum_relative_logit_drift"])
    gates = {
        "all_workers_complete": len(results) == len(methods)
        and all(row["status"] == "complete" for row in results),
        "compatibility_backend_bitwise_exact":
            drift.get("optimized_gram_srq_int8") == 0.0,
        "direct_backend_within_tolerance":
            drift.get("optimized_direct_srq_int8", float("inf")) <= tolerance,
        "blocked_qr_backend_within_tolerance":
            drift.get("optimized_blocked_qr_srq_int8", float("inf")) <= tolerance,
        "chunked_blocked_qr_backend_within_tolerance":
            drift.get(
                "optimized_chunked_blocked_qr_srq_int8", float("inf")
            ) <= tolerance,
        "qr_backend_within_tolerance":
            drift.get("optimized_qr_srq_int8", float("inf")) <= tolerance,
        "persistent_state_unchanged": all(
            row["persistent_state_bytes"]
            == by_method["locked_srq_int8"]["persistent_state_bytes"]
            for row in results if row["method"] != "exact_fly_dense"
        ),
        "solver_stable": all(
            row["solver_relative_residual"] <= config["solver_tolerance"]
            for row in results
        ),
        "optimized_update_ratio_to_exact_fly":
            by_method["optimized_blocked_qr_srq_int8"]["total_update_seconds"]
            / by_method["exact_fly_dense"]["total_update_seconds"]
            <= config.get("maximum_update_ratio_to_exact_fly", 1.5),
    }
    summary = {
        "schema_version": 1,
        "study": config["study"],
        "status": "pass" if all(gates.values()) else "fail",
        "uses_test_set": False,
        "synthetic_only": True,
        "config_sha256": _sha256(config_path),
        "source_identity": {
            "runner": _sha256(Path(__file__).resolve()),
            "locked_learner": _sha256(ROOT / "methods/srq_fly/learner.py"),
            "optimized_learner": _sha256(ROOT / "methods/srq_fly_optimized/learner.py"),
            "optimized_storage": _sha256(ROOT / "methods/srq_fly_optimized/storage.py"),
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "measurement_scope": {
            "peak_cuda": "PyTorch allocator peak during analytic update; feature extraction excluded",
            "persistent_state": "learner tensors after final task",
            "serialized_checkpoint": "temporary torch.save byte count",
        },
        "results": results,
        "relative_logit_drift_from_locked": drift,
        "speedup_over_locked": {
            method: by_method["locked_srq_int8"]["total_update_seconds"]
            / row["total_update_seconds"]
            for method, row in by_method.items() if method != "locked_srq_int8"
        },
        "selected_update_backend": {
            "name": "blocked_qr",
            "panel_size": int(config.get("update_panel_size", 128)),
        },
        "optimized_direct_update_ratio_to_exact_fly":
            by_method["optimized_direct_srq_int8"]["total_update_seconds"]
            / by_method["exact_fly_dense"]["total_update_seconds"],
        "optimized_blocked_qr_update_ratio_to_exact_fly":
            by_method["optimized_blocked_qr_srq_int8"]["total_update_seconds"]
            / by_method["exact_fly_dense"]["total_update_seconds"],
        "optimized_chunked_blocked_qr_update_ratio_to_exact_fly":
            by_method["optimized_chunked_blocked_qr_srq_int8"][
                "total_update_seconds"
            ] / by_method["exact_fly_dense"]["total_update_seconds"],
        "gates": gates,
    }
    (output_dir / "system_benchmark.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "pass":
        raise RuntimeError("isolated SRQ-FLY system benchmark gate failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--method", choices=WORKER_METHODS, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--probe-output", type=Path, required=True)
    worker.add_argument("--device", default="cpu")
    worker.add_argument("--skip-stage-profile", action="store_true")
    driver = subparsers.add_parser("run")
    driver.add_argument("--config", type=Path, required=True)
    driver.add_argument("--output-dir", type=Path, required=True)
    driver.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "worker":
        run_worker(
            config_path=args.config.resolve(), method=args.method,
            output=args.output.resolve(), probe_output=args.probe_output.resolve(),
            device_name=args.device, profile_stages=not args.skip_stage_profile,
        )
    else:
        run_isolated(
            config_path=args.config.resolve(), output_dir=args.output_dir.resolve(),
            device_name=args.device,
        )


if __name__ == "__main__":
    main()
