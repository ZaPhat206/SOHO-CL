"""M20: train-only budget sensitivity and selection criterion for SRQ-Adaptive.

Three CIFAR-100 validation streams with the RanPAC random-ReLU head at width
20,000.  Every unit (stream x method x budget) is atomic and resumable.  The
factor-MSE budget sweep runs the locked ``SquareRootBackend``; the
row-weighted system criterion runs ``CriterionAdaptiveSquareRootBackend``,
whose factor-MSE mode is tested to reproduce the locked backend bitwise.  One
benchmark unit times per-block against batched factor-MSE scoring on the real
factors of one stream.  Nothing here reads a test split, and no accuracy value
selects a method, budget, seed, or hyperparameter.
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
import random
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import (  # noqa: E402
    ExactGramBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from methods.analytic_ridge.adaptive_selection import (  # noqa: E402
    CRITERIA,
    CriterionAdaptiveSquareRootBackend,
    batched_factor_mse_benefits,
    budget_bytes,
    greedy_select,
    score_blocks,
)
from tools import srq_generalization_m4 as m4  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools import srq_generalization_m6 as m6  # noqa: E402
from tools import srq_generalization_m11 as m11  # noqa: E402
from tools.experiment_runner import (  # noqa: E402
    split,
    train_validation_indices,
    validate_cache,
)


STATUS_PASS = "PASS_M20_ADAPTIVE_BUDGET_CRITERION_TRAIN_ONLY"
STATUS_WARNING = "COMPLETE_M20_ADAPTIVE_BUDGET_CRITERION_WITH_WARNINGS"
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name", "checkpoint_sha256",
    "train_cache_sha256", "uses_test_set", "accuracy_based_selection", "seed",
    "num_classes", "num_tasks", "outer_validation_fraction", "statistics_dtype",
    "solver_dtype", "width", "ridge_lambda", "required_device_name_substring",
    "streams", "source_m6", "ranpac", "p2b", "budget_sweep",
    "scoring_benchmark", "interpretation", "gates",
}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _sha256(path: str | Path, *, source: bool = False) -> str:
    digest = hashlib.sha256()
    if source:
        digest.update(Path(path).read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _budget_key(budget: float) -> str:
    return f"{float(budget):.2f}"


def validate_config(config: dict) -> dict:
    if set(config) != TOP_KEYS:
        raise ValueError(
            f"M20 config keys mismatch: missing={sorted(TOP_KEYS - set(config))}, "
            f"unexpected={sorted(set(config) - TOP_KEYS)}"
        )
    if config["schema_version"] != 1 or int(config["seed"]) != 2025:
        raise ValueError("M20 requires schema 1 and protocol seed 2025")
    if config["uses_test_set"] is not False or config["accuracy_based_selection"] is not False:
        raise ValueError("M20 must stay train-only and must not select from accuracy")
    if (
        config["dataset"] != "CIFAR-100"
        or config["model_name"] != "vit_base_patch16_224"
        or config["statistics_dtype"] != "float32"
        or config["solver_dtype"] != "float32"
        or int(config["num_classes"]) != 100
        or int(config["num_tasks"]) != 10
        or int(config["width"]) != 20000
        or float(config["ridge_lambda"]) != 1.0e6
        or not 0 < float(config["outer_validation_fraction"]) < 1
    ):
        raise ValueError("M20 no longer matches the locked M6 width-20,000 stream")
    for key in ("checkpoint_sha256", "train_cache_sha256"):
        if len(str(config[key])) != 64:
            raise ValueError(f"invalid {key}")

    streams = config["streams"]
    identifiers = [stream["stream_id"] for stream in streams]
    if len(streams) < 1 or len(set(identifiers)) != len(identifiers):
        raise ValueError("M20 streams must be non-empty and unique")
    for stream in streams:
        if set(stream) != {
            "stream_id", "class_order_seed", "split_seed", "projection_seed",
            "m6_identity_locked",
        }:
            raise ValueError("M20 stream fields mismatch")
    locked = [stream for stream in streams if stream["m6_identity_locked"]]
    if (
        len(locked) != 1
        or streams[0] is not locked[0]
        or any(int(locked[0][key]) != 2025 for key in ("class_order_seed", "split_seed", "projection_seed"))
    ):
        raise ValueError("M20 needs exactly one first M6-locked stream with seeds 2025")

    source = config["source_m6"]
    if set(source) != {
        "artifact_filename", "artifact_sha256", "result_member", "result_sha256",
        "required_status",
    } or source["required_status"] != "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY":
        raise ValueError("invalid M20 source-M6 lock")

    ranpac = config["ranpac"]
    if set(ranpac) != {
        "path", "feature_dimension", "maximum_expand_dimension",
        "projection_distribution", "activation", "encode_batch_size",
        "evaluation_batch_size",
    } or (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or int(ranpac["feature_dimension"]) != 768
        or int(ranpac["maximum_expand_dimension"]) != 20000
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or min(int(ranpac["encode_batch_size"]), int(ranpac["evaluation_batch_size"])) <= 0
    ):
        raise ValueError("invalid M20 RanPAC semantics")

    p2b = config["p2b"]
    if set(p2b) != {
        "block_size", "group_size", "update_panel_size", "update_trailing_chunk_size",
        "first_update_backend", "quantization_backend", "quantization_batch_blocks",
    } or (
        int(p2b["block_size"]) != 256
        or int(p2b["group_size"]) != 64
        or int(p2b["update_panel_size"]) != 128
        or p2b["update_trailing_chunk_size"] is not None
        or p2b["first_update_backend"] != "gram_cholesky"
        or p2b["quantization_backend"] != "streaming"
        or int(p2b["quantization_batch_blocks"]) != 64
    ):
        raise ValueError("M20 must use the locked P2B storage and update settings")

    sweep = config["budget_sweep"]
    if set(sweep) != {
        "factor_mse_budgets", "row_weighted_system_budgets", "reported_locked_budget",
        "precision_mask_dtype", "tie_break", "selection_signal",
    }:
        raise ValueError("M20 budget sweep fields mismatch")
    for name in ("factor_mse_budgets", "row_weighted_system_budgets"):
        budgets = [float(value) for value in sweep[name]]
        if (
            not budgets
            or budgets != sorted(budgets)
            or len(set(budgets)) != len(budgets)
            or any(not 0.0 <= value <= 1.0 for value in budgets)
        ):
            raise ValueError(f"{name} must be sorted, unique and inside [0, 1]")
    if float(sweep["reported_locked_budget"]) != 0.25 or 0.25 not in [
        float(value) for value in sweep["factor_mse_budgets"]
    ]:
        raise ValueError("M20 must include the locked 25% budget in the factor-MSE sweep")
    if (
        sweep["precision_mask_dtype"] != "uint8"
        or sweep["tie_break"] != "ascending_upper_block_index"
        or sweep["selection_signal"] != "current_factor_values_only_no_labels_or_accuracy"
    ):
        raise ValueError("M20 adaptive policy fields changed")

    benchmark = config["scoring_benchmark"]
    if set(benchmark) != {"stream_id", "criterion", "budget", "batch_blocks"} or (
        benchmark["stream_id"] not in identifiers
        or benchmark["criterion"] != "factor_mse"
        or float(benchmark["budget"]) != 0.25
        or int(benchmark["batch_blocks"]) <= 0
    ):
        raise ValueError("invalid M20 scoring benchmark")

    interpretation = config["interpretation"]
    if set(interpretation) != {
        "sufficient_budget_maximum_aia_loss_pp", "criterion_comparison_budgets",
        "criterion_supported_rule",
    }:
        raise ValueError("M20 interpretation fields mismatch")
    comparison = [float(value) for value in interpretation["criterion_comparison_budgets"]]
    if not comparison or any(
        value not in [float(v) for v in sweep["factor_mse_budgets"]]
        or value not in [float(v) for v in sweep["row_weighted_system_budgets"]]
        for value in comparison
    ):
        raise ValueError("comparison budgets must be run under both criteria")

    gates = config["gates"]
    if set(gates) != {
        "maximum_solver_relative_residual", "maximum_m6_reproduction_accuracy_difference_pp",
        "require_m6_identity_for_locked_stream", "require_budget_conformance",
        "require_state_monotone_in_budget", "require_criterion_backend_fidelity",
        "require_batched_scoring_decision_agreement", "require_t4_device", "accuracy_gate",
    } or gates["accuracy_gate"] is not None:
        raise ValueError("M20 gate fields mismatch or an accuracy gate was added")
    return config


def read_config(path: str | Path) -> dict:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def unit_plan(config: dict) -> list[dict]:
    """Fixed unit order; Exact precedes every unit that compares against it."""
    sweep = config["budget_sweep"]
    benchmark = config["scoring_benchmark"]
    plan = []
    for stream in config["streams"]:
        sid = stream["stream_id"]
        plan.append({"unit_id": f"{sid}__exact", "stream_id": sid, "method": "exact",
                     "criterion": None, "budget": None})
        plan.append({"unit_id": f"{sid}__p2b_int8", "stream_id": sid, "method": "p2b_int8",
                     "criterion": None, "budget": None})
        for budget in sweep["factor_mse_budgets"]:
            plan.append({"unit_id": f"{sid}__factor_mse_b{_budget_key(budget)}",
                         "stream_id": sid, "method": "adaptive_locked",
                         "criterion": "factor_mse", "budget": float(budget)})
        for budget in sweep["row_weighted_system_budgets"]:
            plan.append({"unit_id": f"{sid}__row_weighted_system_b{_budget_key(budget)}",
                         "stream_id": sid, "method": "adaptive_criterion",
                         "criterion": "row_weighted_system", "budget": float(budget)})
        if sid == benchmark["stream_id"]:
            plan.append({
                "unit_id": f"{sid}__benchmark_factor_mse_b{_budget_key(benchmark['budget'])}",
                "stream_id": sid, "method": "adaptive_criterion_benchmark",
                "criterion": "factor_mse", "budget": float(benchmark["budget"]),
            })
    return plan


def _unit_path(output_dir: Path, unit_id: str) -> Path:
    return output_dir / "units" / f"{unit_id}.json"


def _exact_cache_path(output_dir: Path, stream_id: str) -> Path:
    return output_dir / "cache" / f"{stream_id}_exact_weights.pt"


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _lock_precision() -> dict:
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    return {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def assert_train_only_cache(feature_cache_dir: Path) -> None:
    if (Path(feature_cache_dir) / "test.pt").exists():
        raise RuntimeError("M20 refuses a feature cache that contains test.pt")


def _stream_by_id(config: dict, stream_id: str) -> dict:
    for stream in config["streams"]:
        if stream["stream_id"] == stream_id:
            return stream
    raise KeyError(stream_id)


def prepare_stream(config: dict, stream: dict, labels: torch.Tensor, device: torch.device) -> dict:
    num_classes = int(config["num_classes"])
    class_order = random.Random(int(stream["class_order_seed"])).sample(
        list(range(num_classes)), num_classes
    )
    task_indices = split(labels, class_order, int(config["num_tasks"]))
    training_parts, validation_parts = train_validation_indices(
        labels, task_indices, int(stream["split_seed"]),
        float(config["outer_validation_fraction"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(stream["projection_seed"]))
    full_projection = torch.randn(
        int(config["ranpac"]["feature_dimension"]),
        int(config["ranpac"]["maximum_expand_dimension"]),
        generator=generator, dtype=torch.float32,
    )
    projection = full_projection[:, : int(config["width"])].contiguous()
    return {
        "class_order": class_order,
        "training_parts": training_parts,
        "validation_parts": validation_parts,
        "full_projection": full_projection,
        "projection": projection.to(device),
    }


def stream_identity(config: dict, prepared: dict, source: dict) -> dict:
    provenance = source["provenance"]
    width = str(int(config["width"]))
    return {
        "class_order": prepared["class_order"] == provenance["class_order"],
        "training_indices": m6._sequence_sha256(prepared["training_parts"])
        == provenance["training_indices_sha256"],
        "validation_indices": m6._sequence_sha256(prepared["validation_parts"])
        == provenance["outer_validation_indices_sha256"],
        "full_projection": m4._tensor_content_sha256(prepared["full_projection"])
        == provenance["full_projection_sha256"],
        "projection_prefix": m4._tensor_content_sha256(
            prepared["full_projection"][:, : int(width)].contiguous()
        ) == provenance["projection_prefix_sha256"][width],
    }


def _common_backend(config: dict, device: torch.device) -> dict:
    return m5._common_backend(int(config["width"]), float(config["ridge_lambda"]), device)


def _p2b_kwargs(config: dict) -> dict:
    p2b = config["p2b"]
    return {
        "block_size": int(p2b["block_size"]),
        "group_size": int(p2b["group_size"]),
        "update_panel_size": int(p2b["update_panel_size"]),
        "update_trailing_chunk_size": p2b["update_trailing_chunk_size"],
        "first_update_backend": p2b["first_update_backend"],
        "quantization_backend": p2b["quantization_backend"],
        "quantization_batch_blocks": int(p2b["quantization_batch_blocks"]),
    }


def build_backend(config: dict, unit: dict, device: torch.device, hook=None):
    common = _common_backend(config, device)
    if unit["method"] == "exact":
        return ExactGramBackend(**common)
    if unit["method"] == "p2b_int8":
        return SquareRootBackend(storage_mode="int8", **common, **_p2b_kwargs(config))
    if unit["method"] == "adaptive_locked":
        if unit["criterion"] != "factor_mse":
            raise ValueError("the locked backend implements only factor_mse")
        return SquareRootBackend(
            storage_mode="adaptive_int8_fp16", adaptive_budget_fraction=unit["budget"],
            **common, **_p2b_kwargs(config),
        )
    if unit["method"] in {"adaptive_criterion", "adaptive_criterion_benchmark"}:
        if unit["criterion"] not in CRITERIA:
            raise ValueError("unknown criterion")
        return CriterionAdaptiveSquareRootBackend(
            selection_criterion=unit["criterion"], adaptive_budget_fraction=unit["budget"],
            pre_compression_hook=hook, **common, **_p2b_kwargs(config),
        )
    raise ValueError(f"unknown M20 method {unit['method']}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_scoring_hook(config: dict, device: torch.device, records: list[dict]):
    """Time per-block and batched factor-MSE scoring on the unquantized factor."""
    p2b = config["p2b"]
    benchmark = config["scoring_benchmark"]
    block_size, group_size = int(p2b["block_size"]), int(p2b["group_size"])

    def hook(factor: torch.Tensor) -> None:
        _sync(device)
        started = time.perf_counter()
        loop = score_blocks(factor, block_size=block_size, group_size=group_size,
                            criterion="factor_mse")
        _sync(device)
        loop_seconds = time.perf_counter() - started
        loop_benefits = loop["benefits"]
        loop_costs = loop["extra_costs"]
        counts = loop["value_counts"]
        del loop
        gc.collect()
        _sync(device)
        started = time.perf_counter()
        batched, costs, batched_counts = batched_factor_mse_benefits(
            factor, block_size=block_size, group_size=group_size,
            batch_blocks=int(benchmark["batch_blocks"]),
        )
        _sync(device)
        batched_seconds = time.perf_counter() - started
        _, _, extra = budget_bytes(counts, group_size=group_size,
                                   budget_fraction=float(benchmark["budget"]))
        loop_selection, _ = greedy_select(loop_benefits, loop_costs, extra)
        batched_selection, _ = greedy_select(batched, costs, extra)
        differences = [
            abs(a - b) / max(abs(b), 1e-30) for a, b in zip(batched, loop_benefits)
        ]
        records.append({
            "task": len(records) + 1,
            "per_block_scoring_seconds": loop_seconds,
            "batched_scoring_seconds": batched_seconds,
            "costs_identical": costs == loop_costs and batched_counts == counts,
            "selections_identical": batched_selection == loop_selection,
            "differing_selected_blocks": sum(
                a != b for a, b in zip(batched_selection, loop_selection)
            ),
            "maximum_relative_benefit_difference": max(differences) if differences else 0.0,
        })
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return hook


def _mask_sha256(backend) -> str | None:
    factor = getattr(backend, "factor", None)
    mask = getattr(factor, "precision_mask", None)
    if mask is None:
        return None
    return hashlib.sha256(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_unit(
    config: dict, unit: dict, *, train: dict, source: dict, output_dir: Path,
    device: torch.device,
) -> dict:
    precision = _lock_precision()
    stream = _stream_by_id(config, unit["stream_id"])
    prepared = prepare_stream(config, stream, train["labels"], device)
    identity = stream_identity(config, prepared, source) if stream["m6_identity_locked"] else None
    projection = prepared["projection"]
    encoder = lambda values: torch.relu(values.to(device=device, dtype=torch.float32) @ projection)

    exact_weights = None
    if unit["method"] != "exact":
        cache = _exact_cache_path(output_dir, unit["stream_id"])
        if not cache.is_file():
            raise RuntimeError(f"M20 unit {unit['unit_id']} needs the Exact unit of its stream first")
        exact_weights = torch.load(cache, weights_only=True)

    benchmark_records: list[dict] = []
    hook = (
        make_scoring_hook(config, device, benchmark_records)
        if unit["method"] == "adaptive_criterion_benchmark" else None
    )
    backend = build_backend(config, unit, device, hook=hook)
    records = []
    saved_weights = []
    for task_id, train_indices in enumerate(prepared["training_parts"]):
        codes = m5._encode_indices(
            encoder, train["features"], train_indices, int(config["ranpac"]["encode_batch_size"])
        )
        _sync(device)
        started = time.perf_counter()
        backend.update(codes, train["labels"][train_indices])
        _sync(device)
        update_seconds = time.perf_counter() - started
        del codes
        seen = torch.cat(prepared["validation_parts"][: task_id + 1])
        accuracy = m5._evaluate(
            encoder=encoder, backends={"unit": backend}, features=train["features"],
            labels=train["labels"], indices=seen,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )["unit"]
        tensors = {"projection": projection}
        tensors.update(backend.persistent_tensors())
        factor_bytes = persistent_tensor_bytes(
            {name: tensor for name, tensor in backend.persistent_tensors().items()
             if name.startswith("factor.")}
        )
        diagnostics = backend.diagnostics
        record = {
            "task": task_id + 1,
            "validation_accuracy_percent": accuracy,
            "total_persistent_bytes": persistent_tensor_bytes(tensors),
            "factor_persistent_bytes": factor_bytes,
            "solver_relative_residual": float(diagnostics.get("solver_relative_residual", 0.0)),
            "update_seconds": update_seconds,
            "class_ids": [int(value) for value in backend.class_ids],
            "weights_sha256": _tensor_sha256(backend.weights),
        }
        if unit["method"] == "exact":
            saved_weights.append(backend.weights.detach().to("cpu").clone())
            record["relative_classifier_error_vs_exact"] = 0.0
        else:
            reference = exact_weights["weights"][task_id].to(device)
            if exact_weights["class_ids"][task_id] != record["class_ids"]:
                raise AssertionError("M20 class columns differ from the Exact unit")
            record["relative_classifier_error_vs_exact"] = float(
                torch.linalg.vector_norm(backend.weights - reference).item()
                / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)
            )
            record["relative_local_factor_error"] = float(
                diagnostics.get("relative_local_factor_error", float("nan"))
            )
        if unit["method"] not in {"exact", "p2b_int8"}:
            record.update({
                "precision_mask_sha256": _mask_sha256(backend),
                "factor_budget_ceiling_bytes": int(diagnostics["factor_budget_ceiling_bytes"]),
                "extra_budget_bytes": int(diagnostics["extra_budget_bytes"]),
                "used_extra_bytes": int(diagnostics["used_extra_bytes"]),
                "selected_fp16_blocks": int(diagnostics["selected_fp16_blocks"]),
                "total_blocks": int(diagnostics["total_blocks"]),
                "all_int8_relative_factor_error": float(diagnostics["all_int8_relative_factor_error"]),
                "all_fp16_relative_factor_error": float(diagnostics["all_fp16_relative_factor_error"]),
            })
        records.append(record)
        print(
            f"M20 {unit['unit_id']} task={task_id + 1}/{config['num_tasks']} "
            f"acc={accuracy:.4f} update={update_seconds:.2f}s "
            f"werr={record['relative_classifier_error_vs_exact']:.3e}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if unit["method"] == "exact":
        cache = _exact_cache_path(output_dir, unit["stream_id"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        torch.save(
            {"weights": saved_weights, "class_ids": [r["class_ids"] for r in records]},
            temporary,
        )
        os.replace(temporary, cache)

    result = {
        "schema_version": 1,
        "status": "complete",
        "study_id": config["study_id"],
        "uses_test_set": False,
        **{key: unit[key] for key in ("unit_id", "stream_id", "method", "criterion", "budget")},
        "stream": stream,
        "m6_identity_checks": identity,
        "records": records,
        "validation_aia_percent": sum(r["validation_accuracy_percent"] for r in records) / len(records),
        "final_validation_accuracy_percent": records[-1]["validation_accuracy_percent"],
        "final_total_persistent_bytes": records[-1]["total_persistent_bytes"],
        "analytic_update_seconds": sum(r["update_seconds"] for r in records),
        "maximum_solver_relative_residual": max(r["solver_relative_residual"] for r in records),
        "scoring_benchmark": benchmark_records or None,
        "precision_lock": precision,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda" else platform.processor(),
        },
    }
    path = _unit_path(output_dir, unit["unit_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    del backend
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _describe(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "n": len(values),
    }


def summarize(config: dict, units: list[dict], *, source: dict, config_path: Path | None = None) -> dict:
    plan = unit_plan(config)
    by_id = {unit["unit_id"]: unit for unit in units}
    complete = all(
        item["unit_id"] in by_id and by_id[item["unit_id"]].get("status") == "complete"
        for item in plan
    ) and len(by_id) == len(plan)
    tolerance = float(config["gates"]["maximum_m6_reproduction_accuracy_difference_pp"])
    sweep = config["budget_sweep"]

    per_stream: dict[str, dict] = {}
    for stream in config["streams"]:
        sid = stream["stream_id"]
        exact = by_id.get(f"{sid}__exact")
        if exact is None:
            continue
        rows = {}
        for item in plan:
            if item["stream_id"] != sid or item["unit_id"] not in by_id:
                continue
            unit = by_id[item["unit_id"]]
            last = unit["records"][-1]
            rows[item["unit_id"].split("__", 1)[1]] = {
                "validation_aia_percent": unit["validation_aia_percent"],
                "aia_loss_pp": exact["validation_aia_percent"] - unit["validation_aia_percent"],
                "final_accuracy_loss_pp": exact["final_validation_accuracy_percent"]
                - unit["final_validation_accuracy_percent"],
                "final_relative_classifier_error": last["relative_classifier_error_vs_exact"],
                "final_relative_local_factor_error": last.get("relative_local_factor_error"),
                "final_total_persistent_bytes": unit["final_total_persistent_bytes"],
                "final_used_extra_bytes": last.get("used_extra_bytes"),
                "final_selected_fp16_blocks": last.get("selected_fp16_blocks"),
                "analytic_update_seconds": unit["analytic_update_seconds"],
            }
        per_stream[sid] = rows

    locked_stream = config["streams"][0]["stream_id"]
    width = int(config["width"])
    m6_width = next(item for item in source["width_results"] if int(item["width"]) == width)
    identity_ok = all(
        unit.get("m6_identity_checks") and all(unit["m6_identity_checks"].values())
        for unit in units if unit["stream_id"] == locked_stream
    )
    reproduction = {}
    for method, key in (("exact", "exact"), ("p2b_int8", "p2b_int8")):
        unit = by_id.get(f"{locked_stream}__{method}")
        if unit is None:
            reproduction[method] = None
            continue
        reproduction[method] = {
            "aia_difference_pp": abs(unit["validation_aia_percent"] - m6_width["validation_aia_percent"][key]),
            "final_difference_pp": abs(unit["final_validation_accuracy_percent"] - m6_width["final_validation_accuracy_percent"][key]),
            "bytes_equal": unit["final_total_persistent_bytes"] == m6_width["final_total_persistent_bytes"][key],
        }
    reproduction_ok = all(
        value is not None and value["aia_difference_pp"] <= tolerance
        and value["final_difference_pp"] <= tolerance and value["bytes_equal"]
        for value in reproduction.values()
    )

    adaptive_units = [unit for unit in units if unit["method"].startswith("adaptive")]
    budget_ok = all(
        record["factor_persistent_bytes"] <= record["factor_budget_ceiling_bytes"]
        and record["used_extra_bytes"] <= record["extra_budget_bytes"]
        for unit in adaptive_units for record in unit["records"]
    )

    monotone_ok = True
    for stream in config["streams"]:
        sid = stream["stream_id"]
        for criterion, budgets in (
            ("factor_mse", sweep["factor_mse_budgets"]),
            ("row_weighted_system", sweep["row_weighted_system_budgets"]),
        ):
            sizes = [
                by_id[f"{sid}__{criterion}_b{_budget_key(b)}"]["final_total_persistent_bytes"]
                for b in budgets if f"{sid}__{criterion}_b{_budget_key(b)}" in by_id
            ]
            monotone_ok &= all(a <= b for a, b in zip(sizes, sizes[1:]))

    benchmark = config["scoring_benchmark"]
    bench_id = f"{benchmark['stream_id']}__benchmark_factor_mse_b{_budget_key(benchmark['budget'])}"
    locked_id = f"{benchmark['stream_id']}__factor_mse_b{_budget_key(benchmark['budget'])}"
    fidelity = None
    fidelity_ok = False
    agreement_ok = False
    scoring_summary = None
    if bench_id in by_id and locked_id in by_id:
        bench, locked_unit = by_id[bench_id], by_id[locked_id]
        pairs = list(zip(bench["records"], locked_unit["records"]))
        fidelity = {
            "precision_masks_identical": all(a["precision_mask_sha256"] == b["precision_mask_sha256"] for a, b in pairs),
            "accuracies_identical": all(a["validation_accuracy_percent"] == b["validation_accuracy_percent"] for a, b in pairs),
            "state_bytes_identical": all(a["total_persistent_bytes"] == b["total_persistent_bytes"] for a, b in pairs),
            "weights_bitwise_identical": all(a["weights_sha256"] == b["weights_sha256"] for a, b in pairs),
        }
        fidelity_ok = (
            fidelity["precision_masks_identical"] and fidelity["accuracies_identical"]
            and fidelity["state_bytes_identical"]
        )
        rows = bench.get("scoring_benchmark") or []
        agreement_ok = bool(rows) and all(r["selections_identical"] and r["costs_identical"] for r in rows)
        if rows:
            loop_seconds = [r["per_block_scoring_seconds"] for r in rows]
            batched_seconds = [r["batched_scoring_seconds"] for r in rows]
            scoring_summary = {
                "median_per_block_scoring_seconds": statistics.median(loop_seconds),
                "median_batched_scoring_seconds": statistics.median(batched_seconds),
                "median_speedup": statistics.median(
                    a / b for a, b in zip(loop_seconds, batched_seconds) if b > 0
                ),
                "tasks_with_identical_selection": sum(r["selections_identical"] for r in rows),
                "tasks": len(rows),
                "maximum_relative_benefit_difference": max(r["maximum_relative_benefit_difference"] for r in rows),
                "median_locked_unit_update_seconds": statistics.median(
                    r["update_seconds"] for r in locked_unit["records"]
                ),
            }

    residual_ok = all(
        unit["maximum_solver_relative_residual"] <= float(config["gates"]["maximum_solver_relative_residual"])
        for unit in units
    )
    required = str(config["required_device_name_substring"]).lower()
    device_ok = all(required in unit["environment"]["device_name"].lower() for unit in units)

    curve = {}
    for criterion, budgets in (
        ("factor_mse", sweep["factor_mse_budgets"]),
        ("row_weighted_system", sweep["row_weighted_system_budgets"]),
    ):
        for budget in budgets:
            key = f"{criterion}_b{_budget_key(budget)}"
            rows = [per_stream[sid][key] for sid in per_stream if key in per_stream[sid]]
            if not rows:
                continue
            curve[key] = {
                "criterion": criterion,
                "budget": float(budget),
                "aia_loss_pp": _describe([row["aia_loss_pp"] for row in rows]),
                "final_relative_classifier_error": _describe([row["final_relative_classifier_error"] for row in rows]),
                "final_total_persistent_megabytes": _describe([row["final_total_persistent_bytes"] / 1e6 for row in rows]),
                "analytic_update_seconds": _describe([row["analytic_update_seconds"] for row in rows]),
            }
    for method in ("p2b_int8",):
        rows = [per_stream[sid][method] for sid in per_stream if method in per_stream[sid]]
        if rows:
            curve[method] = {
                "criterion": None, "budget": None,
                "aia_loss_pp": _describe([row["aia_loss_pp"] for row in rows]),
                "final_relative_classifier_error": _describe([row["final_relative_classifier_error"] for row in rows]),
                "final_total_persistent_megabytes": _describe([row["final_total_persistent_bytes"] / 1e6 for row in rows]),
                "analytic_update_seconds": _describe([row["analytic_update_seconds"] for row in rows]),
            }

    threshold = float(config["interpretation"]["sufficient_budget_maximum_aia_loss_pp"])
    sufficient = None
    if complete:
        for budget in sweep["factor_mse_budgets"]:
            key = f"factor_mse_b{_budget_key(budget)}"
            if all(per_stream[sid][key]["aia_loss_pp"] <= threshold for sid in per_stream):
                sufficient = float(budget)
                break

    comparison = {}
    supported = complete
    for budget in config["interpretation"]["criterion_comparison_budgets"]:
        mse_key = f"factor_mse_b{_budget_key(budget)}"
        rws_key = f"row_weighted_system_b{_budget_key(budget)}"
        rows = []
        for sid in per_stream:
            if mse_key in per_stream[sid] and rws_key in per_stream[sid]:
                mse, rws = per_stream[sid][mse_key], per_stream[sid][rws_key]
                rows.append({
                    "stream_id": sid,
                    "classifier_error_difference": rws["final_relative_classifier_error"] - mse["final_relative_classifier_error"],
                    "aia_loss_difference_pp": rws["aia_loss_pp"] - mse["aia_loss_pp"],
                    "state_bytes_difference": rws["final_total_persistent_bytes"] - mse["final_total_persistent_bytes"],
                })
        comparison[_budget_key(budget)] = rows
        if not rows or len(rows) != len(config["streams"]):
            supported = False
            continue
        supported &= all(row["classifier_error_difference"] < 0 for row in rows)
        supported &= statistics.fmean(row["aia_loss_difference_pp"] for row in rows) <= 0

    gates = {
        "all_units_complete": complete,
        "m6_identity_for_locked_stream": identity_ok,
        "exact_and_p2b_reproduce_m6": reproduction_ok,
        "budget_conformance": budget_ok,
        "state_monotone_in_budget": monotone_ok,
        "criterion_backend_fidelity": fidelity_ok,
        "batched_scoring_decision_agreement": agreement_ok,
        "solver_residual_within_tolerance": residual_ok,
        "required_t4_device": device_ok,
    }
    return {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": STATUS_PASS if all(gates.values()) else STATUS_WARNING,
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "config_sha256": _sha256(config_path, source=True) if config_path else None,
        "runner_sha256": _sha256(Path(__file__), source=True),
        "selection_module_sha256": _sha256(ROOT / "methods/analytic_ridge/adaptive_selection.py", source=True),
        "locked_adaptive_storage_sha256": _sha256(ROOT / "methods/analytic_ridge/adaptive_upper.py", source=True),
        "locked_backend_sha256": _sha256(ROOT / "methods/analytic_ridge/backends.py", source=True),
        "m6_reproduction": reproduction,
        "per_stream": per_stream,
        "budget_curve": curve,
        "smallest_sufficient_factor_mse_budget": sufficient,
        "criterion_comparison": comparison,
        "row_weighted_criterion_supported": bool(supported),
        "criterion_backend_fidelity": fidelity,
        "scoring_benchmark": scoring_summary,
        "gates": gates,
        "units": units,
    }


def write_curve_csv(result: dict, path: Path) -> None:
    lines = ["key,criterion,budget,stat,aia_loss_pp,final_relative_classifier_error,final_total_persistent_megabytes,analytic_update_seconds"]
    for key, row in result["budget_curve"].items():
        for stat in ("mean", "sample_std", "minimum", "maximum"):
            lines.append(",".join(str(value) for value in (
                key, row["criterion"], row["budget"], stat, row["aia_loss_pp"][stat],
                row["final_relative_classifier_error"][stat],
                row["final_total_persistent_megabytes"][stat],
                row["analytic_update_seconds"][stat],
            )))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    assert_train_only_cache(feature_cache_dir)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M20 requires a clean source checkout")
    source = m11._load_source(config, Path(args.source_m6_artifact).resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("M20 requested CUDA but CUDA is unavailable")

    plan = unit_plan(config)
    pending = [item for item in plan if not _unit_path(output_dir, item["unit_id"]).is_file()]
    if args.max_new_units is not None:
        pending = pending[: args.max_new_units]
    if pending:
        if _sha256(feature_cache_dir / "train.pt") != config["train_cache_sha256"]:
            raise ValueError("M20 train cache SHA-256 mismatch")
        train, _, metadata = validate_cache(
            feature_cache_dir,
            argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
            load_test=False,
        )
        if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
            raise ValueError("M20 feature-cache checkpoint SHA-256 mismatch")
        for item in pending:
            print(f"M20 START UNIT {item['unit_id']}", flush=True)
            run_unit(config, item, train=train, source=source, output_dir=output_dir, device=device)
            print(f"M20 CHECKPOINT {item['unit_id']}: COMPLETE", flush=True)

    units = [
        json.loads(_unit_path(output_dir, item["unit_id"]).read_text(encoding="utf-8"))
        for item in plan if _unit_path(output_dir, item["unit_id"]).is_file()
    ]
    progress = {"study_id": config["study_id"], "completed_units": len(units), "total_units": len(plan)}
    (output_dir / "m20_progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    if len(units) == len(plan):
        result = summarize(config, units, source=source, config_path=config_path)
        (output_dir / "m20_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        write_curve_csv(result, output_dir / "m20_budget_curve.csv")
        print("M20 STATUS:", result["status"], flush=True)
        print("M20 GATES:", json.dumps(result["gates"], indent=2), flush=True)
        print("M20 SMALLEST SUFFICIENT FACTOR-MSE BUDGET:", result["smallest_sufficient_factor_mse_budget"], flush=True)
        print("M20 ROW-WEIGHTED CRITERION SUPPORTED:", result["row_weighted_criterion_supported"], flush=True)
    else:
        print(f"M20 PROGRESS: {len(units)}/{len(plan)} units", flush=True)
    return progress


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--source-m6-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-units", type=int)
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_units is not None and args.max_new_units <= 0:
        raise ValueError("--max-new-units must be positive")
    return args


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
