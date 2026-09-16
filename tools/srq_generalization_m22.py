"""M22: train-only Random / Static / Adaptive SRQ selection controls.

The study uses three paired RanPAC/CIFAR-100 validation streams at width
20,000.  Every method is rerun inside this study.  No test feature cache is
created or read, and all pass/fail gates are structural rather than accuracy
based.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import ExactGramBackend, SquareRootBackend, persistent_tensor_bytes  # noqa: E402
from methods.analytic_ridge.selection_controls import SelectionControlSquareRootBackend  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools import srq_generalization_m6 as m6  # noqa: E402
from tools import srq_generalization_m11 as m11  # noqa: E402
from tools import srq_generalization_m20 as m20  # noqa: E402
from tools.experiment_runner import validate_cache  # noqa: E402


STATUS_PASS = "PASS_M22_SELECTION_CONTROLS_TRAIN_ONLY"
STATUS_WARNING = "COMPLETE_M22_SELECTION_CONTROLS_WITH_WARNINGS"
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name", "checkpoint_sha256",
    "train_cache_sha256", "uses_test_set", "accuracy_based_selection", "seed",
    "num_classes", "num_tasks", "outer_validation_fraction", "statistics_dtype",
    "solver_dtype", "width", "ridge_lambda", "required_device_name_substring",
    "streams", "source_m6", "ranpac", "p2b", "selection_controls", "gates",
}


def _budget_key(value: float) -> str:
    return f"{float(value):.2f}"


def validate_config(config: dict) -> dict:
    if set(config) != TOP_KEYS:
        raise ValueError("M22 config keys mismatch")
    if config["schema_version"] != 1 or int(config["seed"]) != 2025:
        raise ValueError("M22 requires schema 1 and protocol seed 2025")
    if config["uses_test_set"] is not False or config["accuracy_based_selection"] is not False:
        raise ValueError("M22 must remain train-only and accuracy-independent")
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
        raise ValueError("M22 no longer matches the locked width-20,000 stream")
    if any(len(str(config[key])) != 64 for key in ("checkpoint_sha256", "train_cache_sha256")):
        raise ValueError("M22 cache identities must be SHA-256 values")

    streams = config["streams"]
    expected_stream_fields = {
        "stream_id", "class_order_seed", "split_seed", "projection_seed", "m6_identity_locked"
    }
    if len(streams) != 3 or any(set(stream) != expected_stream_fields for stream in streams):
        raise ValueError("M22 requires three fully declared streams")
    if [stream["stream_id"] for stream in streams] != ["s2025", "s2026", "s2027"]:
        raise ValueError("M22 stream identities or order changed")
    if sum(bool(stream["m6_identity_locked"]) for stream in streams) != 1:
        raise ValueError("M22 requires exactly one M6-locked stream")
    if not streams[0]["m6_identity_locked"] or any(
        int(streams[0][key]) != 2025
        for key in ("class_order_seed", "split_seed", "projection_seed")
    ):
        raise ValueError("M22 first stream must be the locked seed-2025 stream")

    source = config["source_m6"]
    if set(source) != {
        "artifact_filename", "artifact_sha256", "result_member", "result_sha256", "required_status"
    } or source["required_status"] != "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY":
        raise ValueError("invalid M22 M6 source lock")
    ranpac = config["ranpac"]
    if ranpac != {
        "path": "phase2_no_petl_random_relu",
        "feature_dimension": 768,
        "maximum_expand_dimension": 20000,
        "projection_distribution": "standard_normal",
        "activation": "relu",
        "encode_batch_size": 256,
        "evaluation_batch_size": 256,
    }:
        raise ValueError("M22 RanPAC semantics changed")
    p2b = config["p2b"]
    if p2b != {
        "block_size": 256,
        "group_size": 64,
        "update_panel_size": 128,
        "update_trailing_chunk_size": None,
        "first_update_backend": "gram_cholesky",
        "quantization_backend": "streaming",
        "quantization_batch_blocks": 64,
    }:
        raise ValueError("M22 P2B settings changed")

    controls = config["selection_controls"]
    if set(controls) != {
        "comparison_budgets", "adaptive_connector_budgets", "random_allocation_seeds",
        "static_initial_policy", "static_later_policy", "random_policy",
        "precision_mask_dtype", "selection_signals_forbidden",
    }:
        raise ValueError("M22 selection-control fields mismatch")
    if [float(x) for x in controls["comparison_budgets"]] != [0.01, 0.02, 0.05]:
        raise ValueError("M22 comparison budgets changed")
    if [float(x) for x in controls["adaptive_connector_budgets"]] != [0.10, 0.25]:
        raise ValueError("M22 connector budgets changed")
    if [int(x) for x in controls["random_allocation_seeds"]] != [202501, 202502]:
        raise ValueError("M22 random allocation seeds changed")
    if (
        controls["static_initial_policy"] != "factor_mse_at_task_1"
        or controls["static_later_policy"] != "reuse_task_1_precision_mask"
        or controls["random_policy"] != "uniform_random_block_order_greedy_byte_fill"
        or controls["precision_mask_dtype"] != "uint8"
        or controls["selection_signals_forbidden"]
        != ["labels", "logits", "predictions", "accuracy", "test_data"]
    ):
        raise ValueError("M22 policy contract changed")
    gates = config["gates"]
    if set(gates) != {
        "maximum_solver_relative_residual", "maximum_m6_reproduction_accuracy_difference_pp",
        "require_m6_identity_for_locked_stream", "require_budget_conformance",
        "require_equal_budget_ceiling", "require_static_mask_after_task_1",
        "require_distinct_random_draws", "require_t4_device", "accuracy_gate",
    } or gates["accuracy_gate"] is not None:
        raise ValueError("M22 structural gates changed or an accuracy gate was added")
    return config


def read_config(path: str | Path) -> dict:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def unit_plan(config: dict) -> list[dict]:
    controls = config["selection_controls"]
    comparison = [float(value) for value in controls["comparison_budgets"]]
    connectors = [float(value) for value in controls["adaptive_connector_budgets"]]
    plan = []
    for stream in config["streams"]:
        sid = stream["stream_id"]
        plan.extend([
            {"unit_id": f"{sid}__exact", "stream_id": sid, "method": "exact", "budget": None, "allocation_seed": None},
            {"unit_id": f"{sid}__p2b_int8", "stream_id": sid, "method": "p2b_int8", "budget": None, "allocation_seed": None},
        ])
        for budget in comparison + connectors:
            plan.append({
                "unit_id": f"{sid}__adaptive_b{_budget_key(budget)}", "stream_id": sid,
                "method": "adaptive", "budget": budget, "allocation_seed": None,
            })
        for budget in comparison:
            plan.append({
                "unit_id": f"{sid}__static_b{_budget_key(budget)}", "stream_id": sid,
                "method": "static", "budget": budget, "allocation_seed": None,
            })
            for seed in controls["random_allocation_seeds"]:
                plan.append({
                    "unit_id": f"{sid}__random_r{int(seed)}_b{_budget_key(budget)}",
                    "stream_id": sid, "method": "random", "budget": budget,
                    "allocation_seed": int(seed),
                })
    return plan


def _unit_path(output_dir: Path, unit_id: str) -> Path:
    return output_dir / "units" / f"{unit_id}.json"


def _exact_cache_path(output_dir: Path, stream_id: str) -> Path:
    return output_dir / "cache" / f"{stream_id}_exact_weights.pt"


def build_backend(config: dict, unit: dict, device: torch.device):
    common = m20._common_backend(config, device)
    if unit["method"] == "exact":
        return ExactGramBackend(**common)
    if unit["method"] == "p2b_int8":
        return SquareRootBackend(storage_mode="int8", **common, **m20._p2b_kwargs(config))
    if unit["method"] == "adaptive":
        return SquareRootBackend(
            storage_mode="adaptive_int8_fp16",
            adaptive_budget_fraction=float(unit["budget"]),
            **common,
            **m20._p2b_kwargs(config),
        )
    if unit["method"] in {"random", "static"}:
        return SelectionControlSquareRootBackend(
            selection_policy=unit["method"],
            adaptive_budget_fraction=float(unit["budget"]),
            allocation_seed=int(unit["allocation_seed"] or config["seed"]),
            **common,
            **m20._p2b_kwargs(config),
        )
    raise ValueError(f"unknown M22 method {unit['method']}")


def _evaluate_against_exact(
    *, encoder, backend, exact_weights: torch.Tensor | None,
    exact_class_ids: list[int] | None, features: torch.Tensor, labels: torch.Tensor,
    indices: torch.Tensor, batch_size: int,
) -> dict[str, float]:
    correct = 0
    agreement = 0
    difference2 = 0.0
    reference2 = 0.0
    method_ids = torch.tensor(backend.class_ids, dtype=torch.long)
    if exact_class_ids is not None and exact_class_ids != backend.class_ids:
        raise AssertionError("M22 class columns differ from Exact")
    total = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        codes = encoder(features[batch_indices])
        logits = backend.predict_logits(codes)
        columns = logits.argmax(1).cpu()
        prediction = method_ids[columns]
        targets = labels[batch_indices].cpu()
        correct += int((prediction == targets).sum().item())
        if exact_weights is not None:
            reference = codes.to(exact_weights.dtype) @ exact_weights
            exact_columns = reference.argmax(1).cpu()
            agreement += int((columns == exact_columns).sum().item())
            difference2 += float((logits.to(torch.float64) - reference.to(torch.float64)).square().sum().item())
            reference2 += float(reference.to(torch.float64).square().sum().item())
        total += len(batch_indices)
    if exact_weights is None:
        return {
            "validation_accuracy_percent": 100.0 * correct / total,
            "relative_logit_frobenius_error_vs_exact": 0.0,
            "prediction_agreement_with_exact": 1.0,
        }
    return {
        "validation_accuracy_percent": 100.0 * correct / total,
        "relative_logit_frobenius_error_vs_exact": (difference2 / max(reference2, 1.0)) ** 0.5,
        "prediction_agreement_with_exact": agreement / total,
    }


def run_unit(
    config: dict, unit: dict, *, train: dict, source: dict,
    output_dir: Path, device: torch.device,
) -> dict:
    precision = m20._lock_precision()
    stream = m20._stream_by_id(config, unit["stream_id"])
    prepared = m20.prepare_stream(config, stream, train["labels"], device)
    identity = m20.stream_identity(config, prepared, source) if stream["m6_identity_locked"] else None
    projection = prepared["projection"]
    encoder = lambda values: torch.relu(values.to(device=device, dtype=torch.float32) @ projection)

    exact_cache = None
    if unit["method"] != "exact":
        path = _exact_cache_path(output_dir, unit["stream_id"])
        if not path.is_file():
            raise RuntimeError(f"M22 unit {unit['unit_id']} requires its Exact unit first")
        exact_cache = torch.load(path, weights_only=True)

    backend = build_backend(config, unit, device)
    records = []
    saved_weights, saved_classes = [], []
    for task_id, train_indices in enumerate(prepared["training_parts"]):
        codes = m5._encode_indices(
            encoder, train["features"], train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        m20._sync(device)
        started = time.perf_counter()
        backend.update(codes, train["labels"][train_indices])
        m20._sync(device)
        update_seconds = time.perf_counter() - started
        del codes

        seen = torch.cat(prepared["validation_parts"][: task_id + 1])
        metrics = _evaluate_against_exact(
            encoder=encoder,
            backend=backend,
            exact_weights=None if exact_cache is None else exact_cache["weights"][task_id].to(device),
            exact_class_ids=None if exact_cache is None else exact_cache["class_ids"][task_id],
            features=train["features"], labels=train["labels"], indices=seen,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )
        tensors = {"projection": projection, **backend.persistent_tensors()}
        backend_tensors = backend.persistent_tensors()
        factor_bytes = persistent_tensor_bytes({
            name: tensor for name, tensor in backend_tensors.items() if name.startswith("factor.")
        })
        diagnostics = backend.diagnostics
        record = {
            "task": task_id + 1,
            **metrics,
            "total_persistent_bytes": persistent_tensor_bytes(tensors),
            "factor_persistent_bytes": factor_bytes,
            "solver_relative_residual": float(diagnostics.get("solver_relative_residual", 0.0)),
            "update_seconds": update_seconds,
            "class_ids": [int(value) for value in backend.class_ids],
            "weights_sha256": m20._tensor_sha256(backend.weights),
        }
        if unit["method"] == "exact":
            record["relative_classifier_error_vs_exact"] = 0.0
            saved_weights.append(backend.weights.detach().cpu().clone())
            saved_classes.append(record["class_ids"])
        else:
            reference = exact_cache["weights"][task_id].to(device)
            record["relative_classifier_error_vs_exact"] = float(
                torch.linalg.vector_norm(backend.weights - reference).item()
                / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)
            )
            record["relative_local_factor_error"] = float(
                diagnostics.get("relative_local_factor_error", float("nan"))
            )
        if unit["method"] in {"adaptive", "random", "static"}:
            record.update({
                "precision_mask_sha256": m20._mask_sha256(backend),
                "factor_budget_ceiling_bytes": int(diagnostics["factor_budget_ceiling_bytes"]),
                "extra_budget_bytes": int(diagnostics["extra_budget_bytes"]),
                "used_extra_bytes": int(diagnostics["used_extra_bytes"]),
                "selected_fp16_blocks": int(diagnostics["selected_fp16_blocks"]),
                "total_blocks": int(diagnostics["total_blocks"]),
                "selection_policy": "adaptive_factor_mse" if unit["method"] == "adaptive" else diagnostics["selection_policy"],
                "mask_source": "current_factor_mse" if unit["method"] == "adaptive" else diagnostics["mask_source"],
            })
        records.append(record)
        print(
            f"M22 {unit['unit_id']} task={task_id + 1}/{config['num_tasks']} "
            f"acc={metrics['validation_accuracy_percent']:.4f} "
            f"logit={metrics['relative_logit_frobenius_error_vs_exact']:.3e} "
            f"update={update_seconds:.2f}s",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if unit["method"] == "exact":
        path = _exact_cache_path(output_dir, unit["stream_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"weights": saved_weights, "class_ids": saved_classes}, temporary)
        os.replace(temporary, path)

    result = {
        "schema_version": 1,
        "status": "complete",
        "study_id": config["study_id"],
        "uses_test_set": False,
        **{key: unit[key] for key in ("unit_id", "stream_id", "method", "budget", "allocation_seed")},
        "stream": stream,
        "m6_identity_checks": identity,
        "records": records,
        "validation_aia_percent": statistics.fmean(r["validation_accuracy_percent"] for r in records),
        "final_validation_accuracy_percent": records[-1]["validation_accuracy_percent"],
        "final_total_persistent_bytes": records[-1]["total_persistent_bytes"],
        "analytic_update_seconds": sum(r["update_seconds"] for r in records),
        "maximum_solver_relative_residual": max(r["solver_relative_residual"] for r in records),
        "precision_lock": precision,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
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


def _describe(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values), "maximum": max(values), "n": len(values),
    }


def summarize(
    config: dict, units: list[dict], *, source: dict, config_path: Path | None = None
) -> dict:
    plan = unit_plan(config)
    by_id = {unit["unit_id"]: unit for unit in units}
    complete = len(by_id) == len(plan) and all(
        item["unit_id"] in by_id and by_id[item["unit_id"]].get("status") == "complete"
        for item in plan
    )
    locked = config["streams"][0]["stream_id"]
    identity_ok = all(
        unit.get("m6_identity_checks") and all(unit["m6_identity_checks"].values())
        for unit in units if unit["stream_id"] == locked
    )
    m6_width = next(row for row in source["width_results"] if int(row["width"]) == int(config["width"]))
    tolerance = float(config["gates"]["maximum_m6_reproduction_accuracy_difference_pp"])
    reproduction = {}
    for method in ("exact", "p2b_int8"):
        unit = by_id.get(f"{locked}__{method}")
        reference_key = "exact" if method == "exact" else "p2b_int8"
        if unit is None:
            reproduction[method] = None
        else:
            reproduction[method] = {
                "aia_difference_pp": abs(unit["validation_aia_percent"] - m6_width["validation_aia_percent"][reference_key]),
                "final_difference_pp": abs(unit["final_validation_accuracy_percent"] - m6_width["final_validation_accuracy_percent"][reference_key]),
                "bytes_equal": unit["final_total_persistent_bytes"] == m6_width["final_total_persistent_bytes"][reference_key],
            }
    reproduction_ok = all(
        row is not None and row["aia_difference_pp"] <= tolerance
        and row["final_difference_pp"] <= tolerance and row["bytes_equal"]
        for row in reproduction.values()
    )

    mixed = [unit for unit in units if unit["method"] in {"adaptive", "static", "random"}]
    budget_ok = all(
        record["factor_persistent_bytes"] <= record["factor_budget_ceiling_bytes"]
        and record["used_extra_bytes"] <= record["extra_budget_bytes"]
        for unit in mixed for record in unit["records"]
    )
    ceiling_ok = True
    actual_byte_gaps = []
    for stream in config["streams"]:
        sid = stream["stream_id"]
        for budget in config["selection_controls"]["comparison_budgets"]:
            candidates = [unit for unit in mixed if unit["stream_id"] == sid and unit["budget"] == budget]
            for task in range(int(config["num_tasks"])):
                ceilings = {unit["records"][task]["extra_budget_bytes"] for unit in candidates}
                ceilings |= {unit["records"][task]["factor_budget_ceiling_bytes"] for unit in candidates}
                # The two quantities live on different scales; compare each separately.
                budget_ceilings = {unit["records"][task]["extra_budget_bytes"] for unit in candidates}
                factor_ceilings = {unit["records"][task]["factor_budget_ceiling_bytes"] for unit in candidates}
                ceiling_ok &= len(budget_ceilings) == 1 and len(factor_ceilings) == 1
                used = [unit["records"][task]["used_extra_bytes"] for unit in candidates]
                actual_byte_gaps.append(max(used) - min(used))
    static_ok = all(
        len({record["precision_mask_sha256"] for record in unit["records"]}) == 1
        for unit in mixed if unit["method"] == "static"
    )
    random_distinct = True
    seeds = config["selection_controls"]["random_allocation_seeds"]
    for stream in config["streams"]:
        sid = stream["stream_id"]
        for budget in config["selection_controls"]["comparison_budgets"]:
            a = by_id.get(f"{sid}__random_r{seeds[0]}_b{_budget_key(budget)}")
            b = by_id.get(f"{sid}__random_r{seeds[1]}_b{_budget_key(budget)}")
            random_distinct &= bool(a and b) and any(
                x["precision_mask_sha256"] != y["precision_mask_sha256"]
                for x, y in zip(a["records"], b["records"])
            )
    residual_ok = all(
        unit["maximum_solver_relative_residual"]
        <= float(config["gates"]["maximum_solver_relative_residual"])
        for unit in units
    )
    required_device = str(config["required_device_name_substring"]).lower()
    device_ok = all(required_device in unit["environment"]["device_name"].lower() for unit in units)

    per_stream = {}
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
            final = unit["records"][-1]
            rows[item["unit_id"].split("__", 1)[1]] = {
                "method": unit["method"], "budget": unit["budget"],
                "allocation_seed": unit["allocation_seed"],
                "validation_aia_percent": unit["validation_aia_percent"],
                "aia_loss_pp": exact["validation_aia_percent"] - unit["validation_aia_percent"],
                "final_accuracy_percent": unit["final_validation_accuracy_percent"],
                "final_relative_logit_error": final["relative_logit_frobenius_error_vs_exact"],
                "final_prediction_agreement": final["prediction_agreement_with_exact"],
                "final_relative_factor_error": final.get("relative_local_factor_error"),
                "final_total_persistent_bytes": unit["final_total_persistent_bytes"],
                "final_used_extra_bytes": final.get("used_extra_bytes"),
                "analytic_update_seconds": unit["analytic_update_seconds"],
            }
        per_stream[sid] = rows

    comparisons = {}
    for budget in config["selection_controls"]["comparison_budgets"]:
        budget_key = _budget_key(budget)
        rows = []
        for sid in per_stream:
            adaptive = per_stream[sid][f"adaptive_b{budget_key}"]
            static = per_stream[sid][f"static_b{budget_key}"]
            random_rows = [
                per_stream[sid][f"random_r{seed}_b{budget_key}"]
                for seed in config["selection_controls"]["random_allocation_seeds"]
            ]
            rows.append({
                "stream_id": sid,
                "adaptive_minus_static_aia_pp": adaptive["validation_aia_percent"] - static["validation_aia_percent"],
                "adaptive_minus_random_mean_aia_pp": adaptive["validation_aia_percent"]
                - statistics.fmean(row["validation_aia_percent"] for row in random_rows),
                "adaptive_minus_static_logit_error": adaptive["final_relative_logit_error"] - static["final_relative_logit_error"],
                "adaptive_minus_random_mean_logit_error": adaptive["final_relative_logit_error"]
                - statistics.fmean(row["final_relative_logit_error"] for row in random_rows),
                "adaptive_used_extra_bytes": adaptive["final_used_extra_bytes"],
                "static_used_extra_bytes": static["final_used_extra_bytes"],
                "random_used_extra_bytes": [row["final_used_extra_bytes"] for row in random_rows],
            })
        comparisons[budget_key] = rows

    aggregate = {}
    for method in ("adaptive", "static", "random"):
        for budget in config["selection_controls"]["comparison_budgets"]:
            rows = [
                row for sid in per_stream for row in per_stream[sid].values()
                if row["method"] == method and row["budget"] == budget
            ]
            key = f"{method}_b{_budget_key(budget)}"
            aggregate[key] = {
                "aia_loss_pp": _describe([row["aia_loss_pp"] for row in rows]),
                "final_relative_logit_error": _describe([row["final_relative_logit_error"] for row in rows]),
                "final_relative_factor_error": _describe([row["final_relative_factor_error"] for row in rows]),
                "final_total_persistent_megabytes": _describe([row["final_total_persistent_bytes"] / 1e6 for row in rows]),
                "analytic_update_seconds": _describe([row["analytic_update_seconds"] for row in rows]),
            }

    gates = {
        "all_units_complete": complete,
        "m6_identity_for_locked_stream": identity_ok,
        "exact_and_p2b_reproduce_m6": reproduction_ok,
        "budget_conformance": budget_ok,
        "equal_budget_ceiling": ceiling_ok,
        "static_mask_after_task_1": static_ok,
        "distinct_random_draws": random_distinct,
        "solver_residual_within_tolerance": residual_ok,
        "required_t4_device": device_ok,
    }
    return {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": STATUS_PASS if all(gates.values()) else STATUS_WARNING,
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "config_sha256": m20._sha256(config_path, source=True) if config_path else None,
        "runner_sha256": m20._sha256(Path(__file__), source=True),
        "control_backend_sha256": m20._sha256(ROOT / "methods/analytic_ridge/selection_controls.py", source=True),
        "locked_adaptive_backend_sha256": m20._sha256(ROOT / "methods/analytic_ridge/backends.py", source=True),
        "m6_reproduction": reproduction,
        "maximum_actual_used_byte_gap_under_equal_ceiling": max(actual_byte_gaps, default=0),
        "per_stream": per_stream,
        "policy_comparisons": comparisons,
        "aggregate": aggregate,
        "gates": gates,
        "units": units,
    }


def write_csv(result: dict, path: Path) -> None:
    lines = [
        "stream_id,arm,method,budget,allocation_seed,aia,aia_loss,final_accuracy,final_logit_error,final_prediction_agreement,final_factor_error,state_bytes,used_extra_bytes,update_seconds"
    ]
    for sid, arms in result["per_stream"].items():
        for arm, row in arms.items():
            lines.append(",".join(str(value) for value in (
                sid, arm, row["method"], row["budget"], row["allocation_seed"],
                row["validation_aia_percent"], row["aia_loss_pp"], row["final_accuracy_percent"],
                row["final_relative_logit_error"], row["final_prediction_agreement"],
                row["final_relative_factor_error"], row["final_total_persistent_bytes"],
                row["final_used_extra_bytes"], row["analytic_update_seconds"],
            )))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    m20.assert_train_only_cache(feature_cache_dir)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M22 requires a clean source checkout")
    source = m11._load_source(config, Path(args.source_m6_artifact).resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("M22 requested CUDA but CUDA is unavailable")

    plan = unit_plan(config)
    pending = [item for item in plan if not _unit_path(output_dir, item["unit_id"]).is_file()]
    if args.max_new_units is not None:
        pending = pending[: args.max_new_units]
    if pending:
        if m20._sha256(feature_cache_dir / "train.pt") != config["train_cache_sha256"]:
            raise ValueError("M22 train cache SHA-256 mismatch")
        train, _, metadata = validate_cache(
            feature_cache_dir,
            argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
            load_test=False,
        )
        if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
            raise ValueError("M22 feature-cache checkpoint mismatch")
        for item in pending:
            print(f"M22 START UNIT {item['unit_id']}", flush=True)
            run_unit(config, item, train=train, source=source, output_dir=output_dir, device=device)
            print(f"M22 CHECKPOINT {item['unit_id']}: COMPLETE", flush=True)

    units = [
        json.loads(_unit_path(output_dir, item["unit_id"]).read_text(encoding="utf-8"))
        for item in plan if _unit_path(output_dir, item["unit_id"]).is_file()
    ]
    progress = {"study_id": config["study_id"], "completed_units": len(units), "total_units": len(plan)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m22_progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    if len(units) == len(plan):
        result = summarize(config, units, source=source, config_path=config_path)
        (output_dir / "m22_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        write_csv(result, output_dir / "m22_selection_controls.csv")
        print("M22 STATUS:", result["status"], flush=True)
        print("M22 GATES:", json.dumps(result["gates"], indent=2), flush=True)
    else:
        print(f"M22 PROGRESS: {len(units)}/{len(plan)} units", flush=True)
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
