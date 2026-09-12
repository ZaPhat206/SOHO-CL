"""M14: train-only multi-seed equal-byte LoRanPAC confirmation."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import persistent_tensor_bytes  # noqa: E402
from methods.frontends.loranpac import (  # noqa: E402
    LoRanPACTSVBackend,
    maximum_loranpac_rank_for_backend_budget,
)
from tools import srq_generalization_m4 as m4  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools import srq_generalization_m6 as m6  # noqa: E402
from tools import srq_generalization_m12 as m12  # noqa: E402
from tools import srq_generalization_m13 as m13  # noqa: E402
from tools.experiment_runner import (  # noqa: E402
    split,
    train_validation_indices,
    validate_cache,
)


METHODS = (
    "exact",
    "p2b_int8",
    "adaptive_int8_fp16",
    "loranpac_p2b_budget",
    "loranpac_adaptive_budget",
)
LORANPAC_BUDGET = {
    "loranpac_p2b_budget": "p2b_int8",
    "loranpac_adaptive_budget": "adaptive_int8_fp16",
}
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "uses_test_set", "accuracy_based_selection",
    "num_classes", "num_tasks", "outer_validation_fraction", "replicates",
    "widths", "methods", "selected_ridge_by_width", "source_m6",
    "source_m11", "source_m13n", "ranpac", "p2b", "adaptive", "loranpac",
    "evaluation", "integrity_gates",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M14 config keys/schema mismatch")
    replicates = config["replicates"]
    expected = [
        {"class_order_seed": seed, "projection_seed": seed, "split_seed": seed}
        for seed in range(4101, 4107)
    ]
    if not (
        config["uses_test_set"] is False
        and config["accuracy_based_selection"] is False
        and config["num_classes"] == 100
        and config["num_tasks"] == 10
        and config["widths"] == [10000, 20000]
        and config["methods"] == list(METHODS)
        and replicates == expected
        and config["selected_ridge_by_width"]
        == {"10000": 1_000_000.0, "20000": 1_000_000.0}
        and 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("M14 locked train-only multi-seed design changed")
    for key, required in (
        ("source_m6", "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY"),
        ("source_m11", "PASS_M11_ADAPTIVE_PRECISION_TRAIN_ONLY"),
        ("source_m13n", "PASS_M13N_NUMERICAL_AUDIT"),
    ):
        source = config[key]
        if set(source) != {
            "artifact_filename", "artifact_sha256", "result_member",
            "result_sha256", "required_status",
        } or source["required_status"] != required:
            raise ValueError(f"invalid M14 {key} lock")
        if len(source["artifact_sha256"]) != 64 or len(source["result_sha256"]) != 64:
            raise ValueError(f"invalid M14 {key} hashes")
    ranpac = config["ranpac"]
    if not (
        ranpac.get("upstream_commit") == "cf4b301d18b0c27db030f4371b72b768005ae58a"
        and ranpac.get("path") == "phase2_no_petl_random_relu"
        and ranpac.get("feature_dimension") == 768
        and ranpac.get("maximum_expand_dimension") == 20000
        and ranpac.get("projection_distribution") == "standard_normal"
        and ranpac.get("activation") == "relu"
        and min(ranpac.get("encode_batch_size", 0), ranpac.get("evaluation_batch_size", 0)) > 0
    ):
        raise ValueError("M14 RanPAC frontend identity changed")
    p2b = config["p2b"]
    if not (
        p2b.get("block_size") == 256
        and p2b.get("group_size") == 64
        and p2b.get("update_panel_size") == 128
        and p2b.get("update_trailing_chunk_size") is None
        and p2b.get("first_update_backend") == "gram_cholesky"
        and p2b.get("quantization_backend") == "streaming"
        and p2b.get("quantization_batch_blocks") == 64
    ):
        raise ValueError("M14 P2B identity changed")
    adaptive = config["adaptive"]
    if not (
        adaptive.get("budget_fraction_between_int8_and_fp16") == 0.25
        and adaptive.get("selection_rule")
        == "largest_factor_mse_reduction_per_added_byte"
        and adaptive.get("selection_signal")
        == "current_factor_values_only_no_labels_or_accuracy"
        and adaptive.get("precision_mask_dtype") == "uint8"
        and adaptive.get("tie_break") == "ascending_upper_block_index"
    ):
        raise ValueError("M14 adaptive policy changed")
    loranpac = config["loranpac"]
    if not (
        loranpac.get("upstream_commit") == "32782f9d260e5d722de5702675ed66cca8234883"
        and loranpac.get("truncate_percent") == 25.0
        and loranpac.get("rank_rule") == "official_code_python_round"
        and loranpac.get("paper_rank_rule") == "ceil"
        and loranpac.get("official_ridge_lambda") == 0.0
        and loranpac.get("primary_ridge")
        == "reuse_locked_m6_ridge_without_accuracy_selection"
        and loranpac.get("rank_cap_policy")
        == "largest_integer_rank_not_exceeding_paired_backend_final_state_bytes"
        and str(loranpac.get("rank_cap_selection_signal", "")).endswith(
            "no_labels_or_accuracy"
        )
    ):
        raise ValueError("M14 LoRanPAC policy changed")
    evaluation = config["evaluation"]
    gates = config["integrity_gates"]
    if not (
        evaluation.get("training_split")
        == "outer_train_portion_of_official_train_split"
        and evaluation.get("validation_split")
        == "outer_validation_portion_of_official_train_split"
        and evaluation.get("schedule")
        == "class_incremental_seen_classes_after_each_task"
        and evaluation.get("primary_metrics")
        == ["validation_aia", "final_validation_accuracy"]
        and evaluation.get("report_mean_and_sample_standard_deviation") is True
        and evaluation.get("report_paired_differences") is True
        and evaluation.get("confidence_intervals_in_main_table") is False
    ):
        raise ValueError("M14 evaluation contract changed")
    if evaluation.get("accuracy_gate", "missing") is not None or gates.get(
        "accuracy_gate", "missing"
    ) is not None:
        raise ValueError("M14 completion cannot depend on accuracy")
    required_true = {
        "require_clean_git", "require_source_artifact_identity",
        "require_all_units_complete", "require_rank_caps_derived_from_bytes",
        "require_total_state_not_above_target",
        "maximum_budget_underfill_one_rank_bytes",
    }
    if not all(gates.get(name) is True for name in required_true):
        raise ValueError("M14 integrity gates changed")
    expected_thresholds = {
        "maximum_normalized_orthogonality_residual": 0.0015,
        "maximum_task1_spectral_orthogonality_residual": 0.0035,
        "maximum_task1_solver_relative_residual": 0.001,
        "maximum_later_task_solver_relative_residual": 0.00002,
        "maximum_srq_solver_relative_residual": 0.00002,
    }
    if any(float(gates.get(name, math.nan)) != value for name, value in expected_thresholds.items()):
        raise ValueError("M14 preregistered numerical thresholds changed")
    return config


def _load_sources(config: dict, args) -> tuple[dict, dict]:
    sources = m13._load_sources(
        config, Path(args.source_m6_artifact), Path(args.source_m11_artifact)
    )
    m13n = m13._load_source(config["source_m13n"], Path(args.source_m13n_artifact))
    if (
        m13n.get("uses_test_set") is not False
        or m13n.get("computes_predictive_metrics") is not False
        or m13n.get("scope", {}).get("m13_status_remains")
        != "FAIL_M13_LORANPAC_TRAIN_ONLY"
    ):
        raise ValueError("M14 M13-N source boundary mismatch")
    return sources, m13n


def _parts(config: dict, train: dict, replicate: dict):
    order = random.Random(replicate["class_order_seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], order, config["num_tasks"])
    training, validation = train_validation_indices(
        train["labels"], task_indices, replicate["split_seed"],
        config["outer_validation_fraction"],
    )
    return order, training, validation


def _state_contracts(sources: dict, width: int, method: str) -> list[dict]:
    m6_widths, m11_widths = m13._source_widths(sources)
    if method in {"exact", "p2b_int8"}:
        return [
            {
                "kind": "fixed_source_bytes",
                "minimum_bytes": int(record["state"][method]["total_persistent_bytes"]),
                "maximum_bytes": int(record["state"][method]["total_persistent_bytes"]),
            }
            for record in m6_widths[width]["records"]
        ]
    if method != "adaptive_int8_fp16":
        raise ValueError(f"no additive state contract for {method}")
    contracts = []
    for m6_record, m11_record in zip(
        m6_widths[width]["records"], m11_widths[width]["records"]
    ):
        source_total = int(m11_record["total_persistent_bytes"])
        source_factor = int(m11_record["factor_persistent_bytes"])
        common = source_total - source_factor
        total_blocks = int(m11_record["total_blocks"])
        minimum = int(m6_record["state"]["p2b_int8"]["total_persistent_bytes"]) + total_blocks
        maximum = common + int(m11_record["factor_budget_ceiling_bytes"])
        contracts.append({
            "kind": "adaptive_locked_byte_interval",
            "minimum_bytes": minimum,
            "maximum_bytes": maximum,
        })
    return contracts


def _unit_identity(config: dict, replicate: dict, width: int, method: str) -> dict:
    identity = {
        "study_id": config["study_id"],
        "class_order_seed": replicate["class_order_seed"],
        "projection_seed": replicate["projection_seed"],
        "split_seed": replicate["split_seed"],
        "width": width,
        "method": method,
        "ridge_lambda": config["selected_ridge_by_width"][str(width)],
    }
    if method in LORANPAC_BUDGET:
        identity["budget_target"] = LORANPAC_BUDGET[method]
    return identity


def _rank_contract_for_target(
    config: dict, *, width: int, budget: str, target_total_bytes: int
) -> dict:
    projection_bytes = 4 * int(config["ranpac"]["feature_dimension"]) * width
    target_backend = int(target_total_bytes) - projection_bytes
    rank = maximum_loranpac_rank_for_backend_budget(
        dimension=width,
        num_classes=int(config["num_classes"]),
        target_bytes=target_backend,
    )
    if rank <= 0:
        raise ValueError(f"M14 {budget}/{width} cannot fit rank one")
    base = 8 * width * int(config["num_classes"]) + 4 * int(config["num_classes"])
    per_rank = 4 * (width + 1)
    expected_backend = base + rank * per_rank
    expected_total = projection_bytes + expected_backend
    return {
        "budget_target": budget,
        "target_total_persistent_bytes": int(target_total_bytes),
        "projection_bytes": projection_bytes,
        "target_backend_bytes": target_backend,
        "derived_max_rank": rank,
        "bytes_per_additional_rank": per_rank,
        "expected_final_backend_bytes": expected_backend,
        "expected_final_total_persistent_bytes": expected_total,
        "final_budget_underfill_bytes": int(target_total_bytes) - expected_total,
        "target_source": "paired_backend_measured_final_state",
    }


def _run_additive_unit(
    *, config: dict, sources: dict, train: dict, replicate: dict, width: int,
    method: str, projection: torch.Tensor, order: list[int],
    training_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    backend = m12._make_backend(config, width, method, device)
    encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ projection
    )
    contracts = _state_contracts(sources, width, method)
    records, encoding_seconds, update_seconds = [], 0.0, 0.0
    for task, train_indices in enumerate(training_parts, start=1):
        m5._sync(device)
        started = time.perf_counter()
        codes = m5._encode_indices(
            encoder, train["features"], train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        m5._sync(device)
        encoding_seconds += time.perf_counter() - started
        m5._sync(device)
        started = time.perf_counter()
        backend.update(codes, train["labels"][train_indices])
        m5._sync(device)
        update_seconds += time.perf_counter() - started
        seen_validation = torch.cat(validation_parts[:task])
        accuracy = m5._evaluate(
            encoder=encoder, backends={method: backend}, features=train["features"],
            labels=train["labels"], indices=seen_validation,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )[method]
        total_bytes = persistent_tensor_bytes(
            {"projection": projection, **backend.persistent_tensors()}
        )
        contract = contracts[task - 1]
        if not contract["minimum_bytes"] <= total_bytes <= contract["maximum_bytes"]:
            raise AssertionError(
                f"M14 state mismatch {method}/{width}/task {task}: {total_bytes} "
                f"not in [{contract['minimum_bytes']},{contract['maximum_bytes']}]"
            )
        records.append({
            "task": task,
            "validation_accuracy_percent": float(accuracy),
            "total_persistent_bytes": int(total_bytes),
            "solver_relative_residual": float(
                backend.diagnostics["solver_relative_residual"]
            ),
            "state_contract": contract,
        })
        print(
            f"M14 c={replicate['class_order_seed']} w={width} {method} "
            f"task={task}/10 val={accuracy:.4f}", flush=True,
        )
        del codes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    accuracies = [record["validation_accuracy_percent"] for record in records]
    return {
        "identity": _unit_identity(config, replicate, width, method),
        "uses_test_set": False,
        "class_order": order,
        "training_indices_sha256": m6._sequence_sha256(training_parts),
        "validation_indices_sha256": m6._sequence_sha256(validation_parts),
        "records": records,
        "validation_aia_percent": sum(accuracies) / len(accuracies),
        "final_validation_accuracy_percent": accuracies[-1],
        "final_total_persistent_bytes": records[-1]["total_persistent_bytes"],
        "maximum_solver_relative_residual": max(
            record["solver_relative_residual"] for record in records
        ),
        "representation_encoding_seconds": encoding_seconds,
        "analytic_update_seconds": update_seconds,
    }


def _orthogonality_metrics(basis: torch.Tensor, *, spectral: bool) -> dict:
    rank = int(basis.shape[1])
    identity = torch.eye(rank, device=basis.device, dtype=basis.dtype)
    error = basis.T @ basis - identity
    frobenius = float(torch.linalg.vector_norm(error).item())
    result = {
        "raw_frobenius": frobenius,
        "frobenius_over_sqrt_rank": frobenius / math.sqrt(rank),
        "maximum_absolute_entry": float(error.abs().max().item()),
        "spectral_norm": None,
    }
    if spectral:
        symmetric = (error + error.T) * 0.5
        result["spectral_norm"] = float(torch.linalg.eigvalsh(symmetric).abs().max().item())
    return result


def _evaluate_loranpac_heads(
    *, backend: LoRanPACTSVBackend, official_weights: torch.Tensor, encoder,
    features: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    correct = {"matched_ridge_primary": 0, "official_ridge0_sensitivity": 0}
    total = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        codes = encoder(features[batch_indices])
        targets = labels[batch_indices].cpu()
        primary = backend.predict(codes)
        columns = (codes.to(official_weights.dtype) @ official_weights).argmax(1)
        official = torch.tensor([
            backend.class_ids[index] for index in columns.cpu().tolist()
        ])
        correct["matched_ridge_primary"] += int((primary == targets).sum().item())
        correct["official_ridge0_sensitivity"] += int((official == targets).sum().item())
        total += len(batch_indices)
    return {name: 100.0 * value / total for name, value in correct.items()}


def _run_loranpac_unit(
    *, config: dict, sources: dict, train: dict, replicate: dict, width: int,
    method: str, projection: torch.Tensor, order: list[int],
    training_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device, target_total_bytes: int,
) -> dict:
    budget = LORANPAC_BUDGET[method]
    contract = _rank_contract_for_target(
        config, width=width, budget=budget,
        target_total_bytes=target_total_bytes,
    )
    matched_ridge = float(config["selected_ridge_by_width"][str(width)])
    backend = LoRanPACTSVBackend(
        dimension=width,
        truncate_percent=float(config["loranpac"]["truncate_percent"]),
        max_rank=int(contract["derived_max_rank"]),
        ridge_lambda=matched_ridge,
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
    )
    encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ projection
    )
    records, encoding_seconds, update_seconds = [], 0.0, 0.0
    for task, train_indices in enumerate(training_parts, start=1):
        m5._sync(device)
        started = time.perf_counter()
        codes = m5._encode_indices(
            encoder, train["features"], train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        m5._sync(device)
        encoding_seconds += time.perf_counter() - started
        m5._sync(device)
        started = time.perf_counter()
        backend.update(codes, train["labels"][train_indices])
        m5._sync(device)
        update_seconds += time.perf_counter() - started
        official_weights, official_residual = backend.solve_weights(0.0)
        seen_validation = torch.cat(validation_parts[:task])
        accuracy = _evaluate_loranpac_heads(
            backend=backend, official_weights=official_weights, encoder=encoder,
            features=train["features"], labels=train["labels"],
            indices=seen_validation,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )
        orthogonality = _orthogonality_metrics(backend.U, spectral=task == 1)
        total_bytes = persistent_tensor_bytes(
            {"projection": projection, **backend.persistent_tensors()}
        )
        if total_bytes > contract["target_total_persistent_bytes"]:
            raise AssertionError(
                f"M14 LoRanPAC exceeds {budget}/{width} budget at task {task}"
            )
        records.append({
            "task": task,
            "accuracy_percent": accuracy,
            "effective_rank": backend.effective_rank,
            "total_persistent_bytes": total_bytes,
            "matched_ridge_solver_relative_residual": float(
                backend.diagnostics["solver_relative_residual"]
            ),
            "official_ridge0_solver_relative_residual": float(official_residual),
            "orthogonality": orthogonality,
        })
        print(
            f"M14 c={replicate['class_order_seed']} w={width} {method} "
            f"task={task}/10 rank={backend.effective_rank} "
            f"val={accuracy['matched_ridge_primary']:.4f}", flush=True,
        )
        del codes, official_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final_bytes = records[-1]["total_persistent_bytes"]
    if final_bytes != contract["expected_final_total_persistent_bytes"]:
        raise AssertionError("M14 LoRanPAC symbolic/actual final-state mismatch")
    names = ("matched_ridge_primary", "official_ridge0_sensitivity")
    aia = {
        name: sum(record["accuracy_percent"][name] for record in records) / len(records)
        for name in names
    }
    return {
        "identity": _unit_identity(config, replicate, width, method),
        "uses_test_set": False,
        "class_order": order,
        "training_indices_sha256": m6._sequence_sha256(training_parts),
        "validation_indices_sha256": m6._sequence_sha256(validation_parts),
        "rank_contract": contract,
        "records": records,
        "validation_aia_percent": aia["matched_ridge_primary"],
        "final_validation_accuracy_percent": records[-1]["accuracy_percent"][
            "matched_ridge_primary"
        ],
        "official_ridge0_sensitivity_aia_percent": aia[
            "official_ridge0_sensitivity"
        ],
        "official_ridge0_sensitivity_final_percent": records[-1][
            "accuracy_percent"
        ]["official_ridge0_sensitivity"],
        "final_total_persistent_bytes": final_bytes,
        "representation_encoding_seconds": encoding_seconds,
        "analytic_update_seconds": update_seconds,
    }


def _validate_unit(
    unit: dict, identity: dict, config: dict, sources: dict,
    target_total_bytes: int | None = None,
) -> None:
    if unit.get("identity") != identity or unit.get("uses_test_set") is not False:
        raise RuntimeError("M14 resume unit identity/boundary mismatch")
    records = unit.get("records", [])
    if len(records) != config["num_tasks"]:
        raise RuntimeError("M14 resume unit task count mismatch")
    if identity["method"] in LORANPAC_BUDGET:
        if target_total_bytes is None:
            raise RuntimeError("M14 paired state target missing")
        expected = _rank_contract_for_target(
            config, width=identity["width"], budget=identity["budget_target"],
            target_total_bytes=target_total_bytes,
        )
        if unit.get("rank_contract") != expected:
            raise RuntimeError("M14 resumed LoRanPAC rank contract mismatch")
    values = [
        unit.get("validation_aia_percent"),
        unit.get("final_validation_accuracy_percent"),
        unit.get("representation_encoding_seconds"),
        unit.get("analytic_update_seconds"),
    ]
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise RuntimeError("M14 resumed unit has invalid summary")


def _mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def _aggregate(units: list[dict], config: dict) -> list[dict]:
    rows = []
    comparator = {
        "exact": "exact",
        "p2b_int8": "exact",
        "adaptive_int8_fp16": "exact",
        "loranpac_p2b_budget": "p2b_int8",
        "loranpac_adaptive_budget": "adaptive_int8_fp16",
    }
    for width in config["widths"]:
        width_units = [unit for unit in units if unit["identity"]["width"] == width]
        by_method_seed = {
            method: {
                unit["identity"]["class_order_seed"]: unit
                for unit in width_units if unit["identity"]["method"] == method
            }
            for method in METHODS
        }
        for method in METHODS:
            values = [by_method_seed[method][seed] for seed in range(4101, 4107)]
            reference = by_method_seed[comparator[method]]
            delta_aia = [
                value["validation_aia_percent"]
                - reference[value["identity"]["class_order_seed"]]["validation_aia_percent"]
                for value in values
            ]
            delta_final = [
                value["final_validation_accuracy_percent"]
                - reference[value["identity"]["class_order_seed"]][
                    "final_validation_accuracy_percent"
                ]
                for value in values
            ]
            rows.append({
                "width": width,
                "method": method,
                "paired_reference": comparator[method],
                "validation_aia_percent": _mean_sd([
                    value["validation_aia_percent"] for value in values
                ]),
                "final_validation_accuracy_percent": _mean_sd([
                    value["final_validation_accuracy_percent"] for value in values
                ]),
                "paired_delta_aia_percent": _mean_sd(delta_aia),
                "paired_delta_final_percent": _mean_sd(delta_final),
                "positive_delta_aia_seed_count": sum(delta > 0 for delta in delta_aia),
                "paired_delta_aia_by_seed": delta_aia,
                "paired_delta_final_by_seed": delta_final,
                "final_total_persistent_bytes": _mean_sd([
                    float(value["final_total_persistent_bytes"]) for value in values
                ]),
                "analytic_update_seconds": _mean_sd([
                    value["analytic_update_seconds"] for value in values
                ]),
            })
    return rows


def _write_csv(path: Path, aggregate: list[dict]) -> None:
    rows = []
    for item in aggregate:
        rows.append({
            "width": item["width"],
            "method": item["method"],
            "paired_reference": item["paired_reference"],
            "aia_mean": item["validation_aia_percent"]["mean"],
            "aia_sample_sd": item["validation_aia_percent"]["sample_standard_deviation"],
            "final_mean": item["final_validation_accuracy_percent"]["mean"],
            "final_sample_sd": item["final_validation_accuracy_percent"]["sample_standard_deviation"],
            "paired_delta_aia_mean": item["paired_delta_aia_percent"]["mean"],
            "paired_delta_aia_sample_sd": item["paired_delta_aia_percent"]["sample_standard_deviation"],
            "positive_delta_aia_seed_count": item["positive_delta_aia_seed_count"],
            "state_bytes_mean": item["final_total_persistent_bytes"]["mean"],
            "update_seconds_mean": item["analytic_update_seconds"]["mean"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M14 requires a clean source checkout")
    cache_dir = Path(args.feature_cache_dir).resolve()
    if (cache_dir / "test.pt").exists():
        raise RuntimeError("M14 refuses a visible test.pt")
    sources, m13n_source = _load_sources(config, args)
    train, _, metadata = validate_cache(
        cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    m6_provenance = sources["m6"]["provenance"]
    cache_identity = {
        "checkpoint": metadata.get("checkpoint_sha256") == config["checkpoint_sha256"],
        "train_sha256": _sha256_file(cache_dir / "train.pt")
        == m6_provenance["train_sha256"],
        "feature_dimension": int(train["features"].shape[1])
        == config["ranpac"]["feature_dimension"],
        "class_inventory": sorted(map(int, torch.unique(train["labels"]).tolist()))
        == list(range(config["num_classes"])),
        "test_absent": not (cache_dir / "test.pt").exists(),
    }
    output_dir = Path(args.output_dir).resolve()
    units_dir = output_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    units = []
    projection_hashes = {}
    split_hashes = {}
    for replicate in config["replicates"]:
        order, training_parts, validation_parts = _parts(config, train, replicate)
        seed = replicate["class_order_seed"]
        split_hashes[str(seed)] = {
            "class_order": order,
            "training_indices_sha256": m6._sequence_sha256(training_parts),
            "validation_indices_sha256": m6._sequence_sha256(validation_parts),
        }
        generator = torch.Generator(device="cpu").manual_seed(
            replicate["projection_seed"]
        )
        full_projection = torch.randn(
            config["ranpac"]["feature_dimension"],
            config["ranpac"]["maximum_expand_dimension"],
            generator=generator, dtype=torch.float32,
        ).to(device)
        for width in config["widths"]:
            projection = full_projection[:, :width].contiguous()
            projection_hashes[f"{seed}/{width}"] = m4._tensor_content_sha256(projection)
            paired_state_bytes = {}
            for method in METHODS:
                identity = _unit_identity(config, replicate, width, method)
                destination = units_dir / f"s{seed}_w{width}_{method}.json"
                target_total_bytes = None
                if method in LORANPAC_BUDGET:
                    paired_method = LORANPAC_BUDGET[method]
                    if paired_method not in paired_state_bytes:
                        raise RuntimeError(
                            f"M14 paired backend must precede {method}"
                        )
                    target_total_bytes = paired_state_bytes[paired_method]
                if destination.is_file():
                    unit = json.loads(destination.read_text(encoding="utf-8"))
                    print(f"M14 UNIT REUSED {destination.name}", flush=True)
                elif method in LORANPAC_BUDGET:
                    unit = _run_loranpac_unit(
                        config=config, sources=sources, train=train,
                        replicate=replicate, width=width, method=method,
                        projection=projection, order=order,
                        training_parts=training_parts,
                        validation_parts=validation_parts, device=device,
                        target_total_bytes=target_total_bytes,
                    )
                    _atomic_json(destination, unit)
                else:
                    unit = _run_additive_unit(
                        config=config, sources=sources, train=train,
                        replicate=replicate, width=width, method=method,
                        projection=projection, order=order,
                        training_parts=training_parts,
                        validation_parts=validation_parts, device=device,
                    )
                    _atomic_json(destination, unit)
                _validate_unit(
                    unit, identity, config, sources,
                    target_total_bytes=target_total_bytes,
                )
                units.append(unit)
                if method in {"p2b_int8", "adaptive_int8_fp16"}:
                    paired_state_bytes[method] = int(
                        unit["final_total_persistent_bytes"]
                    )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del projection
        del full_projection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    expected_units = len(config["replicates"]) * len(config["widths"]) * len(METHODS)
    loranpac_units = [
        unit for unit in units if unit["identity"]["method"] in LORANPAC_BUDGET
    ]
    additive_units = [
        unit for unit in units if unit["identity"]["method"] not in LORANPAC_BUDGET
    ]
    thresholds = config["integrity_gates"]
    task1_solver = max(
        max(
            unit["records"][0]["matched_ridge_solver_relative_residual"],
            unit["records"][0]["official_ridge0_solver_relative_residual"],
        ) for unit in loranpac_units
    )
    later_solver = max(
        max(
            record["matched_ridge_solver_relative_residual"],
            record["official_ridge0_solver_relative_residual"],
        )
        for unit in loranpac_units for record in unit["records"][1:]
    )
    maximum_normalized = max(
        record["orthogonality"]["frobenius_over_sqrt_rank"]
        for unit in loranpac_units for record in unit["records"]
    )
    maximum_task1_spectral = max(
        unit["records"][0]["orthogonality"]["spectral_norm"]
        for unit in loranpac_units
    )
    paired_units = {
        (
            unit["identity"]["class_order_seed"],
            unit["identity"]["width"],
            unit["identity"]["method"],
        ): unit
        for unit in additive_units
    }
    gates = {
        "source_artifact_identity": True,
        "train_cache_identity": all(cache_identity.values()),
        "all_units_complete": len(units) == expected_units,
        "all_metrics_finite": _all_finite(units),
        "rank_caps_derived_from_bytes": all(
            unit["rank_contract"] == _rank_contract_for_target(
                config,
                width=unit["identity"]["width"],
                budget=unit["identity"]["budget_target"],
                target_total_bytes=paired_units[
                    (
                        unit["identity"]["class_order_seed"],
                        unit["identity"]["width"],
                        unit["identity"]["budget_target"],
                    )
                ]["final_total_persistent_bytes"],
            ) for unit in loranpac_units
        ),
        "total_state_not_above_target": all(
            unit["final_total_persistent_bytes"]
            <= unit["rank_contract"]["target_total_persistent_bytes"]
            for unit in loranpac_units
        ),
        "budget_underfill_below_one_rank": all(
            0 <= unit["rank_contract"]["final_budget_underfill_bytes"]
            < unit["rank_contract"]["bytes_per_additional_rank"]
            for unit in loranpac_units
        ),
        "normalized_orthogonality": maximum_normalized
        <= thresholds["maximum_normalized_orthogonality_residual"],
        "task1_spectral_orthogonality": maximum_task1_spectral
        <= thresholds["maximum_task1_spectral_orthogonality_residual"],
        "task1_solver_residual": task1_solver
        <= thresholds["maximum_task1_solver_relative_residual"],
        "later_task_solver_residual": later_solver
        <= thresholds["maximum_later_task_solver_relative_residual"],
        "srq_solver_residual": max(
            unit["maximum_solver_relative_residual"] for unit in additive_units
        ) <= thresholds["maximum_srq_solver_relative_residual"],
    }
    aggregate = _aggregate(units, config)
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": (
            "PASS_M14_LORANPAC_MULTISEED_TRAIN_ONLY"
            if passed else "FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY"
        ),
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "purpose": "multi-seed equal-byte LoRanPAC confirmation",
            "official_end_to_end_loranpac_reproduction": False,
            "accuracy_gate": None,
            "main_loranpac_path_uses_qr_reorthogonalization_patch": False,
            "raw_frobenius_reported_but_not_gated": True,
        },
        "sources": {
            "m6": config["source_m6"],
            "m11": config["source_m11"],
            "m13n": config["source_m13n"],
            "m13_status_remains": m13n_source["scope"]["m13_status_remains"],
        },
        "cache_identity": cache_identity,
        "split_identities": split_hashes,
        "projection_sha256": projection_hashes,
        "units": units,
        "aggregate": aggregate,
        "summary": {
            "completed_units": len(units),
            "expected_units": expected_units,
            "maximum_normalized_orthogonality_residual": maximum_normalized,
            "maximum_task1_spectral_orthogonality_residual": maximum_task1_spectral,
            "maximum_task1_solver_relative_residual": task1_solver,
            "maximum_later_task_solver_relative_residual": later_solver,
        },
        "gates": gates,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()),
            "config_sha256": _sha256_file(config_path),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "loranpac_backend_sha256": _sha256_file(
                ROOT / "methods/frontends/loranpac.py"
            ),
            "train_sha256": _sha256_file(cache_dir / "train.pt"),
        },
    }
    _atomic_json(output_dir / "m14_results.json", payload)
    _write_csv(output_dir / "m14_aggregate.csv", aggregate)
    print(f"M14 STATUS: {payload['status']}", flush=True)
    if not passed:
        raise RuntimeError("M14 integrity/numerical gates failed")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--source-m6-artifact", required=True)
    run_parser.add_argument("--source-m11-artifact", required=True)
    run_parser.add_argument("--source-m13n-artifact", required=True)
    run_parser.add_argument("--feature-cache-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "run":
        run(args)


if __name__ == "__main__":
    main()
