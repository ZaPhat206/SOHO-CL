"""M15: non-predictive closure audit for the M14 LoRanPAC task-one residual."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import srq_generalization_m4 as m4  # noqa: E402
from tools import srq_generalization_m6 as m6  # noqa: E402
from methods.frontends.loranpac import official_loranpac_rank  # noqa: E402
from tools.experiment_runner import (  # noqa: E402
    split,
    train_validation_indices,
    validate_cache,
)


TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "uses_test_set", "computes_predictive_metrics",
    "uses_accuracy_for_selection", "seed", "num_classes", "num_tasks",
    "outer_validation_fraction", "diagnostic_seeds", "seed_roles", "widths",
    "budget_targets", "source_m14", "train_identity", "ranpac", "loranpac",
    "audit", "gates",
}
METHOD_BY_BUDGET = {
    "p2b_int8": "loranpac_p2b_budget",
    "adaptive_int8_fp16": "loranpac_adaptive_budget",
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
        raise ValueError("M15 config keys/schema mismatch")
    if not (
        config["uses_test_set"] is False
        and config["computes_predictive_metrics"] is False
        and config["uses_accuracy_for_selection"] is False
        and config["seed"] == 2025
        and config["num_classes"] == 100
        and config["num_tasks"] == 10
        and config["diagnostic_seeds"] == [4105, 4101]
        and config["seed_roles"] == {
            "4105": "m14_task1_solver_residual_maximum",
            "4101": "m14_passing_sentinel",
        }
        and config["widths"] == [10000, 20000]
        and config["budget_targets"] == ["p2b_int8", "adaptive_int8_fp16"]
        and 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("M15 locked non-predictive design changed")
    source = config["source_m14"]
    if not (
        set(source) == {
            "artifact_filename", "artifact_sha256", "result_member",
            "result_sha256", "required_status", "required_failed_gate",
            "locked_task1_solver_threshold",
            "observed_maximum_task1_solver_residual", "source_commit",
        }
        and source["artifact_filename"]
        == "srq_generalization_m14_loranpac_multiseed_train_only.zip"
        and source["result_member"] == "m14_results.json"
        and source["required_status"]
        == "FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY"
        and source["required_failed_gate"] == "task1_solver_residual"
        and source["locked_task1_solver_threshold"] == 1e-3
        and source["observed_maximum_task1_solver_residual"]
        == 0.001914756829772187
        and all(len(source[name]) == 64 for name in ("artifact_sha256", "result_sha256"))
        and len(source["source_commit"]) == 40
    ):
        raise ValueError("M15 M14 source lock changed")
    identity = config["train_identity"]
    if not (
        set(identity) == {"train_pt_sha256", "feature_dimension", "class_inventory"}
        and len(identity["train_pt_sha256"]) == 64
        and identity["feature_dimension"] == 768
        and identity["class_inventory"] == list(range(100))
    ):
        raise ValueError("M15 train identity changed")
    ranpac = config["ranpac"]
    if not (
        ranpac == {
            "feature_dimension": 768,
            "maximum_expand_dimension": 20000,
            "projection_distribution": "standard_normal",
            "activation": "relu",
            "encode_batch_size": 256,
        }
    ):
        raise ValueError("M15 RanPAC frontend changed")
    loranpac = config["loranpac"]
    if not (
        loranpac.get("upstream_commit") == "32782f9d260e5d722de5702675ed66cca8234883"
        and loranpac.get("truncate_percent") == 25.0
        and loranpac.get("rank_rule") == "official_code_python_round"
        and loranpac.get("official_ridge_lambda") == 0.0
        and loranpac.get("matched_ridge_by_width")
        == {"10000": 1_000_000.0, "20000": 1_000_000.0}
    ):
        raise ValueError("M15 LoRanPAC identity changed")
    audit = config["audit"]
    if not (
        audit.get("raw_solver_dtypes")
        == ["float32", "float64_same_fp32_factor"]
        and audit.get("qr_mode")
        == "reduced_once_at_largest_rank_then_prefix"
        and audit.get("qr_core_rule") == "T_diag_s_squared_T_transpose"
        and audit.get("qr_is_diagnostic_only") is True
        and audit.get("changes_m14_predictions") is False
        and audit.get("predictive_fields_permitted") is False
    ):
        raise ValueError("M15 audit semantics changed")
    gates = config["gates"]
    required_true = {
        "require_clean_git", "require_source_artifact_identity",
        "require_m14_status_remains_fail", "require_loranpac_backend_identity",
        "require_train_identity", "require_split_identity_from_m14",
        "require_projection_identity_from_m14", "require_rank_identity_from_m14",
        "require_all_eight_records", "require_all_metrics_finite",
    }
    if not all(gates.get(name) is True for name in required_true):
        raise ValueError("M15 integrity gates changed")
    if not (
        gates.get("maximum_qr_factor_reconstruction_relative_error") == 1e-5
        and gates.get("maximum_fp64_system_preserving_core_backward_error") == 1e-10
        and gates.get("minimum_fp64_backward_error_improvement_factor") == 100.0
        and gates.get("accuracy_gate", "missing") is None
    ):
        raise ValueError("M15 numerical gates changed")
    return config


def _load_m14_source(config: dict, path: str | Path) -> tuple[dict, dict]:
    source = config["source_m14"]
    artifact = Path(path).resolve()
    if artifact.name != source["artifact_filename"]:
        raise ValueError("M15 M14 source filename mismatch")
    if _sha256_file(artifact) != source["artifact_sha256"]:
        raise ValueError("M15 M14 artifact SHA-256 mismatch")
    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or source["result_member"] not in names:
            raise ValueError("M15 M14 archive member contract failed")
        raw = archive.read(source["result_member"])
    if hashlib.sha256(raw).hexdigest() != source["result_sha256"]:
        raise ValueError("M15 M14 result SHA-256 mismatch")
    result = json.loads(raw.decode("utf-8"))
    failed_gate = source["required_failed_gate"]
    other_gates = {name: value for name, value in result.get("gates", {}).items()
                   if name != failed_gate}
    loranpac_units = [
        unit for unit in result.get("units", [])
        if unit.get("identity", {}).get("method") in METHOD_BY_BUDGET.values()
    ]
    source_maximum = max(
        max(
            unit["records"][0]["matched_ridge_solver_relative_residual"],
            unit["records"][0]["official_ridge0_solver_relative_residual"],
        )
        for unit in loranpac_units
    )
    culprit = _source_unit(result, 4105, 10000, "p2b_int8")
    sentinel = _source_unit(result, 4101, 10000, "p2b_int8")
    if not (
        result.get("status") == source["required_status"]
        and result.get("uses_test_set") is False
        and result.get("accuracy_based_selection") is False
        and result.get("gates", {}).get(failed_gate) is False
        and other_gates
        and all(value is True for value in other_gates.values())
        and result.get("summary", {}).get("maximum_task1_solver_relative_residual")
        == source["observed_maximum_task1_solver_residual"]
        and result.get("summary", {}).get("completed_units") == 60
        and result.get("summary", {}).get("expected_units") == 60
        and len(loranpac_units) == 24
        and source_maximum == source["observed_maximum_task1_solver_residual"]
        and culprit["records"][0]["official_ridge0_solver_relative_residual"]
        == source_maximum
        and max(
            sentinel["records"][0]["matched_ridge_solver_relative_residual"],
            sentinel["records"][0]["official_ridge0_solver_relative_residual"],
        ) <= source["locked_task1_solver_threshold"]
        and result.get("provenance", {}).get("git_commit") == source["source_commit"]
    ):
        raise ValueError("M15 M14 failure/status boundary mismatch")
    identity = {
        "artifact_sha256": source["artifact_sha256"],
        "result_sha256": source["result_sha256"],
        "reported_status": result["status"],
        "failed_gate": failed_gate,
        "locked_threshold": source["locked_task1_solver_threshold"],
        "observed_maximum": source["observed_maximum_task1_solver_residual"],
        "source_commit": source["source_commit"],
        "source_loranpac_backend_sha256": result["provenance"][
            "loranpac_backend_sha256"
        ],
    }
    return result, identity


def _source_unit(result: dict, seed: int, width: int, budget: str) -> dict:
    method = METHOD_BY_BUDGET[budget]
    matches = [
        unit for unit in result["units"]
        if unit["identity"]["class_order_seed"] == seed
        and unit["identity"]["projection_seed"] == seed
        and unit["identity"]["split_seed"] == seed
        and unit["identity"]["width"] == width
        and unit["identity"]["method"] == method
        and unit["identity"]["budget_target"] == budget
    ]
    if len(matches) != 1 or matches[0].get("uses_test_set") is not False:
        raise ValueError(f"M15 missing unique source unit {seed}/{width}/{budget}")
    return matches[0]


def _parts(config: dict, train: dict, seed: int):
    order = random.Random(seed).sample(list(range(config["num_classes"])), 100)
    task_indices = split(train["labels"], order, config["num_tasks"])
    training, validation = train_validation_indices(
        train["labels"], task_indices, seed, config["outer_validation_fraction"]
    )
    return order, training, validation


def _encode_first_task(
    features: torch.Tensor,
    indices: torch.Tensor,
    projection: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        values = features[batch_indices].to(device=device, dtype=torch.float32)
        parts.append(torch.relu(values @ projection))
    return torch.cat(parts, dim=0)


def _task_cross(codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    class_ids = torch.unique(labels, sorted=True).to(device=codes.device)
    columns = torch.searchsorted(class_ids, labels.to(device=codes.device))
    targets = torch.nn.functional.one_hot(
        columns, num_classes=int(class_ids.numel())
    ).to(dtype=codes.dtype)
    return codes.T @ targets


def _orthogonality_metrics(basis: torch.Tensor) -> dict[str, float]:
    rank = int(basis.shape[1])
    identity = torch.eye(rank, device=basis.device, dtype=basis.dtype)
    error = basis.T @ basis - identity
    symmetric = (error + error.T) * 0.5
    frobenius = float(torch.linalg.vector_norm(error).item())
    return {
        "raw_frobenius": frobenius,
        "frobenius_over_sqrt_rank": frobenius / math.sqrt(rank),
        "spectral_norm": float(torch.linalg.eigvalsh(symmetric).abs().max().item()),
        "maximum_absolute_entry": float(error.abs().max().item()),
    }


def _raw_formula_metrics(
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    cross: torch.Tensor,
    ridge_lambda: float,
    dtype: torch.dtype,
) -> tuple[dict[str, float], torch.Tensor]:
    work_u = basis.to(dtype=dtype)
    work_s = singular_values.to(dtype=dtype)
    work_cross = cross.to(dtype=dtype)
    denominator = work_s.square() + float(ridge_lambda)
    if bool((denominator <= torch.finfo(dtype).tiny).any()):
        raise RuntimeError("M15 encountered a zero retained singular direction")
    projected_cross = work_u.T @ work_cross
    coordinates = projected_cross / denominator.unsqueeze(1)
    weights = work_u @ coordinates
    projected_weights = work_u.T @ weights
    residual = denominator.unsqueeze(1) * projected_weights - projected_cross
    residual_norm = torch.linalg.vector_norm(residual)
    cross_norm = torch.linalg.vector_norm(projected_cross)
    backward_denominator = (
        denominator.abs().max() * torch.linalg.vector_norm(projected_weights)
        + cross_norm
    )
    metrics = {
        "legacy_relative_residual": float(
            residual_norm.item() / max(float(cross_norm.item()), 1.0)
        ),
        "normwise_backward_error": float(
            residual_norm.item()
            / max(float(backward_denominator.item()), torch.finfo(dtype).tiny)
        ),
        "projected_cross_frobenius": float(cross_norm.item()),
        "residual_frobenius": float(residual_norm.item()),
        "diagonal_condition_proxy": float(
            (denominator.max() / denominator.min()).item()
        ),
    }
    return metrics, weights


def _system_preserving_qr_metrics(
    *,
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    cross: torch.Tensor,
    q_prefix: torch.Tensor,
    triangular_prefix: torch.Tensor,
    ridge_lambda: float,
    dtype: torch.dtype,
    raw_weights: torch.Tensor,
) -> dict[str, float]:
    work_u = basis.to(dtype=dtype)
    work_s = singular_values.to(dtype=dtype)
    work_cross = cross.to(dtype=dtype)
    work_q = q_prefix.to(dtype=dtype)
    work_t = triangular_prefix.to(dtype=dtype)
    reconstructed = work_q @ work_t
    reconstruction_error = torch.linalg.vector_norm(work_u - reconstructed)
    reconstruction_scale = max(float(torch.linalg.vector_norm(work_u).item()), 1.0)
    factor = work_t * work_s.unsqueeze(0)
    core = factor @ factor.T
    operator = core + float(ridge_lambda) * torch.eye(
        core.shape[0], device=core.device, dtype=dtype
    )
    projected_cross = work_q.T @ work_cross
    coordinates = torch.linalg.solve(operator, projected_cross)
    algebraic_residual = operator @ coordinates - projected_cross
    residual_norm = torch.linalg.vector_norm(algebraic_residual)
    cross_norm = torch.linalg.vector_norm(projected_cross)
    backward_denominator = (
        torch.linalg.vector_norm(operator)
        * torch.linalg.vector_norm(coordinates)
        + cross_norm
    )
    weights = work_q @ coordinates
    raw = raw_weights.to(dtype=dtype)
    weight_scale = max(float(torch.linalg.vector_norm(raw).item()), 1.0)
    return {
        "factor_reconstruction_relative_error": float(
            reconstruction_error.item() / reconstruction_scale
        ),
        "core_relative_residual": float(
            residual_norm.item() / max(float(cross_norm.item()), 1.0)
        ),
        "core_normwise_backward_error": float(
            residual_norm.item()
            / max(float(backward_denominator.item()), torch.finfo(dtype).tiny)
        ),
        "weight_difference_from_raw_relative": float(
            torch.linalg.vector_norm(weights - raw).item() / weight_scale
        ),
    }


def _all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _predictive_fields_absent(value) -> bool:
    forbidden = ("accuracy", "aia", "logit", "prediction")
    if isinstance(value, dict):
        return all(
            not any(token in str(key).lower() for token in forbidden)
            and _predictive_fields_absent(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_predictive_fields_absent(item) for item in value)
    return True


def _write_csv(path: Path, records: list[dict]) -> None:
    rows = []
    for record in records:
        for ridge_name, ridge in record["ridges"].items():
            rows.append({
                "seed": record["seed"],
                "seed_role": record["seed_role"],
                "width": record["width"],
                "budget_target": record["budget_target"],
                "rank": record["rank"],
                "ridge": ridge_name,
                "source_m14_residual": ridge["source_m14_legacy_relative_residual"],
                "raw_fp32_residual": ridge["raw_fp32"]["legacy_relative_residual"],
                "raw_fp64_residual": ridge["raw_fp64_same_factor"]["legacy_relative_residual"],
                "raw_fp64_backward_error": ridge["raw_fp64_same_factor"]["normwise_backward_error"],
                "qr_fp32_reconstruction": ridge["system_preserving_qr_fp32"]["factor_reconstruction_relative_error"],
                "qr_fp64_core_backward_error": ridge["system_preserving_qr_fp64"]["core_normwise_backward_error"],
                "fp64_backward_improvement": ridge["fp64_backward_error_improvement_factor"],
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M15 requires a clean source checkout")
    cache_dir = Path(args.feature_cache_dir).resolve()
    if (cache_dir / "test.pt").exists():
        raise RuntimeError("M15 refuses a visible test.pt")
    source_result, source_identity = _load_m14_source(
        config, args.source_m14_artifact
    )
    current_backend_sha = _sha256_file(ROOT / "methods/frontends/loranpac.py")
    backend_identity_matches = (
        current_backend_sha == source_identity["source_loranpac_backend_sha256"]
    )
    train, _, metadata = validate_cache(
        cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    expected_train = config["train_identity"]
    train_identity = {
        "train_pt_sha256": _sha256_file(cache_dir / "train.pt"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "feature_dimension": int(train["features"].shape[1]),
        "class_inventory": sorted(map(int, torch.unique(train["labels"]).tolist())),
        "test_absent": not (cache_dir / "test.pt").exists(),
    }
    train_identity_matches = (
        train_identity["train_pt_sha256"] == expected_train["train_pt_sha256"]
        and train_identity["checkpoint_sha256"] == config["checkpoint_sha256"]
        and train_identity["feature_dimension"] == expected_train["feature_dimension"]
        and train_identity["class_inventory"] == expected_train["class_inventory"]
        and train_identity["test_absent"]
    )
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    units_dir = output_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    records = []
    split_identity_matches = True
    projection_identity_matches = True
    rank_identity_matches = True
    for seed in config["diagnostic_seeds"]:
        order, training_parts, validation_parts = _parts(config, train, seed)
        training_hash = m6._sequence_sha256(training_parts)
        validation_hash = m6._sequence_sha256(validation_parts)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        full_projection = torch.randn(
            config["ranpac"]["feature_dimension"],
            config["ranpac"]["maximum_expand_dimension"],
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        for width in config["widths"]:
            source_units = {
                budget: _source_unit(source_result, seed, width, budget)
                for budget in config["budget_targets"]
            }
            split_identity_matches &= all(
                unit["class_order"] == order
                and unit["training_indices_sha256"] == training_hash
                and unit["validation_indices_sha256"] == validation_hash
                for unit in source_units.values()
            )
            projection = full_projection[:, :width].contiguous()
            projection_key = f"{seed}/{width}"
            projection_hash = m4._tensor_content_sha256(projection)
            projection_identity_matches &= (
                projection_hash == source_result["projection_sha256"][projection_key]
            )
            ranks = {
                budget: int(unit["records"][0]["effective_rank"])
                for budget, unit in source_units.items()
            }
            for budget, source_unit in source_units.items():
                derived_task1_rank = official_loranpac_rank(
                    len(training_parts[0]),
                    dimension=width,
                    truncate_percent=float(config["loranpac"]["truncate_percent"]),
                    max_rank=int(source_unit["rank_contract"]["derived_max_rank"]),
                )
                rank_identity_matches &= ranks[budget] == derived_task1_rank
            destination = units_dir / f"s{seed}_w{width}.json"
            if destination.is_file():
                chunk = json.loads(destination.read_text(encoding="utf-8"))
                expected_pairs = [
                    (budget, ranks[budget]) for budget in config["budget_targets"]
                ]
                observed_pairs = [
                    (record["budget_target"], record["rank"])
                    for record in chunk.get("records", [])
                ]
                if not (
                    chunk.get("schema_version") == 1
                    and chunk.get("study_id") == config["study_id"]
                    and chunk.get("seed") == seed
                    and chunk.get("width") == width
                    and chunk.get("training_indices_sha256") == training_hash
                    and chunk.get("validation_indices_sha256") == validation_hash
                    and chunk.get("projection_sha256") == projection_hash
                    and chunk.get("source_m14_result_sha256")
                    == source_identity["result_sha256"]
                    and chunk.get("loranpac_backend_sha256") == current_backend_sha
                    and observed_pairs == expected_pairs
                    and _all_finite(chunk.get("records", []))
                    and _predictive_fields_absent(chunk.get("records", []))
                ):
                    raise ValueError(f"M15 invalid resume unit {destination.name}")
                records.extend(chunk["records"])
                print(f"M15 UNIT REUSED {destination.name}", flush=True)
                del projection
                continue
            codes = _encode_first_task(
                train["features"], training_parts[0], projection,
                int(config["ranpac"]["encode_batch_size"]), device,
            )
            labels = train["labels"][training_parts[0]]
            cross = _task_cross(codes, labels)
            left, singular_values, right = torch.linalg.svd(
                codes.T, full_matrices=False
            )
            del right, codes
            maximum_rank = max(ranks.values())
            if maximum_rank > left.shape[1]:
                raise RuntimeError("M15 source rank exceeds reconstructed task-one SVD")
            q_max, triangular_max = torch.linalg.qr(
                left[:, :maximum_rank].contiguous(), mode="reduced"
            )
            width_records = []
            for budget in config["budget_targets"]:
                source_unit = source_units[budget]
                rank = ranks[budget]
                basis = left[:, :rank].contiguous()
                values = singular_values[:rank].contiguous()
                q_prefix = q_max[:, :rank].contiguous()
                triangular_prefix = triangular_max[:rank, :rank].contiguous()
                orthogonality = _orthogonality_metrics(basis)
                ridge_records = {}
                for ridge_name, ridge_lambda in (
                    ("matched", float(config["loranpac"]["matched_ridge_by_width"][str(width)])),
                    ("official_zero", float(config["loranpac"]["official_ridge_lambda"])),
                ):
                    raw32, weights32 = _raw_formula_metrics(
                        basis, values, cross, ridge_lambda, torch.float32
                    )
                    raw64, weights64 = _raw_formula_metrics(
                        basis, values, cross, ridge_lambda, torch.float64
                    )
                    qr32 = _system_preserving_qr_metrics(
                        basis=basis, singular_values=values, cross=cross,
                        q_prefix=q_prefix, triangular_prefix=triangular_prefix,
                        ridge_lambda=ridge_lambda, dtype=torch.float32,
                        raw_weights=weights32,
                    )
                    qr64 = _system_preserving_qr_metrics(
                        basis=basis, singular_values=values, cross=cross,
                        q_prefix=q_prefix, triangular_prefix=triangular_prefix,
                        ridge_lambda=ridge_lambda, dtype=torch.float64,
                        raw_weights=weights64,
                    )
                    source_key = (
                        "matched_ridge_solver_relative_residual"
                        if ridge_name == "matched"
                        else "official_ridge0_solver_relative_residual"
                    )
                    improvement = raw64["normwise_backward_error"] / max(
                        qr64["core_normwise_backward_error"],
                        torch.finfo(torch.float64).tiny,
                    )
                    ridge_records[ridge_name] = {
                        "ridge_lambda": ridge_lambda,
                        "source_m14_legacy_relative_residual": float(
                            source_unit["records"][0][source_key]
                        ),
                        "raw_fp32": raw32,
                        "raw_fp64_same_factor": raw64,
                        "system_preserving_qr_fp32": qr32,
                        "system_preserving_qr_fp64": qr64,
                        "fp64_backward_error_improvement_factor": improvement,
                    }
                    del weights32, weights64
                record = {
                    "seed": seed,
                    "seed_role": config["seed_roles"][str(seed)],
                    "width": width,
                    "budget_target": budget,
                    "rank": rank,
                    "source_rank_cap": int(
                        source_unit["rank_contract"]["derived_max_rank"]
                    ),
                    "orthogonality_fp32": orthogonality,
                    "ridges": ridge_records,
                }
                width_records.append(record)
                print(
                    f"M15 seed={seed} width={width} budget={budget} rank={rank} "
                    f"raw={ridge_records['official_zero']['raw_fp32']['legacy_relative_residual']:.6g} "
                    f"core64={ridge_records['official_zero']['system_preserving_qr_fp64']['core_normwise_backward_error']:.3g}",
                    flush=True,
                )
                del basis, values, q_prefix, triangular_prefix
            chunk = {
                "schema_version": 1,
                "study_id": config["study_id"],
                "seed": seed,
                "width": width,
                "training_indices_sha256": training_hash,
                "validation_indices_sha256": validation_hash,
                "projection_sha256": projection_hash,
                "source_m14_result_sha256": source_identity["result_sha256"],
                "loranpac_backend_sha256": current_backend_sha,
                "records": width_records,
            }
            _atomic_json(destination, chunk)
            records.extend(width_records)
            del left, singular_values, q_max, triangular_max, cross, projection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del full_projection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    qr_reconstruction_max = max(
        ridge[variant]["factor_reconstruction_relative_error"]
        for record in records for ridge in record["ridges"].values()
        for variant in ("system_preserving_qr_fp32", "system_preserving_qr_fp64")
    )
    fp64_core_backward_max = max(
        ridge["system_preserving_qr_fp64"]["core_normwise_backward_error"]
        for record in records for ridge in record["ridges"].values()
    )
    fp64_improvement_min = min(
        ridge["fp64_backward_error_improvement_factor"]
        for record in records for ridge in record["ridges"].values()
    )
    thresholds = config["gates"]
    provisional = {
        "source_artifact_identity": True,
        "m14_status_remains_fail": source_identity["reported_status"]
        == "FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY",
        "loranpac_backend_identity": backend_identity_matches,
        "train_identity": train_identity_matches,
        "split_identity_from_m14": bool(split_identity_matches),
        "projection_identity_from_m14": bool(projection_identity_matches),
        "rank_identity_from_m14": bool(rank_identity_matches),
        "all_eight_records": len(records) == 8,
        "all_metrics_finite": _all_finite(records),
        "qr_factor_reconstruction": qr_reconstruction_max
        <= thresholds["maximum_qr_factor_reconstruction_relative_error"],
        "fp64_system_preserving_core_backward_error": fp64_core_backward_max
        <= thresholds["maximum_fp64_system_preserving_core_backward_error"],
        "fp64_backward_error_improvement": fp64_improvement_min
        >= thresholds["minimum_fp64_backward_error_improvement_factor"],
    }
    gates = dict(provisional)
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PENDING_M15_LORANPAC_TASK1_CLOSURE",
        "uses_test_set": False,
        "computes_predictive_metrics": False,
        "scope": {
            "m14_status_remains": source_identity["reported_status"],
            "qr_is_diagnostic_only": True,
            "m14_gate_was_not_relaxed": True,
            "m14_units_or_seeds_excluded": False,
            "changes_m14_predictions": False,
        },
        "source_identity": source_identity,
        "train_identity": train_identity,
        "records": records,
        "summary": {
            "completed_records": len(records),
            "expected_records": 8,
            "maximum_qr_factor_reconstruction_relative_error": qr_reconstruction_max,
            "maximum_fp64_system_preserving_core_backward_error": fp64_core_backward_max,
            "minimum_fp64_backward_error_improvement_factor": fp64_improvement_min,
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
            "loranpac_backend_sha256": current_backend_sha,
            "train_sha256": _sha256_file(cache_dir / "train.pt"),
        },
    }
    gates["predictive_fields_absent"] = _predictive_fields_absent(records)
    passed = all(gates.values())
    payload["status"] = (
        "PASS_M15_LORANPAC_TASK1_CLOSURE"
        if passed else "FAIL_M15_LORANPAC_TASK1_CLOSURE"
    )
    _atomic_json(output_dir / "m15_results.json", payload)
    _write_csv(output_dir / "m15_numerical_metrics.csv", records)
    print(f"M15 STATUS: {payload['status']}", flush=True)
    if not passed:
        raise RuntimeError("M15 integrity/numerical gates failed")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--source-m14-artifact", required=True)
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
