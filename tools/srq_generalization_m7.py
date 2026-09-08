"""Train-only task-wise error trajectory after the M6 width-scaling failure."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import ExactGramBackend, SquareRootBackend  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools.experiment_runner import split, train_validation_indices, validate_cache  # noqa: E402


TOP_KEYS = {
    "schema_version",
    "study_id",
    "dataset",
    "model_name",
    "checkpoint_sha256",
    "uses_test_set",
    "accuracy_based_selection",
    "seed",
    "num_classes",
    "num_tasks",
    "outer_validation_fraction",
    "statistics_dtype",
    "solver_dtype",
    "diagnostic_widths",
    "ridge_by_width",
    "source_m6",
    "ranpac",
    "diagnostics",
    "p2b",
    "gates",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sequence_sha256(parts: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.to(torch.int64).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M7 config keys/schema mismatch")
    if config["uses_test_set"] is not False:
        raise ValueError("M7 must remain train-only")
    if config["accuracy_based_selection"] is not False:
        raise ValueError("M7 is diagnostic and cannot select by accuracy")
    if config["seed"] != 2025:
        raise ValueError("M7 seed must remain aligned with M6")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("M7 locks FP32 statistics and solves")
    if (
        config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or not 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("invalid M7 class/task split")
    if config["diagnostic_widths"] != [10000, 20000]:
        raise ValueError("M7 diagnostic widths must remain 10k and 20k")
    if set(config["ridge_by_width"]) != {"10000", "20000"} or any(
        float(value) <= 0 for value in config["ridge_by_width"].values()
    ):
        raise ValueError("invalid M7 locked Ridge values")

    source = config["source_m6"]
    if set(source) != {
        "artifact_filename",
        "artifact_sha256",
        "result_member",
        "result_sha256",
        "required_status",
    } or source["required_status"] != "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY":
        raise ValueError("invalid M7 source-M6 lock")
    if len(source["artifact_sha256"]) != 64 or len(source["result_sha256"]) != 64:
        raise ValueError("invalid M7 source hashes")

    ranpac = config["ranpac"]
    if set(ranpac) != {
        "upstream_repository",
        "upstream_commit",
        "upstream_ranpac_py_sha256",
        "path",
        "feature_dimension",
        "maximum_expand_dimension",
        "projection_distribution",
        "activation",
        "projection_seed",
        "encode_batch_size",
        "evaluation_batch_size",
    }:
        raise ValueError("M7 RanPAC fields mismatch")
    if (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or ranpac["feature_dimension"] != 768
        or ranpac["maximum_expand_dimension"] != 20000
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or ranpac["projection_seed"] != 2025
        or min(ranpac["encode_batch_size"], ranpac["evaluation_batch_size"]) <= 0
    ):
        raise ValueError("invalid locked M7 RanPAC semantics")

    diagnostics = config["diagnostics"]
    required_diagnostics = {
        "system_probe_count",
        "system_probe_seed",
        "record_local_factor_error",
        "record_system_action_error",
        "record_weight_error",
        "record_logit_error",
        "record_prediction_agreement",
        "record_margin_certificate",
    }
    if set(diagnostics) != required_diagnostics:
        raise ValueError("M7 diagnostic fields mismatch")
    if diagnostics["system_probe_count"] <= 0 or any(
        diagnostics[name] is not True
        for name in required_diagnostics
        if name.startswith("record_")
    ):
        raise ValueError("M7 requires every locked diagnostic")

    p2b = config["p2b"]
    if set(p2b) != {
        "block_size",
        "group_size",
        "update_panel_size",
        "update_trailing_chunk_size",
        "first_update_backend",
        "quantization_backend",
        "quantization_batch_blocks",
    }:
        raise ValueError("M7 P2B fields mismatch")
    if (
        p2b["first_update_backend"] != "gram_cholesky"
        or p2b["quantization_backend"] != "streaming"
        or min(
            p2b[name]
            for name in (
                "block_size",
                "group_size",
                "update_panel_size",
                "quantization_batch_blocks",
            )
        )
        <= 0
    ):
        raise ValueError("M7 no longer matches frozen P2B")
    if p2b["update_trailing_chunk_size"] is not None and p2b[
        "update_trailing_chunk_size"
    ] <= 0:
        raise ValueError("invalid M7 trailing chunk size")

    gates = config["gates"]
    if set(gates) != {
        "require_source_m6_identity",
        "require_all_widths_and_tasks",
        "maximum_exact_probe_identity_error",
        "maximum_solver_relative_residual",
        "require_finite_diagnostics",
    }:
        raise ValueError("M7 gate fields mismatch")
    if (
        gates["require_source_m6_identity"] is not True
        or gates["require_all_widths_and_tasks"] is not True
        or gates["require_finite_diagnostics"] is not True
        or min(
            gates["maximum_exact_probe_identity_error"],
            gates["maximum_solver_relative_residual"],
        )
        < 0
    ):
        raise ValueError("invalid M7 gates")
    return config


def _verify_m6_artifact(config: dict, artifact_path: str | Path) -> dict:
    path = Path(artifact_path).resolve()
    source = config["source_m6"]
    if path.name != source["artifact_filename"]:
        raise ValueError("M7 source M6 artifact filename mismatch")
    if _sha256_file(path) != source["artifact_sha256"]:
        raise ValueError("M7 source M6 artifact SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("M7 source M6 artifact failed CRC")
        result_bytes = archive.read(source["result_member"])
    if _sha256_bytes(result_bytes) != source["result_sha256"]:
        raise ValueError("M7 source M6 result SHA-256 mismatch")
    payload = json.loads(result_bytes)
    if payload.get("status") != source["required_status"]:
        raise ValueError("M7 source M6 status mismatch")
    if payload.get("uses_test_set") is not False:
        raise ValueError("M7 refuses a test-used source artifact")
    by_width = {int(item["width"]): item for item in payload["width_results"]}
    for width in config["diagnostic_widths"]:
        if width not in by_width:
            raise ValueError("M7 diagnostic width is absent from M6")
        expected = float(config["ridge_by_width"][str(width)])
        if float(by_width[width]["selected_ridge_lambda"]) != expected:
            raise ValueError("M7 Ridge value does not match M6")
    return {
        "artifact_sha256": source["artifact_sha256"],
        "result_sha256": source["result_sha256"],
        "status": payload["status"],
        "verified_widths": list(config["diagnostic_widths"]),
        "m6_maximum_p2b_validation_aia_loss_pp": payload["summary"][
            "maximum_p2b_validation_aia_loss_pp"
        ],
    }


def _relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(candidate - reference)
    denominator = max(float(torch.linalg.vector_norm(reference).item()), 1.0)
    return float(numerator.item()) / denominator


def _encode_indices(encoder, features, indices, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, len(indices), batch_size):
        batch = features[indices[start : start + batch_size]]
        parts.append(encoder(batch))
    return torch.cat(parts)


def _prediction_and_margin(logits: torch.Tensor, class_ids: list[int]):
    columns = logits.argmax(1)
    mapping = torch.tensor(class_ids, dtype=torch.long)
    prediction = mapping[columns]
    top2 = torch.topk(logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return prediction, margin


def _evaluate_logits(
    *, backend, encoder, features, labels, indices, batch_size: int
) -> dict:
    logits_parts = []
    target_parts = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        codes = encoder(features[batch_indices])
        logits_parts.append(backend.predict_logits(codes).detach().cpu())
        target_parts.append(labels[batch_indices].to(torch.long).cpu())
        del codes
    logits = torch.cat(logits_parts)
    targets = torch.cat(target_parts)
    prediction, margin = _prediction_and_margin(logits, backend.class_ids)
    accuracy = 100.0 * float((prediction == targets).sum().item()) / len(targets)
    return {
        "logits": logits,
        "targets": targets,
        "prediction": prediction,
        "margin": margin,
        "accuracy_percent": accuracy,
    }


def _backend(config: dict, width: int, ridge: float, mode: str, device):
    common = {
        "dimension": width,
        "ridge_lambda": ridge,
        "device": device,
        "statistics_dtype": torch.float32,
        "solver_dtype": torch.float32,
    }
    if mode == "exact":
        return ExactGramBackend(**common)
    p2b = config["p2b"]
    return SquareRootBackend(
        storage_mode="float16" if mode == "fp16_square_root" else "int8",
        block_size=int(p2b["block_size"]),
        group_size=int(p2b["group_size"]),
        update_panel_size=int(p2b["update_panel_size"]),
        update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
        first_update_backend=p2b["first_update_backend"],
        quantization_backend=p2b["quantization_backend"],
        quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
        **common,
    )


def _run_exact(
    *, config, width, ridge, projection, probes, features, labels,
    training_parts, validation_parts, device
):
    backend = _backend(config, width, ridge, "exact", device)
    encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ projection
    )
    cumulative_action = ridge * probes.clone()
    references = []
    json_records = []
    for task_id, train_indices in enumerate(training_parts, start=1):
        codes = _encode_indices(
            encoder, features, train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        cumulative_action.add_(codes.T @ (codes @ probes))
        backend.update(codes, labels[train_indices])
        direct_action = backend.gram @ probes + ridge * probes
        probe_identity_error = _relative_error(direct_action, cumulative_action)
        seen = torch.cat(validation_parts[:task_id])
        evaluation = _evaluate_logits(
            backend=backend,
            encoder=encoder,
            features=features,
            labels=labels,
            indices=seen,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )
        references.append(
            {
                "weights": backend.weights.detach().cpu().clone(),
                "system_action": direct_action.detach().cpu().clone(),
                "logits": evaluation["logits"],
                "targets": evaluation["targets"],
                "prediction": evaluation["prediction"],
                "margin": evaluation["margin"],
                "accuracy_percent": evaluation["accuracy_percent"],
                "class_ids": list(backend.class_ids),
            }
        )
        json_records.append(
            {
                "task": task_id,
                "accuracy_percent": evaluation["accuracy_percent"],
                "solver_relative_residual": float(
                    backend.diagnostics["solver_relative_residual"]
                ),
                "exact_probe_identity_error": probe_identity_error,
                "mean_exact_margin": float(evaluation["margin"].mean().item()),
                "p05_exact_margin": float(
                    torch.quantile(evaluation["margin"], 0.05).item()
                ),
                "persistent_state_bytes_excluding_projection": backend.persistent_state_bytes(),
            }
        )
        del codes, direct_action, evaluation
    return references, json_records


def _compare_snapshot(evaluation: dict, backend, reference: dict) -> dict:
    if list(backend.class_ids) != reference["class_ids"]:
        raise AssertionError("M7 class-column alignment changed")
    logits = evaluation["logits"]
    logit_delta = logits - reference["logits"]
    sample_linf = logit_delta.abs().amax(dim=1)
    prediction = evaluation["prediction"]
    agreement = float((prediction == reference["prediction"]).float().mean().item())
    changed = prediction != reference["prediction"]
    changed_margin = (
        float(reference["margin"][changed].mean().item()) if bool(changed.any()) else 0.0
    )
    return {
        "relative_weight_error": _relative_error(
            backend.weights.detach().cpu(), reference["weights"]
        ),
        "relative_logit_error": _relative_error(logits, reference["logits"]),
        "mean_absolute_logit_error": float(logit_delta.abs().mean().item()),
        "maximum_absolute_logit_error": float(logit_delta.abs().max().item()),
        "prediction_agreement": agreement,
        "prediction_change_fraction": 1.0 - agreement,
        "margin_certified_fraction": float(
            (2.0 * sample_linf < reference["margin"]).float().mean().item()
        ),
        "mean_exact_margin_on_changed_predictions": changed_margin,
        "accuracy_gap_exact_minus_method_pp": reference["accuracy_percent"]
        - evaluation["accuracy_percent"],
    }


def _run_compressed(
    *, config, width, ridge, mode, projection, probes, references, features,
    labels, training_parts, validation_parts, device
):
    backend = _backend(config, width, ridge, mode, device)
    encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ projection
    )
    records = []
    for task_id, train_indices in enumerate(training_parts, start=1):
        codes = _encode_indices(
            encoder, features, train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        backend.update(codes, labels[train_indices])
        factor = backend.factor.reconstruct_upper(dtype=torch.float32)
        system_action = factor.T @ (factor @ probes)
        reference = references[task_id - 1]
        system_action_error = _relative_error(
            system_action.detach().cpu(), reference["system_action"]
        )
        seen = torch.cat(validation_parts[:task_id])
        evaluation = _evaluate_logits(
            backend=backend,
            encoder=encoder,
            features=features,
            labels=labels,
            indices=seen,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )
        comparison = _compare_snapshot(evaluation, backend, reference)
        records.append(
            {
                "task": task_id,
                "accuracy_percent": evaluation["accuracy_percent"],
                "solver_relative_residual": float(
                    backend.diagnostics["solver_relative_residual"]
                ),
                "relative_local_factor_quantization_error": float(
                    backend.diagnostics["relative_local_factor_error"]
                ),
                "relative_system_action_error": system_action_error,
                "persistent_state_bytes_excluding_projection": backend.persistent_state_bytes(),
                **comparison,
            }
        )
        del codes, factor, system_action, evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def _pearson(x: list[float], y: list[float]) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 0.0


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _summarize(width_results: list[dict], config: dict, source_verified: bool) -> dict:
    methods = ("exact", "fp16_square_root", "p2b_int8")
    maximum_exact_probe_error = max(
        record["exact_probe_identity_error"]
        for item in width_results
        for record in item["records"]["exact"]
    )
    maximum_solver_residual = max(
        record["solver_relative_residual"]
        for item in width_results
        for method in methods
        for record in item["records"][method]
    )
    per_width = {}
    for item in width_results:
        p2b = item["records"]["p2b_int8"]
        fp16 = item["records"]["fp16_square_root"]
        per_width[str(item["width"])] = {
            "validation_aia_percent": {
                method: sum(r["accuracy_percent"] for r in item["records"][method])
                / len(item["records"][method])
                for method in methods
            },
            "final_validation_accuracy_percent": {
                method: item["records"][method][-1]["accuracy_percent"]
                for method in methods
            },
            "p2b_final_relative_system_action_error": p2b[-1][
                "relative_system_action_error"
            ],
            "p2b_final_relative_weight_error": p2b[-1]["relative_weight_error"],
            "p2b_final_relative_logit_error": p2b[-1]["relative_logit_error"],
            "p2b_final_prediction_agreement": p2b[-1]["prediction_agreement"],
            "p2b_final_margin_certified_fraction": p2b[-1][
                "margin_certified_fraction"
            ],
            "p2b_system_error_accuracy_gap_pearson": _pearson(
                [r["relative_system_action_error"] for r in p2b],
                [r["accuracy_gap_exact_minus_method_pp"] for r in p2b],
            ),
            "maximum_fp16_relative_system_action_error": max(
                r["relative_system_action_error"] for r in fp16
            ),
        }
    thresholds = config["gates"]
    complete = [item["width"] for item in width_results] == config[
        "diagnostic_widths"
    ] and all(
        len(item["records"][method]) == config["num_tasks"]
        for item in width_results
        for method in methods
    )
    gates = {
        "source_m6_identity": source_verified,
        "all_widths_and_tasks_complete": complete,
        "exact_probe_identity": maximum_exact_probe_error
        <= thresholds["maximum_exact_probe_identity_error"],
        "solver_residual": maximum_solver_residual
        <= thresholds["maximum_solver_relative_residual"],
        "finite_diagnostics": _finite_tree(width_results),
    }
    return {
        "maximum_exact_probe_identity_error": maximum_exact_probe_error,
        "maximum_solver_relative_residual": maximum_solver_residual,
        "per_width": per_width,
        "gates": gates,
    }


def _write_csv(path: Path, width_results: list[dict]) -> None:
    fields = [
        "width", "task", "method", "accuracy_percent",
        "solver_relative_residual", "relative_local_factor_quantization_error",
        "relative_system_action_error", "relative_weight_error",
        "relative_logit_error", "prediction_agreement",
        "margin_certified_fraction", "accuracy_gap_exact_minus_method_pp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in width_results:
            for method, records in item["records"].items():
                for record in records:
                    writer.writerow(
                        {
                            field: (
                                item["width"] if field == "width"
                                else method if field == "method"
                                else record.get(field, "")
                            )
                            for field in fields
                        }
                    )


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M7 requires a clean source checkout")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M7 refuses a visible test.pt")
    source_identity = _verify_m6_artifact(config, args.m6_artifact)
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    if int(train["features"].shape[1]) != config["ranpac"]["feature_dimension"]:
        raise ValueError("M7 feature dimension mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(
        range(config["num_classes"])
    ):
        raise ValueError("M7 training labels do not match locked classes")

    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"], task_indices, config["seed"],
        config["outer_validation_fraction"],
    )
    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["ranpac"]["projection_seed"])
    )
    full_projection_cpu = torch.randn(
        int(config["ranpac"]["feature_dimension"]),
        int(config["ranpac"]["maximum_expand_dimension"]),
        generator=generator,
        dtype=torch.float32,
    )
    full_projection = full_projection_cpu.to(device)
    probe_generator = torch.Generator(device="cpu").manual_seed(
        int(config["diagnostics"]["system_probe_seed"])
    )
    full_probe_signs = torch.randint(
        0, 2,
        (
            int(config["ranpac"]["maximum_expand_dimension"]),
            int(config["diagnostics"]["system_probe_count"]),
        ),
        generator=probe_generator,
        dtype=torch.int8,
    )
    projection_hashes = {
        str(width): _tensor_sha256(full_projection_cpu[:, :width])
        for width in config["diagnostic_widths"]
    }
    probe_hashes = {
        str(width): _tensor_sha256(full_probe_signs[:width])
        for width in config["diagnostic_widths"]
    }
    del full_projection_cpu

    width_results = []
    numerical_failure = None
    try:
        for width in config["diagnostic_widths"]:
            print(f"M7 WIDTH START {width}", flush=True)
            projection = full_projection[:, :width].contiguous()
            probes = full_probe_signs[:width].to(
                device=device, dtype=torch.float32
            )
            probes.mul_(2).sub_(1).div_(width**0.5)
            ridge = float(config["ridge_by_width"][str(width)])

            references, exact_records = _run_exact(
                config=config, width=width, ridge=ridge, projection=projection,
                probes=probes, features=train["features"], labels=train["labels"],
                training_parts=training_parts, validation_parts=validation_parts,
                device=device,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            method_records = {"exact": exact_records}
            for mode in ("fp16_square_root", "p2b_int8"):
                print(f"M7 METHOD START width={width} method={mode}", flush=True)
                method_records[mode] = _run_compressed(
                    config=config, width=width, ridge=ridge, mode=mode,
                    projection=projection, probes=probes, references=references,
                    features=train["features"], labels=train["labels"],
                    training_parts=training_parts, validation_parts=validation_parts,
                    device=device,
                )
            width_results.append(
                {"width": width, "ridge_lambda": ridge, "records": method_records}
            )
            print(f"M7 WIDTH DONE {width}", flush=True)
            del references, exact_records, method_records, projection, probes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = _summarize(width_results, config, True)
    except (RuntimeError, AssertionError, ValueError, torch.linalg.LinAlgError) as error:
        numerical_failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        summary = {"gates": {"numerical_success": False}}

    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "config_sha256": _sha256_file(config_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "generic_backend_sha256": _sha256_file(
            ROOT / "methods/analytic_ridge/backends.py"
        ),
        "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
        "source_m6": source_identity,
        "class_order": class_order,
        "training_indices_sha256": _sequence_sha256(training_parts),
        "outer_validation_indices_sha256": _sequence_sha256(validation_parts),
        "projection_prefix_sha256": projection_hashes,
        "system_probe_prefix_sha256": probe_hashes,
    }
    passed = numerical_failure is None and all(summary["gates"].values())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M7_ERROR_TRAJECTORY_TRAIN_ONLY"
        if passed else "FAIL_M7_ERROR_TRAJECTORY_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "frontend": "RanPAC Phase-2 random-ReLU analytic head",
            "comparison": "post-M6 task-wise error diagnosis",
            "diagnostic_not_method_selection": True,
            "system_error_estimator": (
                f"fixed_{config['diagnostics']['system_probe_count']}_vector_"
                "Rademacher_system_action"
            ),
        },
        "provenance": provenance,
        "numerical_failure": numerical_failure,
        "width_results": width_results,
        "summary": {key: value for key, value in summary.items() if key != "gates"},
        "gates": summary["gates"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "m7_results.json"
    temporary = output_dir / "m7_results.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    _write_csv(output_dir / "error_trajectory.csv", width_results)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"],
                      "gates": payload["gates"]}, indent=2), flush=True)
    if not passed:
        raise RuntimeError("M7 task-wise error trajectory gate failed")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--m6-artifact", required=True)
    run_parser.add_argument("--feature-cache-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        run(args)


if __name__ == "__main__":
    main()
