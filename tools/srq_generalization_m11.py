"""Train-only adaptive-precision follow-up to the M6 width-stress failure."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import (  # noqa: E402
    ExactGramBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from tools import srq_generalization_m4 as m4  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools import srq_generalization_m6 as m6  # noqa: E402
from tools.experiment_runner import (  # noqa: E402
    split,
    train_validation_indices,
    validate_cache,
)


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
    "source_m6",
    "ranpac",
    "p2b",
    "adaptive",
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


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M11 config keys/schema mismatch")
    if config["uses_test_set"] is not False:
        raise ValueError("M11 must remain train-only")
    if config["accuracy_based_selection"] is not False:
        raise ValueError("M11 cannot select precision using accuracy")
    if (
        config["seed"] != 2025
        or config["statistics_dtype"] != "float32"
        or config["solver_dtype"] != "float32"
        or config["diagnostic_widths"] != [10000, 20000]
        or config["num_classes"] != 100
        or config["num_tasks"] != 10
        or not 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("M11 no longer matches the locked M6 stress settings")

    source = config["source_m6"]
    if set(source) != {
        "artifact_filename",
        "artifact_sha256",
        "result_member",
        "result_sha256",
        "required_status",
    } or source["required_status"] != "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY":
        raise ValueError("invalid M11 source-M6 lock")
    if len(source["artifact_sha256"]) != 64 or len(source["result_sha256"]) != 64:
        raise ValueError("invalid M11 source hashes")

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
        raise ValueError("M11 RanPAC fields mismatch")
    if (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or ranpac["feature_dimension"] != 768
        or ranpac["maximum_expand_dimension"] != 20000
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or ranpac["projection_seed"] != 2025
        or min(ranpac["encode_batch_size"], ranpac["evaluation_batch_size"]) <= 0
    ):
        raise ValueError("invalid locked M11 RanPAC semantics")

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
        raise ValueError("M11 P2B fields mismatch")
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
        raise ValueError("M11 no longer matches P2B")

    adaptive = config["adaptive"]
    if set(adaptive) != {
        "budget_fraction_between_int8_and_fp16",
        "selection_rule",
        "selection_signal",
        "precision_mask_dtype",
        "tie_break",
    }:
        raise ValueError("M11 adaptive fields mismatch")
    if (
        adaptive["budget_fraction_between_int8_and_fp16"] != 0.25
        or adaptive["selection_rule"]
        != "largest_factor_mse_reduction_per_added_byte"
        or adaptive["selection_signal"]
        != "current_factor_values_only_no_labels_or_accuracy"
        or adaptive["precision_mask_dtype"] != "uint8"
        or adaptive["tie_break"] != "ascending_upper_block_index"
    ):
        raise ValueError("M11 adaptive policy is not the preregistered policy")

    gates = config["gates"]
    if set(gates) != {
        "require_source_identity_match",
        "maximum_exact_reproduction_accuracy_difference_pp",
        "maximum_adaptive_validation_aia_loss_pp",
        "require_adaptive_aia_not_worse_than_p2b",
        "minimum_adaptive_total_state_reduction_fraction",
        "maximum_solver_relative_residual",
    }:
        raise ValueError("M11 gate fields mismatch")
    if (
        gates["require_source_identity_match"] is not True
        or gates["require_adaptive_aia_not_worse_than_p2b"] is not True
        or min(
            float(gates[name])
            for name in (
                "maximum_adaptive_validation_aia_loss_pp",
                "maximum_exact_reproduction_accuracy_difference_pp",
                "minimum_adaptive_total_state_reduction_fraction",
                "maximum_solver_relative_residual",
            )
        )
        < 0
        or not 0
        <= float(gates["minimum_adaptive_total_state_reduction_fraction"])
        <= 1
    ):
        raise ValueError("invalid M11 gates")
    return config


def _load_source(config: dict, artifact_path: Path) -> dict:
    source = config["source_m6"]
    if artifact_path.name != source["artifact_filename"]:
        raise ValueError("M11 source artifact filename mismatch")
    if _sha256_file(artifact_path) != source["artifact_sha256"]:
        raise ValueError("M11 source artifact SHA-256 mismatch")
    with zipfile.ZipFile(artifact_path) as archive:
        payload = archive.read(source["result_member"])
    if _sha256_bytes(payload) != source["result_sha256"]:
        raise ValueError("M11 source result SHA-256 mismatch")
    result = json.loads(payload)
    if result.get("status") != source["required_status"]:
        raise ValueError("M11 source status mismatch")
    source_gates = result.get("gates", {})
    if source_gates.get("p2b_accuracy_retention") is not False or any(
        value is not True
        for name, value in source_gates.items()
        if name != "p2b_accuracy_retention"
    ):
        raise ValueError("M11 source is not the isolated M6 P2B retention failure")
    return result


def _factor_bytes(backend: SquareRootBackend) -> int:
    return persistent_tensor_bytes(
        {
            name: tensor
            for name, tensor in backend.persistent_tensors().items()
            if name.startswith("factor.")
        }
    )


def _run_adaptive_width(
    *,
    config: dict,
    width: int,
    ridge: float,
    projection: torch.Tensor,
    train: dict,
    training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    p2b = config["p2b"]
    adaptive = config["adaptive"]
    backend = SquareRootBackend(
        **m5._common_backend(width, ridge, device),
        storage_mode="adaptive_int8_fp16",
        adaptive_budget_fraction=float(
            adaptive["budget_fraction_between_int8_and_fp16"]
        ),
        block_size=int(p2b["block_size"]),
        group_size=int(p2b["group_size"]),
        update_panel_size=int(p2b["update_panel_size"]),
        update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
        first_update_backend=p2b["first_update_backend"],
        quantization_backend=p2b["quantization_backend"],
        quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
    )
    encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ projection
    )
    records = []
    update_seconds = 0.0
    encoding_seconds = 0.0
    for task_id, train_indices in enumerate(training_parts):
        m5._sync(device)
        started = time.perf_counter()
        codes = m5._encode_indices(
            encoder,
            train["features"],
            train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        m5._sync(device)
        encoding_seconds += time.perf_counter() - started
        m5._sync(device)
        started = time.perf_counter()
        backend.update(codes, train["labels"][train_indices])
        m5._sync(device)
        update_seconds += time.perf_counter() - started
        seen_validation = torch.cat(validation_parts[: task_id + 1])
        accuracy = m5._evaluate(
            encoder=encoder,
            backends={"adaptive": backend},
            features=train["features"],
            labels=train["labels"],
            indices=seen_validation,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )["adaptive"]
        tensors = {"projection": projection}
        tensors.update(backend.persistent_tensors())
        factor_bytes = _factor_bytes(backend)
        if factor_bytes != int(backend.diagnostics["factor_persistent_bytes"]):
            raise AssertionError("M11 adaptive factor accounting mismatch")
        mask = backend.factor.precision_mask.detach().cpu().contiguous()
        records.append(
            {
                "task": task_id + 1,
                "validation_accuracy_percent": accuracy,
                "total_persistent_bytes": persistent_tensor_bytes(tensors),
                "factor_persistent_bytes": factor_bytes,
                "factor_budget_ceiling_bytes": int(
                    backend.diagnostics["factor_budget_ceiling_bytes"]
                ),
                "selected_fp16_blocks": int(
                    backend.diagnostics["selected_fp16_blocks"]
                ),
                "total_blocks": int(backend.diagnostics["total_blocks"]),
                "selected_fp16_values": int(
                    backend.diagnostics["selected_fp16_values"]
                ),
                "total_strict_upper_values": int(
                    backend.diagnostics["total_strict_upper_values"]
                ),
                "precision_mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                "adaptive_relative_factor_error": float(
                    backend.diagnostics["relative_local_factor_error"]
                ),
                "all_int8_relative_factor_error": float(
                    backend.diagnostics["all_int8_relative_factor_error"]
                ),
                "all_fp16_relative_factor_error": float(
                    backend.diagnostics["all_fp16_relative_factor_error"]
                ),
                "solver_relative_residual": float(
                    backend.diagnostics["solver_relative_residual"]
                ),
            }
        )
        print(
            f"M11 TASK width={width} task={task_id + 1}/{len(training_parts)} "
            f"acc={accuracy:.4f} fp16_blocks="
            f"{records[-1]['selected_fp16_blocks']}/{records[-1]['total_blocks']}",
            flush=True,
        )
        del codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "width": width,
        "selected_ridge_lambda": ridge,
        "records": records,
        "validation_aia_percent": sum(
            record["validation_accuracy_percent"] for record in records
        )
        / len(records),
        "final_validation_accuracy_percent": records[-1][
            "validation_accuracy_percent"
        ],
        "final_total_persistent_bytes": records[-1]["total_persistent_bytes"],
        "final_factor_persistent_bytes": records[-1]["factor_persistent_bytes"],
        "analytic_update_seconds": update_seconds,
        "representation_encoding_seconds": encoding_seconds,
    }


def _write_csv(path: Path, results: list[dict]) -> None:
    fields = [
        "width",
        "task",
        "validation_accuracy_percent",
        "total_persistent_bytes",
        "factor_persistent_bytes",
        "factor_budget_ceiling_bytes",
        "selected_fp16_blocks",
        "total_blocks",
        "adaptive_relative_factor_error",
        "all_int8_relative_factor_error",
        "solver_relative_residual",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for record in result["records"]:
                writer.writerow(
                    {"width": result["width"], **{key: record[key] for key in fields[1:]}}
                )


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    source_artifact = Path(args.source_m6_artifact).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    source = _load_source(config, source_artifact)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M11 requires a clean source checkout")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M11 refuses a visible test.pt")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("M11 feature-cache checkpoint SHA-256 mismatch")
    if int(train["features"].shape[1]) != config["ranpac"]["feature_dimension"]:
        raise ValueError("M11 feature dimension mismatch")

    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"],
        task_indices,
        config["seed"],
        config["outer_validation_fraction"],
    )
    source_provenance = source["provenance"]
    identity_checks = {
        "checkpoint": metadata.get("checkpoint_sha256")
        == config["checkpoint_sha256"],
        "class_ids": sorted(map(int, torch.unique(train["labels"]).tolist()))
        == list(range(config["num_classes"])),
        "class_order": class_order == source_provenance["class_order"],
        "training_indices": m6._sequence_sha256(training_parts)
        == source_provenance["training_indices_sha256"],
        "validation_indices": m6._sequence_sha256(validation_parts)
        == source_provenance["outer_validation_indices_sha256"],
    }

    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["ranpac"]["projection_seed"])
    )
    full_projection = torch.randn(
        config["ranpac"]["feature_dimension"],
        config["ranpac"]["maximum_expand_dimension"],
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    identity_checks["full_projection"] = (
        m4._tensor_content_sha256(full_projection)
        == source_provenance["full_projection_sha256"]
    )
    source_widths = {item["width"]: item for item in source["width_results"]}
    results = []
    for width in config["diagnostic_widths"]:
        projection = full_projection[:, :width].contiguous()
        identity_checks[f"projection_{width}"] = (
            m4._tensor_content_sha256(projection)
            == source_provenance["projection_prefix_sha256"][str(width)]
        )
        source_width = source_widths[width]
        ridge = float(source_width["selected_ridge_lambda"])
        print(f"M11 WIDTH START {width} ridge={ridge}", flush=True)
        exact_group = m5._run_group(
            encoder=lambda values, current=projection: torch.relu(
                values.to(device=device, dtype=torch.float32) @ current
            ),
            backends={
                "exact": ExactGramBackend(
                    **m5._common_backend(width, ridge, device)
                )
            },
            extra_tensors={"projection": projection},
            expected_total_bytes={
                "exact": source_width["final_total_persistent_bytes"]["exact"]
            },
            features=train["features"],
            labels=train["labels"],
            training_parts=training_parts,
            validation_parts=validation_parts,
            encode_batch_size=int(config["ranpac"]["encode_batch_size"]),
            evaluation_batch_size=int(config["ranpac"]["evaluation_batch_size"]),
            device=device,
        )
        exact_records = exact_group["records"]
        reproduced_exact_aia = sum(
            record["accuracy_percent"]["exact"] for record in exact_records
        ) / len(exact_records)
        reproduced_exact_final = exact_records[-1]["accuracy_percent"]["exact"]
        del exact_group, exact_records
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = _run_adaptive_width(
            config=config,
            width=width,
            ridge=ridge,
            projection=projection,
            train=train,
            training_parts=training_parts,
            validation_parts=validation_parts,
            device=device,
        )
        result["source_m6_reference"] = {
            "exact_validation_aia_percent": source_width["validation_aia_percent"]["exact"],
            "p2b_validation_aia_percent": source_width["validation_aia_percent"]["p2b_int8"],
            "fp16_validation_aia_percent": source_width["validation_aia_percent"]["fp16_square_root"],
            "exact_final_validation_accuracy_percent": source_width[
                "final_validation_accuracy_percent"
            ]["exact"],
            "p2b_final_validation_accuracy_percent": source_width[
                "final_validation_accuracy_percent"
            ]["p2b_int8"],
            "fp16_final_validation_accuracy_percent": source_width[
                "final_validation_accuracy_percent"
            ]["fp16_square_root"],
            "exact_total_persistent_bytes": source_width[
                "final_total_persistent_bytes"
            ]["exact"],
            "p2b_total_persistent_bytes": source_width[
                "final_total_persistent_bytes"
            ]["p2b_int8"],
            "fp16_total_persistent_bytes": source_width[
                "final_total_persistent_bytes"
            ]["fp16_square_root"],
        }
        result["exact_reproduction"] = {
            "validation_aia_percent": reproduced_exact_aia,
            "final_validation_accuracy_percent": reproduced_exact_final,
            "maximum_accuracy_difference_pp": max(
                abs(
                    reproduced_exact_aia
                    - source_width["validation_aia_percent"]["exact"]
                ),
                abs(
                    reproduced_exact_final
                    - source_width["final_validation_accuracy_percent"]["exact"]
                ),
            ),
        }
        reference = result["source_m6_reference"]
        result["adaptive_validation_aia_loss_pp"] = (
            reference["exact_validation_aia_percent"]
            - result["validation_aia_percent"]
        )
        result["adaptive_minus_p2b_validation_aia_pp"] = (
            result["validation_aia_percent"]
            - reference["p2b_validation_aia_percent"]
        )
        result["adaptive_total_state_reduction_fraction"] = 1.0 - (
            result["final_total_persistent_bytes"]
            / reference["exact_total_persistent_bytes"]
        )
        results.append(result)
        print(
            f"M11 WIDTH DONE {width} adaptive_aia="
            f"{result['validation_aia_percent']:.4f} loss="
            f"{result['adaptive_validation_aia_loss_pp']:.4f}",
            flush=True,
        )
        del projection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gates_config = config["gates"]
    gates = {
        "source_identity_match": all(identity_checks.values()),
        "all_widths_complete": [item["width"] for item in results]
        == config["diagnostic_widths"],
        "exact_accuracy_reproduced": max(
            item["exact_reproduction"]["maximum_accuracy_difference_pp"]
            for item in results
        )
        <= gates_config["maximum_exact_reproduction_accuracy_difference_pp"],
        "factor_budget_honored": all(
            record["factor_persistent_bytes"]
            <= record["factor_budget_ceiling_bytes"]
            for item in results
            for record in item["records"]
        ),
        "factor_error_not_worse_than_int8": all(
            record["adaptive_relative_factor_error"]
            <= record["all_int8_relative_factor_error"] + 1e-12
            for item in results
            for record in item["records"]
        ),
        "state_between_p2b_and_fp16": all(
            item["source_m6_reference"]["p2b_total_persistent_bytes"]
            < item["final_total_persistent_bytes"]
            < item["source_m6_reference"]["fp16_total_persistent_bytes"]
            for item in results
        ),
        "adaptive_accuracy_retention": all(
            item["adaptive_validation_aia_loss_pp"]
            <= gates_config["maximum_adaptive_validation_aia_loss_pp"]
            for item in results
        ),
        "adaptive_not_worse_than_p2b": all(
            item["adaptive_minus_p2b_validation_aia_pp"] >= 0
            for item in results
        ),
        "adaptive_state_reduction": all(
            item["adaptive_total_state_reduction_fraction"]
            >= gates_config["minimum_adaptive_total_state_reduction_fraction"]
            for item in results
        ),
        "solver_residual": max(
            record["solver_relative_residual"]
            for item in results
            for record in item["records"]
        )
        <= gates_config["maximum_solver_relative_residual"],
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M11_ADAPTIVE_PRECISION_TRAIN_ONLY"
        if passed
        else "FAIL_M11_ADAPTIVE_PRECISION_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "frontend": "RanPAC Phase-2 random-ReLU analytic head",
            "petl_reproduced": False,
            "comparison": "M6-locked 10k/20k adaptive precision follow-up",
            "adaptive_policy_selected_from_accuracy": False,
        },
        "source_m6": config["source_m6"],
        "identity_checks": identity_checks,
        "width_results": results,
        "summary": {
            "maximum_adaptive_validation_aia_loss_pp": max(
                item["adaptive_validation_aia_loss_pp"] for item in results
            ),
            "minimum_adaptive_minus_p2b_validation_aia_pp": min(
                item["adaptive_minus_p2b_validation_aia_pp"] for item in results
            ),
            "minimum_adaptive_total_state_reduction_fraction": min(
                item["adaptive_total_state_reduction_fraction"] for item in results
            ),
            "maximum_solver_relative_residual": max(
                record["solver_relative_residual"]
                for item in results
                for record in item["records"]
            ),
        },
        "gates": gates,
        "provenance": {
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
            "adaptive_storage_sha256": _sha256_file(
                ROOT / "methods/analytic_ridge/adaptive_upper.py"
            ),
            "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
            "source_artifact_sha256": _sha256_file(source_artifact),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "m11_results.json"
    temporary = output_dir / "m11_results.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    _write_csv(output_dir / "m11_task_trajectory.csv", results)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary": payload["summary"],
                "gates": payload["gates"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise RuntimeError("M11 adaptive precision gate failed; do not tune after output")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--feature-cache-dir", required=True)
    run_parser.add_argument("--source-m6-artifact", required=True)
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
