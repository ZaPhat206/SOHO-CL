"""One-shot same-byte INT8 scale-refinement follow-up to M6 and M11."""

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

from methods.analytic_ridge import ExactGramBackend, persistent_tensor_bytes  # noqa: E402
from methods.analytic_ridge.refined_backend import (  # noqa: E402
    ScaleRefinedSquareRootBackend,
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
    "source_m11",
    "ranpac",
    "p2b",
    "scale_refinement",
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
        raise ValueError("M11b config keys/schema mismatch")
    if config["uses_test_set"] is not False:
        raise ValueError("M11b must remain train-only")
    if config["accuracy_based_selection"] is not False:
        raise ValueError("M11b cannot select scales using accuracy")
    if (
        config["seed"] != 2025
        or config["statistics_dtype"] != "float32"
        or config["solver_dtype"] != "float32"
        or config["diagnostic_widths"] != [10000, 20000]
        or config["num_classes"] != 100
        or config["num_tasks"] != 10
        or not 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("M11b no longer matches the locked M6/M11 stream")

    for name, required_status in (
        ("source_m6", "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY"),
        ("source_m11", "PASS_M11_ADAPTIVE_PRECISION_TRAIN_ONLY"),
    ):
        source = config[name]
        if set(source) != {
            "artifact_filename",
            "artifact_sha256",
            "result_member",
            "result_sha256",
            "required_status",
        } or source["required_status"] != required_status:
            raise ValueError(f"invalid M11b {name} lock")
        if len(source["artifact_sha256"]) != 64 or len(source["result_sha256"]) != 64:
            raise ValueError(f"invalid M11b {name} hashes")

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
        raise ValueError("M11b RanPAC fields mismatch")
    if (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or ranpac["feature_dimension"] != 768
        or ranpac["maximum_expand_dimension"] != 20000
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or ranpac["projection_seed"] != 2025
        or min(ranpac["encode_batch_size"], ranpac["evaluation_batch_size"]) <= 0
    ):
        raise ValueError("invalid locked M11b RanPAC semantics")

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
        raise ValueError("M11b P2B fields mismatch")
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
        raise ValueError("M11b no longer matches P2B")

    refinement = config["scale_refinement"]
    if set(refinement) != {
        "initializer",
        "update_rule",
        "iterations",
        "selection_signal",
        "persistent_payload",
    } or refinement != {
        "initializer": "max_abs_over_127",
        "update_rule": "least_squares_scale_then_nearest_int8_assignment",
        "iterations": 4,
        "selection_signal": "current_group_values_only_no_labels_or_accuracy",
        "persistent_payload": "identical_shapes_and_dtypes_to_p2b_int8",
    }:
        raise ValueError("M11b scale-refinement rule is not preregistered")

    gates = config["gates"]
    if set(gates) != {
        "require_source_identity_match",
        "maximum_exact_reproduction_accuracy_difference_pp",
        "require_same_persistent_bytes_as_p2b_every_task",
        "require_local_factor_error_not_worse_than_maxabs",
        "maximum_validation_aia_loss_pp",
        "require_aia_not_worse_than_p2b",
        "maximum_solver_relative_residual",
    }:
        raise ValueError("M11b gate fields mismatch")
    if (
        any(
            gates[name] is not True
            for name in (
                "require_source_identity_match",
                "require_same_persistent_bytes_as_p2b_every_task",
                "require_local_factor_error_not_worse_than_maxabs",
                "require_aia_not_worse_than_p2b",
            )
        )
        or min(
            float(gates[name])
            for name in (
                "maximum_exact_reproduction_accuracy_difference_pp",
                "maximum_validation_aia_loss_pp",
                "maximum_solver_relative_residual",
            )
        )
        < 0
    ):
        raise ValueError("invalid M11b gates")
    return config


def _load_locked_result(lock: dict, path: Path, label: str) -> dict:
    if path.name != lock["artifact_filename"]:
        raise ValueError(f"M11b {label} artifact filename mismatch")
    if _sha256_file(path) != lock["artifact_sha256"]:
        raise ValueError(f"M11b {label} artifact SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(lock["result_member"])
    if _sha256_bytes(payload) != lock["result_sha256"]:
        raise ValueError(f"M11b {label} result SHA-256 mismatch")
    result = json.loads(payload)
    if result.get("status") != lock["required_status"]:
        raise ValueError(f"M11b {label} status mismatch")
    return result


def _factor_bytes(backend: ScaleRefinedSquareRootBackend) -> int:
    return persistent_tensor_bytes(
        {
            name: tensor
            for name, tensor in backend.persistent_tensors().items()
            if name.startswith("factor.")
        }
    )


def _run_refined_width(
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
    backend = ScaleRefinedSquareRootBackend(
        **m5._common_backend(width, ridge, device),
        block_size=int(p2b["block_size"]),
        group_size=int(p2b["group_size"]),
        update_panel_size=int(p2b["update_panel_size"]),
        update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
        first_update_backend=p2b["first_update_backend"],
        quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
        scale_refinement_iterations=int(config["scale_refinement"]["iterations"]),
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
            backends={"refined": backend},
            features=train["features"],
            labels=train["labels"],
            indices=seen_validation,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )["refined"]
        tensors = {"projection": projection}
        tensors.update(backend.persistent_tensors())
        records.append(
            {
                "task": task_id + 1,
                "validation_accuracy_percent": accuracy,
                "total_persistent_bytes": persistent_tensor_bytes(tensors),
                "factor_persistent_bytes": _factor_bytes(backend),
                "refined_relative_factor_error": float(
                    backend.diagnostics["relative_local_factor_error"]
                ),
                "maxabs_same_input_relative_factor_error": float(
                    backend.diagnostics[
                        "maxabs_same_input_relative_factor_error"
                    ]
                ),
                "solver_relative_residual": float(
                    backend.diagnostics["solver_relative_residual"]
                ),
            }
        )
        print(
            f"M11B TASK width={width} task={task_id + 1}/{len(training_parts)} "
            f"acc={accuracy:.4f} factor_error="
            f"{records[-1]['refined_relative_factor_error']:.6e}",
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
        "refined_relative_factor_error",
        "maxabs_same_input_relative_factor_error",
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
    source_m6_path = Path(args.source_m6_artifact).resolve()
    source_m11_path = Path(args.source_m11_artifact).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    source_m6 = m11._load_source(config, source_m6_path)
    source_m11 = _load_locked_result(
        config["source_m11"], source_m11_path, "M11"
    )
    if source_m11.get("source_m6", {}).get("artifact_sha256") != config[
        "source_m6"
    ]["artifact_sha256"]:
        raise ValueError("M11b source artifacts do not share the same M6 identity")
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M11b requires a clean source checkout")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M11b refuses a visible test.pt")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("M11b feature-cache checkpoint SHA-256 mismatch")
    if int(train["features"].shape[1]) != config["ranpac"]["feature_dimension"]:
        raise ValueError("M11b feature dimension mismatch")

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
    source_provenance = source_m6["provenance"]
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
    m6_widths = {item["width"]: item for item in source_m6["width_results"]}
    m11_widths = {item["width"]: item for item in source_m11["width_results"]}
    results = []
    for width in config["diagnostic_widths"]:
        projection = full_projection[:, :width].contiguous()
        identity_checks[f"projection_{width}"] = (
            m4._tensor_content_sha256(projection)
            == source_provenance["projection_prefix_sha256"][str(width)]
        )
        source_width = m6_widths[width]
        adaptive_width = m11_widths[width]
        ridge = float(source_width["selected_ridge_lambda"])
        print(f"M11B WIDTH START {width} ridge={ridge}", flush=True)
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

        result = _run_refined_width(
            config=config,
            width=width,
            ridge=ridge,
            projection=projection,
            train=train,
            training_parts=training_parts,
            validation_parts=validation_parts,
            device=device,
        )
        result["source_references"] = {
            "exact_validation_aia_percent": source_width["validation_aia_percent"]["exact"],
            "p2b_validation_aia_percent": source_width["validation_aia_percent"]["p2b_int8"],
            "fp16_validation_aia_percent": source_width["validation_aia_percent"]["fp16_square_root"],
            "adaptive_validation_aia_percent": adaptive_width["validation_aia_percent"],
            "exact_final_validation_accuracy_percent": source_width[
                "final_validation_accuracy_percent"
            ]["exact"],
            "p2b_final_validation_accuracy_percent": source_width[
                "final_validation_accuracy_percent"
            ]["p2b_int8"],
            "adaptive_final_validation_accuracy_percent": adaptive_width[
                "final_validation_accuracy_percent"
            ],
            "exact_total_persistent_bytes": source_width[
                "final_total_persistent_bytes"
            ]["exact"],
            "p2b_total_persistent_bytes": source_width[
                "final_total_persistent_bytes"
            ]["p2b_int8"],
            "adaptive_total_persistent_bytes": adaptive_width[
                "final_total_persistent_bytes"
            ],
            "p2b_analytic_update_seconds": source_width[
                "analytic_update_seconds"
            ]["p2b_int8"],
            "adaptive_analytic_update_seconds": adaptive_width[
                "analytic_update_seconds"
            ],
        }
        result["expected_p2b_task_state_bytes"] = [
            record["state"]["p2b_int8"]["total_persistent_bytes"]
            for record in source_width["records"]
        ]
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
        reference = result["source_references"]
        result["refined_validation_aia_loss_pp"] = (
            reference["exact_validation_aia_percent"]
            - result["validation_aia_percent"]
        )
        result["refined_minus_p2b_validation_aia_pp"] = (
            result["validation_aia_percent"]
            - reference["p2b_validation_aia_percent"]
        )
        result["refined_minus_adaptive_validation_aia_pp"] = (
            result["validation_aia_percent"]
            - reference["adaptive_validation_aia_percent"]
        )
        result["refined_total_state_reduction_fraction"] = 1.0 - (
            result["final_total_persistent_bytes"]
            / reference["exact_total_persistent_bytes"]
        )
        result["update_ratio_to_p2b"] = (
            result["analytic_update_seconds"]
            / reference["p2b_analytic_update_seconds"]
        )
        result["update_ratio_to_adaptive"] = (
            result["analytic_update_seconds"]
            / reference["adaptive_analytic_update_seconds"]
        )
        results.append(result)
        print(
            f"M11B WIDTH DONE {width} refined_aia="
            f"{result['validation_aia_percent']:.4f} loss="
            f"{result['refined_validation_aia_loss_pp']:.4f}",
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
        "same_persistent_bytes_as_p2b_every_task": all(
            [record["total_persistent_bytes"] for record in item["records"]]
            == item["expected_p2b_task_state_bytes"]
            for item in results
        ),
        "local_factor_error_not_worse_than_maxabs": all(
            record["refined_relative_factor_error"]
            <= record["maxabs_same_input_relative_factor_error"] + 1e-12
            for item in results
            for record in item["records"]
        ),
        "accuracy_retention": all(
            item["refined_validation_aia_loss_pp"]
            <= gates_config["maximum_validation_aia_loss_pp"]
            for item in results
        ),
        "aia_not_worse_than_p2b": all(
            item["refined_minus_p2b_validation_aia_pp"] >= 0
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
        "status": "PASS_M11B_SCALE_REFINED_INT8_TRAIN_ONLY"
        if passed
        else "FAIL_M11B_SCALE_REFINED_INT8_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "frontend": "RanPAC Phase-2 random-ReLU analytic head",
            "petl_reproduced": False,
            "comparison": "M6/M11-locked same-byte INT8 scale refinement",
            "scale_rule_selected_from_accuracy": False,
        },
        "source_m6": config["source_m6"],
        "source_m11": config["source_m11"],
        "identity_checks": identity_checks,
        "width_results": results,
        "summary": {
            "maximum_refined_validation_aia_loss_pp": max(
                item["refined_validation_aia_loss_pp"] for item in results
            ),
            "minimum_refined_minus_p2b_validation_aia_pp": min(
                item["refined_minus_p2b_validation_aia_pp"] for item in results
            ),
            "minimum_refined_minus_adaptive_validation_aia_pp": min(
                item["refined_minus_adaptive_validation_aia_pp"] for item in results
            ),
            "maximum_update_ratio_to_p2b": max(
                item["update_ratio_to_p2b"] for item in results
            ),
            "maximum_update_ratio_to_adaptive": max(
                item["update_ratio_to_adaptive"] for item in results
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
            "refined_backend_sha256": _sha256_file(
                ROOT / "methods/analytic_ridge/refined_backend.py"
            ),
            "refined_storage_sha256": _sha256_file(
                ROOT / "methods/analytic_ridge/refined_upper.py"
            ),
            "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
            "source_m6_artifact_sha256": _sha256_file(source_m6_path),
            "source_m11_artifact_sha256": _sha256_file(source_m11_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "m11b_results.json"
    temporary = output_dir / "m11b_results.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    _write_csv(output_dir / "m11b_task_trajectory.csv", results)
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
        raise RuntimeError("M11b scale-refinement gate failed; do not tune or retry")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--feature-cache-dir", required=True)
    run_parser.add_argument("--source-m6-artifact", required=True)
    run_parser.add_argument("--source-m11-artifact", required=True)
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
