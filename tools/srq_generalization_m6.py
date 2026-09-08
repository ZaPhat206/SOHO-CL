"""Train-only width-scaling audit for the reusable SRQ Ridge backend."""

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

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import ExactGramBackend, SquareRootBackend  # noqa: E402
from tools import srq_generalization_m4 as m4  # noqa: E402
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
    "widths",
    "ranpac",
    "ridge_selection",
    "p2b",
    "gates",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(parts: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.to(torch.int64).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M6 config keys/schema mismatch")
    if config["uses_test_set"] is not False:
        raise ValueError("M6 must remain train-only")
    if config["accuracy_based_selection"] is not False:
        raise ValueError("M6 cannot select widths using accuracy")
    if config["seed"] != 2025:
        raise ValueError("M6 seed must remain 2025")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("M6 locks FP32 statistics and solves")
    if (
        config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or not 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("invalid M6 class/task split")

    widths = config["widths"]
    if (
        not widths
        or any(not isinstance(width, int) or width <= 0 for width in widths)
        or widths != sorted(set(widths))
    ):
        raise ValueError("M6 widths must be unique increasing positive integers")

    ranpac = config["ranpac"]
    required_ranpac = {
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
    }
    if set(ranpac) != required_ranpac:
        raise ValueError("M6 RanPAC fields mismatch")
    if (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or ranpac["projection_seed"] != 2025
        or max(widths) != ranpac["maximum_expand_dimension"]
        or min(
            ranpac[name]
            for name in (
                "feature_dimension",
                "maximum_expand_dimension",
                "encode_batch_size",
                "evaluation_batch_size",
            )
        )
        <= 0
        or len(ranpac["upstream_commit"]) != 40
        or len(ranpac["upstream_ranpac_py_sha256"]) != 64
    ):
        raise ValueError("invalid locked M6 RanPAC semantics")

    selection = config["ridge_selection"]
    if set(selection) != {
        "policy",
        "candidate_lambdas",
        "fit_samples_per_class",
        "validation_samples_per_class",
        "metric",
        "tie_break",
    }:
        raise ValueError("M6 Ridge-selection fields mismatch")
    candidates = list(map(float, selection["candidate_lambdas"]))
    if (
        selection["policy"]
        != "per_width_train_only_calibration_shared_across_backends"
        or selection["metric"] != "mean_squared_error"
        or selection["tie_break"] != "smallest_lambda"
        or not candidates
        or candidates != sorted(set(candidates))
        or min(candidates) <= 0
        or min(
            selection[name]
            for name in ("fit_samples_per_class", "validation_samples_per_class")
        )
        <= 0
    ):
        raise ValueError("invalid M6 Ridge selection")

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
        raise ValueError("M6 P2B fields mismatch")
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
        raise ValueError("M6 no longer matches frozen P2B")
    if p2b["update_trailing_chunk_size"] is not None and p2b[
        "update_trailing_chunk_size"
    ] <= 0:
        raise ValueError("invalid M6 trailing chunk size")

    gates = config["gates"]
    if set(gates) != {
        "require_all_widths_complete",
        "require_projection_prefix_nesting",
        "require_strict_state_growth",
        "require_p2b_smaller_than_exact_at_every_width",
        "maximum_fp16_validation_aia_loss_pp_per_width",
        "maximum_p2b_validation_aia_loss_pp_per_width",
        "maximum_solver_relative_residual",
    }:
        raise ValueError("M6 gate fields mismatch")
    if any(
        gates[name] is not True
        for name in (
            "require_all_widths_complete",
            "require_projection_prefix_nesting",
            "require_strict_state_growth",
            "require_p2b_smaller_than_exact_at_every_width",
        )
    ) or min(
        gates[name]
        for name in (
            "maximum_fp16_validation_aia_loss_pp_per_width",
            "maximum_p2b_validation_aia_loss_pp_per_width",
            "maximum_solver_relative_residual",
        )
    ) < 0:
        raise ValueError("invalid M6 gates")
    return config


def _state_lock(config: dict, width: int) -> dict:
    classes = int(config["num_classes"])
    feature_dimension = int(config["ranpac"]["feature_dimension"])
    p2b = config["p2b"]
    projection_bytes = 4 * feature_dimension * width
    exact_quadratic = 4 * width * width
    fp16_factor = m5._compressed_upper_bytes(
        width,
        block_size=int(p2b["block_size"]),
        group_size=int(p2b["group_size"]),
        mode="float16",
    )
    p2b_factor = m5._compressed_upper_bytes(
        width,
        block_size=int(p2b["block_size"]),
        group_size=int(p2b["group_size"]),
        mode="int8",
    )
    common_linear = 8 * width * classes + 4 * classes
    return {
        "width": width,
        "projection_bytes": projection_bytes,
        "quadratic_or_factor_bytes": {
            "exact": exact_quadratic,
            "fp16_square_root": fp16_factor,
            "p2b_int8": p2b_factor,
        },
        "total_persistent_bytes": {
            "exact": projection_bytes + exact_quadratic + common_linear,
            "fp16_square_root": projection_bytes + fp16_factor + common_linear,
            "p2b_int8": projection_bytes + p2b_factor + common_linear,
        },
    }


def _backend_group(config: dict, width: int, ridge: float, device: torch.device) -> dict:
    common = m5._common_backend(width, ridge, device)
    p2b = config["p2b"]
    shared = {
        "block_size": int(p2b["block_size"]),
        "group_size": int(p2b["group_size"]),
        "update_panel_size": int(p2b["update_panel_size"]),
        "update_trailing_chunk_size": p2b["update_trailing_chunk_size"],
        "first_update_backend": p2b["first_update_backend"],
        "quantization_backend": p2b["quantization_backend"],
        "quantization_batch_blocks": int(p2b["quantization_batch_blocks"]),
    }
    return {
        "exact": ExactGramBackend(**common),
        "fp16_square_root": SquareRootBackend(
            storage_mode="float16", **shared, **common
        ),
        "p2b_int8": SquareRootBackend(storage_mode="int8", **shared, **common),
    }


def _width_summary(groups: dict[str, dict], ridge_result: dict, state_lock: dict) -> dict:
    names = ("exact", "fp16_square_root", "p2b_int8")
    tasks = len(next(iter(groups.values()))["records"])
    if any(len(group["records"]) != tasks for group in groups.values()):
        raise AssertionError("method task-record counts differ")
    records = []
    for task_index in range(tasks):
        records.append(
            {
                "task": task_index + 1,
                "accuracy_percent": {
                    name: groups[name]["records"][task_index]["accuracy_percent"][name]
                    for name in names
                },
                "state": {
                    name: groups[name]["records"][task_index]["state"][name]
                    for name in names
                },
            }
        )
    aia = {
        name: sum(record["accuracy_percent"][name] for record in records)
        / len(records)
        for name in names
    }
    final_accuracy = {
        name: records[-1]["accuracy_percent"][name] for name in names
    }
    final_state = {
        name: records[-1]["state"][name]["total_persistent_bytes"] for name in names
    }
    maximum_residual = max(
        record["state"][name]["solver_relative_residual"]
        for record in records
        for name in names
    )
    return {
        "width": state_lock["width"],
        "records": records,
        "selected_ridge_lambda": ridge_result["selected_ridge_lambda"],
        "validation_aia_percent": aia,
        "final_validation_accuracy_percent": final_accuracy,
        "final_total_persistent_bytes": final_state,
        "quadratic_or_factor_bytes": state_lock["quadratic_or_factor_bytes"],
        "projection_bytes": state_lock["projection_bytes"],
        "analytic_update_seconds": {
            name: groups[name]["analytic_update_seconds"][name] for name in names
        },
        "representation_encoding_seconds": {
            name: groups[name]["encoding_seconds"] for name in names
        },
        "fp16_validation_aia_loss_pp": aia["exact"] - aia["fp16_square_root"],
        "p2b_validation_aia_loss_pp": aia["exact"] - aia["p2b_int8"],
        "p2b_total_state_reduction_fraction": 1.0
        - final_state["p2b_int8"] / final_state["exact"],
        "maximum_solver_relative_residual": maximum_residual,
    }


def _log_log_slope(widths: list[int], values: list[int]) -> float:
    x = [math.log(float(value)) for value in widths]
    y = [math.log(float(value)) for value in values]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = sum((a - x_mean) ** 2 for a in x)
    return numerator / denominator


def _summarize(width_results: list[dict], config: dict) -> dict:
    widths = [item["width"] for item in width_results]
    states = {
        name: [item["final_total_persistent_bytes"][name] for item in width_results]
        for name in ("exact", "fp16_square_root", "p2b_int8")
    }
    strict_growth = {
        name: all(a < b for a, b in zip(values, values[1:]))
        for name, values in states.items()
    }
    maximum_residual = max(
        item["maximum_solver_relative_residual"] for item in width_results
    )
    maximum_fp16_loss = max(
        item["fp16_validation_aia_loss_pp"] for item in width_results
    )
    maximum_p2b_loss = max(
        item["p2b_validation_aia_loss_pp"] for item in width_results
    )
    p2b_smaller = all(
        item["final_total_persistent_bytes"]["p2b_int8"]
        < item["final_total_persistent_bytes"]["exact"]
        for item in width_results
    )
    thresholds = config["gates"]
    gates = {
        "all_widths_complete": widths == config["widths"],
        "projection_prefix_nesting": True,
        "strict_state_growth": all(strict_growth.values()),
        "p2b_smaller_than_exact_at_every_width": p2b_smaller,
        "fp16_accuracy_retention": maximum_fp16_loss
        <= thresholds["maximum_fp16_validation_aia_loss_pp_per_width"],
        "p2b_accuracy_retention": maximum_p2b_loss
        <= thresholds["maximum_p2b_validation_aia_loss_pp_per_width"],
        "solver_residual": maximum_residual
        <= thresholds["maximum_solver_relative_residual"],
    }
    return {
        "widths": widths,
        "maximum_fp16_validation_aia_loss_pp": maximum_fp16_loss,
        "maximum_p2b_validation_aia_loss_pp": maximum_p2b_loss,
        "maximum_solver_relative_residual": maximum_residual,
        "strict_state_growth_by_method": strict_growth,
        "total_state_log_log_slope": {
            name: _log_log_slope(widths, values) for name, values in states.items()
        },
        "gates": gates,
    }


def _write_csv(path: Path, width_results: list[dict]) -> None:
    fields = [
        "width",
        "method",
        "selected_ridge_lambda",
        "validation_aia_percent",
        "final_validation_accuracy_percent",
        "final_total_persistent_bytes",
        "quadratic_or_factor_bytes",
        "projection_bytes",
        "analytic_update_seconds",
        "representation_encoding_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in width_results:
            for method in ("exact", "fp16_square_root", "p2b_int8"):
                writer.writerow(
                    {
                        "width": item["width"],
                        "method": method,
                        "selected_ridge_lambda": item["selected_ridge_lambda"],
                        "validation_aia_percent": item["validation_aia_percent"][method],
                        "final_validation_accuracy_percent": item[
                            "final_validation_accuracy_percent"
                        ][method],
                        "final_total_persistent_bytes": item[
                            "final_total_persistent_bytes"
                        ][method],
                        "quadratic_or_factor_bytes": item[
                            "quadratic_or_factor_bytes"
                        ][method],
                        "projection_bytes": item["projection_bytes"],
                        "analytic_update_seconds": item["analytic_update_seconds"][
                            method
                        ],
                        "representation_encoding_seconds": item[
                            "representation_encoding_seconds"
                        ][method],
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
        raise RuntimeError("M6 requires a clean source checkout")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M6 refuses a visible test.pt")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    feature_dimension = int(config["ranpac"]["feature_dimension"])
    if int(train["features"].shape[1]) != feature_dimension:
        raise ValueError("M6 feature dimension mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(
        range(config["num_classes"])
    ):
        raise ValueError("M6 training labels do not match locked classes")

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
    selection = config["ridge_selection"]
    fit_indices, calibration_validation_indices = m4._calibration_indices(
        train["labels"],
        training_parts,
        seed=config["seed"],
        fit_per_class=int(selection["fit_samples_per_class"]),
        validation_per_class=int(selection["validation_samples_per_class"]),
    )

    device = torch.device(args.device)
    maximum_width = int(config["ranpac"]["maximum_expand_dimension"])
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["ranpac"]["projection_seed"])
    )
    full_projection = torch.randn(
        feature_dimension,
        maximum_width,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    encode_batch_size = int(config["ranpac"]["encode_batch_size"])
    evaluation_batch_size = int(config["ranpac"]["evaluation_batch_size"])
    candidates = list(map(float, selection["candidate_lambdas"]))

    width_results = []
    ridge_results = {}
    projection_hashes = {}
    numerical_failure = None
    try:
        for width in config["widths"]:
            projection = full_projection[:, :width].contiguous()
            if not torch.equal(projection, full_projection[:, :width]):
                raise AssertionError("projection prefix nesting failed")
            projection_hashes[str(width)] = m4._tensor_content_sha256(projection)
            encoder = lambda values, current=projection: torch.relu(
                values.to(device=device, dtype=torch.float32) @ current
            )
            ridge_result = m5._select_ridge(
                name=f"random_relu_width_{width}",
                encoder=encoder,
                features=train["features"],
                labels=train["labels"],
                fit_indices=fit_indices,
                validation_indices=calibration_validation_indices,
                candidate_lambdas=candidates,
                num_classes=int(config["num_classes"]),
                batch_size=encode_batch_size,
            )
            ridge_results[str(width)] = ridge_result
            ridge = float(ridge_result["selected_ridge_lambda"])
            lock = _state_lock(config, width)
            print(f"WIDTH START {width} RIDGE {ridge}", flush=True)
            backends = _backend_group(config, width, ridge, device)
            method_groups = {}
            for name in ("exact", "fp16_square_root", "p2b_int8"):
                print(f"METHOD START width={width} method={name}", flush=True)
                backend = backends.pop(name)
                method_groups[name] = m5._run_group(
                    encoder=encoder,
                    backends={name: backend},
                    extra_tensors={"projection": projection},
                    expected_total_bytes={
                        name: lock["total_persistent_bytes"][name]
                    },
                    features=train["features"],
                    labels=train["labels"],
                    training_parts=training_parts,
                    validation_parts=validation_parts,
                    encode_batch_size=encode_batch_size,
                    evaluation_batch_size=evaluation_batch_size,
                    device=device,
                )
                del backend
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            item = _width_summary(method_groups, ridge_result, lock)
            width_results.append(item)
            print(
                f"WIDTH DONE {width} exact={item['validation_aia_percent']['exact']:.4f} "
                f"fp16={item['validation_aia_percent']['fp16_square_root']:.4f} "
                f"p2b={item['validation_aia_percent']['p2b_int8']:.4f}",
                flush=True,
            )
            del backends, method_groups, projection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = _summarize(width_results, config)
    except (RuntimeError, AssertionError, torch.linalg.LinAlgError) as error:
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
        "m5_runner_sha256": _sha256_file(ROOT / "tools/srq_generalization_m5.py"),
        "generic_backend_sha256": _sha256_file(
            ROOT / "methods/analytic_ridge/backends.py"
        ),
        "ranpac_frontend_sha256": _sha256_file(
            ROOT / "methods/frontends/ranpac.py"
        ),
        "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
        "full_projection_sha256": m4._tensor_content_sha256(full_projection),
        "projection_prefix_sha256": projection_hashes,
        "class_order": class_order,
        "training_indices_sha256": _sequence_sha256(training_parts),
        "outer_validation_indices_sha256": _sequence_sha256(validation_parts),
        "calibration_fit_indices_sha256": _sequence_sha256([fit_indices]),
        "calibration_validation_indices_sha256": _sequence_sha256(
            [calibration_validation_indices]
        ),
        "upstream_repository": config["ranpac"]["upstream_repository"],
        "upstream_commit": config["ranpac"]["upstream_commit"],
        "upstream_ranpac_py_sha256": config["ranpac"][
            "upstream_ranpac_py_sha256"
        ],
    }
    passed = numerical_failure is None and all(summary["gates"].values())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M6_WIDTH_SWEEP_TRAIN_ONLY"
        if passed
        else "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "frontend": "RanPAC Phase-2 random-ReLU analytic head",
            "petl_reproduced": False,
            "comparison": "train-only width scaling",
            "widths_are_selection_candidates": False,
        },
        "ridge_selection": ridge_results,
        "numerical_failure": numerical_failure,
        "provenance": provenance,
        "width_results": width_results,
        "summary": {key: value for key, value in summary.items() if key != "gates"},
        "gates": summary["gates"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "m6_results.json"
    temporary = output_dir / "m6_results.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    _write_csv(output_dir / "width_sweep.csv", width_results)
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
        raise RuntimeError("M6 width-sweep train-only gate failed")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
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
