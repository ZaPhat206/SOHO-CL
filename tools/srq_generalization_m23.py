"""Three-stream, equal-memory repeat of the M5 RanPAC control.

M23 deliberately reuses the numerical helpers from :mod:`m5`.  Its only
experimental degree of freedom is the stream seed: the same seed controls the
class order, train/validation split, random-ReLU projection, and CountSketch
map.  No test cache is ever read or created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from typing import Any
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import (  # noqa: E402
    DenseSquareRootBackend,
    ExactGramBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from methods.frontends import CountSketchAnalyticLearner, RanPACAnalyticLearner  # noqa: E402
from tools.experiment_runner import validate_cache, split, train_validation_indices  # noqa: E402
from tools import srq_generalization_m4 as m4  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402


STREAMS = (
    {"stream_id": "s2025", "seed": 2025},
    {"stream_id": "s2026", "seed": 2026},
    {"stream_id": "s2027", "seed": 2027},
)
METHODS = (
    "full_width_exact",
    "fp16_square_root",
    "p2b_int8",
    "byte_matched_exact",
    "countsketch_exact",
    "raw_feature_ridge",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_config(path: str | Path) -> dict:
    """Validate the M5 contract with only the stream seed generalized.

    Calling ``m5._read_config`` directly is intentionally impossible because
    that validator locks a single seed of 2025.  The checks below mirror its
    contract; all numerical fields remain frozen to the M5 protocol.
    """

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_top = (set(m5.TOP_KEYS) - {"seed"}) | {"streams"}
    if set(config) != expected_top or config.get("schema_version") != 1:
        raise ValueError("M23 config keys/schema mismatch")
    if config["uses_test_set"] is not False:
        raise ValueError("M23 must remain train-only")
    if config["accuracy_based_selection"] is not False:
        raise ValueError("M23 cannot select dimensions using accuracy")
    if config.get("streams") != list(STREAMS):
        raise ValueError("M23 streams must be exactly s2025/s2026/s2027")
    if (
        config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or not 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("invalid M23 class/task split")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("M23 locks FP32 statistics and solves")

    ranpac = config["ranpac"]
    required_ranpac = {
        "upstream_repository",
        "upstream_commit",
        "upstream_ranpac_py_sha256",
        "path",
        "feature_dimension",
        "full_expand_dimension",
        "projection_distribution",
        "activation",
        "encode_batch_size",
        "evaluation_batch_size",
    }
    if set(ranpac) != required_ranpac:
        raise ValueError("M23 RanPAC fields mismatch")
    if (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or min(
            ranpac[name]
            for name in (
                "feature_dimension",
                "full_expand_dimension",
                "encode_batch_size",
                "evaluation_batch_size",
            )
        )
        <= 0
        or len(ranpac["upstream_commit"]) != 40
        or len(ranpac["upstream_ranpac_py_sha256"]) != 64
    ):
        raise ValueError("invalid locked M23 RanPAC semantics")

    selection = config["ridge_selection"]
    if set(selection) != {
        "policy",
        "candidate_lambdas",
        "fit_samples_per_class",
        "validation_samples_per_class",
        "metric",
        "tie_break",
    }:
        raise ValueError("M23 Ridge-selection fields mismatch")
    candidates = list(map(float, selection["candidate_lambdas"]))
    if (
        selection["policy"] != "per_representation_train_only_calibration"
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
        raise ValueError("invalid M23 Ridge selection")

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
        raise ValueError("M23 P2B fields mismatch")
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
        raise ValueError("M23 no longer matches frozen P2B")
    if p2b["update_trailing_chunk_size"] is not None and p2b[
        "update_trailing_chunk_size"
    ] <= 0:
        raise ValueError("invalid M23 trailing chunk size")

    sketch = config["countsketch"]
    if set(sketch) != {
        "family",
        "bucket_dtype",
        "sign_dtype",
        "normalization",
    }:
        raise ValueError("M23 CountSketch fields mismatch")
    if sketch != {
        "family": "one_nonzero_signed_hash",
        "bucket_dtype": "int32",
        "sign_dtype": "int8",
        "normalization": "none",
    }:
        raise ValueError("invalid M23 CountSketch contract")

    budget = config["budget"]
    if set(budget) != {
        "target",
        "derive_dimensions_before_accuracy",
        "include_projection",
        "include_cross_counts_and_classifier",
        "maximum_underfill_fraction",
    }:
        raise ValueError("M23 budget fields mismatch")
    if (
        budget["target"] != "full_width_p2b_final_total_persistent_tensor_bytes"
        or budget["derive_dimensions_before_accuracy"] is not True
        or budget["include_projection"] is not True
        or budget["include_cross_counts_and_classifier"] is not True
        or not 0 <= budget["maximum_underfill_fraction"] < 1
    ):
        raise ValueError("invalid M23 budget policy")

    gates = config["gates"]
    if set(gates) != {
        "require_byte_derived_dimensions",
        "maximum_p2b_validation_aia_loss_pp",
        "maximum_solver_relative_residual",
        "maximum_equal_or_lower_state_aia_advantage_pp",
        "fail_if_alternative_pareto_dominates_srq",
    }:
        raise ValueError("M23 gate fields mismatch")
    if (
        gates["require_byte_derived_dimensions"] is not True
        or gates["fail_if_alternative_pareto_dominates_srq"] is not True
        or min(
            gates[name]
            for name in (
                "maximum_p2b_validation_aia_loss_pp",
                "maximum_solver_relative_residual",
                "maximum_equal_or_lower_state_aia_advantage_pp",
            )
        )
        < 0
    ):
        raise ValueError("invalid M23 gates")
    return config


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stream_result_is_complete(payload: dict, stream: dict) -> bool:
    return (
        payload.get("status") == "PASS_M5_EQUAL_BUDGET_TRAIN_ONLY"
        and payload.get("uses_test_set") is False
        and payload.get("provenance", {}).get("stream_seed") == stream["seed"]
        and payload.get("stream_id") == stream["stream_id"]
        and all(payload.get("gates", {}).values())
    )


def _run_one_stream(
    *,
    config: dict,
    config_path: Path,
    stream: dict,
    feature_cache_dir: Path,
    output_dir: Path,
    device_name: str,
    require_clean_git: bool,
) -> dict:
    """Run the M5 body once, changing only the stream seed."""

    seed = int(stream["seed"])
    if require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M23 requires a clean source checkout")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M23 refuses a visible test.pt")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    if int(train["features"].shape[1]) != int(config["ranpac"]["feature_dimension"]):
        raise ValueError("M23 feature dimension mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(
        range(config["num_classes"])
    ):
        raise ValueError("M23 training labels do not match locked classes")

    # This is intentionally before projection encoding, calibration, or any
    # validation accuracy, exactly as in M5.
    budget_lock = m5._derive_budget_lock(config)
    print(
        f"BUDGET LOCK {stream['stream_id']} "
        + json.dumps(budget_lock, sort_keys=True),
        flush=True,
    )

    class_order = random.Random(seed).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"],
        task_indices,
        seed,
        config["outer_validation_fraction"],
    )
    selection = config["ridge_selection"]
    fit_indices, calibration_validation_indices = m4._calibration_indices(
        train["labels"],
        training_parts,
        seed=seed,
        fit_per_class=int(selection["fit_samples_per_class"]),
        validation_per_class=int(selection["validation_samples_per_class"]),
    )
    device = torch.device(device_name)
    feature_dimension = int(config["ranpac"]["feature_dimension"])
    full_dimension = int(config["ranpac"]["full_expand_dimension"])
    batch_size = int(config["ranpac"]["encode_batch_size"])
    evaluation_batch_size = int(config["ranpac"]["evaluation_batch_size"])

    owner = DenseSquareRootBackend(
        dimension=full_dimension,
        ridge_lambda=1.0,
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        update_backend="blocked_qr",
        update_panel_size=int(config["p2b"]["update_panel_size"]),
    )
    full_frontend = RanPACAnalyticLearner(
        feature_dim=feature_dimension, backend=owner, seed=seed
    )
    projection = full_frontend.projection
    full_encoder = full_frontend.encode
    reduced_dimension = int(budget_lock["byte_matched_exact_dimension"])
    reduced_projection = projection[:, :reduced_dimension].contiguous()
    reduced_encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ reduced_projection
    )
    raw_encoder = lambda values: values.to(device=device, dtype=torch.float32)
    sketch_dimension = int(budget_lock["countsketch_dimension"])
    sketch_owner = DenseSquareRootBackend(
        dimension=sketch_dimension,
        ridge_lambda=1.0,
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        update_backend="blocked_qr",
        update_panel_size=int(config["p2b"]["update_panel_size"]),
    )
    countsketch = CountSketchAnalyticLearner(
        input_dimension=full_dimension, backend=sketch_owner, seed=seed
    )
    sketch_encoder = lambda values: countsketch.encode_codes(full_encoder(values))

    candidates = list(map(float, selection["candidate_lambdas"]))
    selection_results = {}
    for name, encoder in (
        ("full_random_relu", full_encoder),
        ("reduced_width_random_relu", reduced_encoder),
        ("countsketch_random_relu", sketch_encoder),
        ("raw_features", raw_encoder),
    ):
        selection_results[name] = m5._select_ridge(
            name=name,
            encoder=encoder,
            features=train["features"],
            labels=train["labels"],
            fit_indices=fit_indices,
            validation_indices=calibration_validation_indices,
            candidate_lambdas=candidates,
            num_classes=int(config["num_classes"]),
            batch_size=batch_size,
        )
        print(
            f"RIDGE LOCKED {stream['stream_id']} {name}="
            f"{selection_results[name]['selected_ridge_lambda']}",
            flush=True,
        )
    del owner, sketch_owner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    p2b = config["p2b"]
    full_ridge = selection_results["full_random_relu"]["selected_ridge_lambda"]
    full_common = m5._common_backend(full_dimension, full_ridge, device)
    full_backends = {
        "full_width_exact": ExactGramBackend(**full_common),
        "fp16_square_root": SquareRootBackend(
            storage_mode="float16",
            block_size=int(p2b["block_size"]),
            group_size=int(p2b["group_size"]),
            update_panel_size=int(p2b["update_panel_size"]),
            update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
            first_update_backend=p2b["first_update_backend"],
            quantization_backend=p2b["quantization_backend"],
            quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
            **full_common,
        ),
        "p2b_int8": SquareRootBackend(
            storage_mode="int8",
            block_size=int(p2b["block_size"]),
            group_size=int(p2b["group_size"]),
            update_panel_size=int(p2b["update_panel_size"]),
            update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
            first_update_backend=p2b["first_update_backend"],
            quantization_backend=p2b["quantization_backend"],
            quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
            **full_common,
        ),
    }
    projection_bytes = 4 * feature_dimension * full_dimension
    expected_full = {
        "full_width_exact": projection_bytes
        + m5._exact_backend_bytes(full_dimension, int(config["num_classes"])),
        "fp16_square_root": projection_bytes
        + m5._square_root_backend_bytes(
            full_dimension,
            int(config["num_classes"]),
            block_size=int(p2b["block_size"]),
            group_size=int(p2b["group_size"]),
            mode="float16",
        ),
        "p2b_int8": budget_lock["target_p2b_total_persistent_bytes"],
    }
    numerical_failure = None
    groups = []
    try:
        print(f"START full-width methods {stream['stream_id']}", flush=True)
        groups.append(
            m5._run_group(
                encoder=full_encoder,
                backends=full_backends,
                extra_tensors={"projection": projection},
                expected_total_bytes=expected_full,
                features=train["features"],
                labels=train["labels"],
                training_parts=training_parts,
                validation_parts=validation_parts,
                encode_batch_size=batch_size,
                evaluation_batch_size=evaluation_batch_size,
                device=device,
            )
        )
        del full_backends
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        alternatives = (
            (
                "byte_matched_exact",
                reduced_dimension,
                selection_results["reduced_width_random_relu"][
                    "selected_ridge_lambda"
                ],
                reduced_encoder,
                {"projection": reduced_projection},
                budget_lock["byte_matched_exact_total_persistent_bytes"],
            ),
            (
                "countsketch_exact",
                sketch_dimension,
                selection_results["countsketch_random_relu"][
                    "selected_ridge_lambda"
                ],
                sketch_encoder,
                {
                    "projection": projection,
                    "countsketch.bucket_indices": countsketch.bucket_indices,
                    "countsketch.signs": countsketch.signs,
                },
                budget_lock["countsketch_total_persistent_bytes"],
            ),
            (
                "raw_feature_ridge",
                feature_dimension,
                selection_results["raw_features"]["selected_ridge_lambda"],
                raw_encoder,
                {},
                budget_lock["raw_ridge_total_persistent_bytes"],
            ),
        )
        for name, dimension, ridge, encoder, extras, expected in alternatives:
            print(f"START {stream['stream_id']} {name} dimension={dimension}", flush=True)
            backend = ExactGramBackend(
                **m5._common_backend(dimension, ridge, device)
            )
            groups.append(
                m5._run_group(
                    encoder=encoder,
                    backends={name: backend},
                    extra_tensors=extras,
                    expected_total_bytes={name: expected},
                    features=train["features"],
                    labels=train["labels"],
                    training_parts=training_parts,
                    validation_parts=validation_parts,
                    encode_batch_size=batch_size,
                    evaluation_batch_size=evaluation_batch_size,
                    device=device,
                )
            )
            del backend
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = m5._summarize(groups, config, budget_lock)
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
        "stream_seed": seed,
        "config_sha256": _sha256_file(config_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "generic_backend_sha256": _sha256_file(
            ROOT / "methods/analytic_ridge/backends.py"
        ),
        "ranpac_frontend_sha256": _sha256_file(ROOT / "methods/frontends/ranpac.py"),
        "countsketch_frontend_sha256": _sha256_file(
            ROOT / "methods/frontends/countsketch.py"
        ),
        "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
        "projection_sha256": m4._tensor_content_sha256(projection),
        "countsketch_bucket_sha256": hashlib.sha256(
            countsketch.bucket_indices.cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "countsketch_sign_sha256": hashlib.sha256(
            countsketch.signs.cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "class_order": class_order,
        "training_indices_sha256": m5._sequence_sha256(training_parts),
        "outer_validation_indices_sha256": m5._sequence_sha256(validation_parts),
        "calibration_fit_indices_sha256": m5._sequence_sha256([fit_indices]),
        "calibration_validation_indices_sha256": m5._sequence_sha256(
            [calibration_validation_indices]
        ),
        "upstream_repository": config["ranpac"]["upstream_repository"],
        "upstream_commit": config["ranpac"]["upstream_commit"],
        "upstream_ranpac_py_sha256": config["ranpac"]["upstream_ranpac_py_sha256"],
    }
    passed = numerical_failure is None and all(summary["gates"].values())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "stream_id": stream["stream_id"],
        "status": "PASS_M5_EQUAL_BUDGET_TRAIN_ONLY"
        if passed
        else "FAIL_M5_EQUAL_BUDGET_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "frontend": "RanPAC Phase-2 random-ReLU analytic head",
            "petl_reproduced": False,
            "comparison": "train-only accuracy-state-update Pareto control",
            "countsketch_is": "fixed signed-hash feature sketch before Exact Ridge",
        },
        "budget_lock": budget_lock,
        "ridge_selection": selection_results,
        "numerical_failure": numerical_failure,
        "provenance": provenance,
        "groups": groups,
        "summary": {key: value for key, value in summary.items() if key != "gates"},
        "gates": summary["gates"],
    }
    destination = output_dir / f"stream_{seed}_results.json"
    _write_json_atomic(destination, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "stream_id": stream["stream_id"],
                "summary": payload["summary"],
                "gates": payload["gates"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise RuntimeError("M23 stream failed the original M5 gates")
    return payload


# Public spelling used by the M23 protocol; the implementation remains
# private so callers cannot accidentally bypass the stream contract.
run_one_stream = _run_one_stream


def _read_m5_artifact(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("m5_results.json").decode("utf-8"))


def _close(a: Any, b: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
    return a == b


def _m5_scientific_match(new: dict, old: dict) -> bool:
    """Compare all deterministic scientific fields, excluding wall-clock data."""

    if new.get("budget_lock") != old.get("budget_lock"):
        return False
    for field in (
        "validation_aia_percent",
        "final_validation_accuracy_percent",
        "final_total_persistent_bytes",
        "p2b_validation_aia_loss_pp",
        "best_equal_or_lower_state_aia_advantage_over_p2b_pp",
        "maximum_solver_relative_residual",
    ):
        left, right = new.get("summary", {}).get(field), old.get("summary", {}).get(field)
        if isinstance(left, dict):
            if set(left) != set(right or {}):
                return False
            if any(not _close(left[name], right[name]) for name in left):
                return False
        elif not _close(left, right):
            return False
    if new.get("summary", {}).get("pareto_dominators_of_p2b") != old.get(
        "summary", {}
    ).get("pareto_dominators_of_p2b"):
        return False
    if new.get("ridge_selection") != old.get("ridge_selection"):
        return False
    left_groups, right_groups = new.get("groups"), old.get("groups")
    if not isinstance(left_groups, list) or len(left_groups) != len(right_groups or []):
        return False
    for left_group, right_group in zip(left_groups, right_groups):
        left_records, right_records = left_group.get("records"), right_group.get("records")
        if len(left_records or []) != len(right_records or []):
            return False
        for left_record, right_record in zip(left_records, right_records):
            if left_record.get("task") != right_record.get("task"):
                return False
            if set(left_record.get("accuracy_percent", {})) != set(
                right_record.get("accuracy_percent", {})
            ):
                return False
            if any(
                not _close(left_record["accuracy_percent"][name], right_record["accuracy_percent"][name])
                for name in left_record.get("accuracy_percent", {})
            ):
                return False
            if left_record.get("state") != right_record.get("state"):
                # Residuals are deterministic up to harmless BLAS ordering;
                # bytes must remain exact and residuals use a conservative tol.
                if set(left_record.get("state", {})) != set(right_record.get("state", {})):
                    return False
                for name, left_state in left_record.get("state", {}).items():
                    right_state = right_record["state"][name]
                    if left_state.get("total_persistent_bytes") != right_state.get(
                        "total_persistent_bytes"
                    ) or not _close(
                        left_state.get("solver_relative_residual"),
                        right_state.get("solver_relative_residual"),
                        tolerance=1e-5,
                    ):
                        return False
    return True


def _sample_stats(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("M23 sample statistics require at least two streams")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def summarize_streams(
    output_dir: str | Path | None = None,
    *,
    config: dict | None = None,
    stream_payloads: dict[str, dict] | None = None,
    original_m5_artifact: str | Path | None = None,
) -> dict:
    """Aggregate stream JSON files, or an already-loaded payload mapping.

    The positional ``output_dir`` form is kept for the standalone M23
    protocol (``summarize_streams(output_dir)``).  The runner uses the mapping
    form to avoid rereading files that it just validated.
    """

    if stream_payloads is None:
        if output_dir is None:
            raise ValueError("M23 summary needs output_dir or stream_payloads")
        root = Path(output_dir)
        stream_dir = root / "m23_results" if (root / "m23_results").is_dir() else root
        stream_payloads = {
            item["stream_id"]: _load_json(
                stream_dir / f"stream_{item['seed']}_results.json"
            )
            for item in STREAMS
        }
    if config is None:
        config = _read_config(
            ROOT / "configs" / "srq_generalization_m23_equal_budget_multistream_train_only.json"
        )
    ordered = [stream_payloads[item["stream_id"]] for item in STREAMS]
    locks = [payload["budget_lock"] for payload in ordered]
    lock_identical = all(lock == locks[0] for lock in locks[1:])
    original_match = False
    if original_m5_artifact is None:
        candidate = ROOT / "paper" / "srq_generalization_m5_equal_budget_train_only.zip"
        original_m5_artifact = candidate if candidate.is_file() else None
    if original_m5_artifact is not None and Path(original_m5_artifact).is_file():
        original_match = _m5_scientific_match(
            ordered[0], _read_m5_artifact(Path(original_m5_artifact))
        )
    aggregate: dict[str, dict[str, float]] = {}
    for method in METHODS:
        aia_mean, aia_std = _sample_stats(
            [payload["summary"]["validation_aia_percent"][method] for payload in ordered]
        )
        final_mean, final_std = _sample_stats(
            [
                payload["summary"]["final_validation_accuracy_percent"][method]
                for payload in ordered
            ]
        )
        state_mean, state_std = _sample_stats(
            [
                float(payload["summary"]["final_total_persistent_bytes"][method])
                for payload in ordered
            ]
        )
        update_mean, update_std = _sample_stats(
            [
                payload["summary"]["analytic_update_seconds"][method]
                for payload in ordered
            ]
        )
        aggregate[method] = {
            "aia_mean": aia_mean,
            "aia_std": aia_std,
            "final_accuracy_mean": final_mean,
            "final_accuracy_std": final_std,
            "final_validation_accuracy_mean": final_mean,
            "final_validation_accuracy_std": final_std,
            "state_bytes_mean": state_mean,
            "state_bytes_std": state_std,
            "update_seconds_mean": update_mean,
            "update_seconds_std": update_std,
        }
    every_pass = all(
        _stream_result_is_complete(payload, stream)
        for stream, payload in zip(STREAMS, ordered)
    )
    gates = {
        "all_three_streams_complete": every_pass,
        "budget_lock_identical_across_streams": lock_identical,
        "stream_s2025_matches_original_m5_artifact": original_match,
        "every_stream_passes_original_m5_gates": every_pass,
    }
    return {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M23_EQUAL_BUDGET_MULTISTREAM_TRAIN_ONLY"
        if all(gates.values())
        else "FAIL_M23_EQUAL_BUDGET_MULTISTREAM_TRAIN_ONLY",
        "uses_test_set": False,
        "streams": [item["stream_id"] for item in STREAMS],
        "budget_lock": locks[0] if lock_identical else {"per_stream": locks},
        "per_stream": {item["stream_id"]: stream_payloads[item["stream_id"]] for item in STREAMS},
        "aggregate": aggregate,
        "gates": gates,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    root_output = Path(args.output_dir).resolve()
    per_stream_dir = root_output / "m23_results"
    if getattr(args, "require_clean_git", False) and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M23 requires a clean source checkout")
    if str(getattr(args, "device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M23 requested CUDA but CUDA is unavailable")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M23 refuses a visible test.pt")

    stream_payloads: dict[str, dict] = {}
    for stream in STREAMS:
        destination = per_stream_dir / f"stream_{stream['seed']}_results.json"
        if destination.is_file():
            payload = _load_json(destination)
            if _stream_result_is_complete(payload, stream):
                print(f"RESUME SKIP {stream['stream_id']} {destination}", flush=True)
            else:
                # A failed/incomplete JSON is not a checkpoint.  Recompute it
                # and replace it atomically, preserving any other streams.
                print(
                    f"RESUME REPLACE INCOMPLETE {stream['stream_id']} {destination}",
                    flush=True,
                )
                payload = _run_one_stream(
                    config=config,
                    config_path=config_path,
                    stream=stream,
                    feature_cache_dir=feature_cache_dir,
                    output_dir=per_stream_dir,
                    device_name=args.device,
                    require_clean_git=args.require_clean_git,
                )
        else:
            payload = _run_one_stream(
                config=config,
                config_path=config_path,
                stream=stream,
                feature_cache_dir=feature_cache_dir,
                output_dir=per_stream_dir,
                device_name=args.device,
                require_clean_git=args.require_clean_git,
            )
        stream_payloads[stream["stream_id"]] = payload

    artifact = getattr(args, "original_m5_artifact", None)
    combined = summarize_streams(
        config=config,
        stream_payloads=stream_payloads,
        original_m5_artifact=artifact,
    )
    destination = root_output / "m23_results.json"
    _write_json_atomic(destination, combined)
    print(json.dumps({"status": combined["status"], "gates": combined["gates"]}, indent=2), flush=True)
    if combined["status"] != "PASS_M23_EQUAL_BUDGET_MULTISTREAM_TRAIN_ONLY":
        raise RuntimeError("M23 equal-budget multistream gate failed")
    return combined


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--original-m5-artifact")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
