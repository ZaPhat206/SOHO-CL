"""M24 train-only equal-memory packed-Exact and FD-Ridge controls.

The experiment is anchored to the immutable M23 artifact.  It adds two
controls without changing M23: a physically packed exact symmetric Gram at the
largest width fitting the SRQ-INT8 byte ceiling, and a full-width standard
Frequent Directions summary at the largest rank fitting that ceiling.
Accuracy is measured but is deliberately not a completion gate.
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
import time
import traceback
from typing import Any, Callable
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge.accounting import persistent_tensor_bytes  # noqa: E402
from methods.analytic_ridge.equal_memory_controls import (  # noqa: E402
    FrequentDirectionsRidgeBackend,
    PackedExactGramBackend,
    frequent_directions_backend_bytes,
    largest_frequent_directions_rank,
    largest_packed_exact_dimension,
    packed_exact_backend_bytes,
)
from tools import srq_generalization_m4 as m4  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools.experiment_runner import (  # noqa: E402
    split,
    train_validation_indices,
    validate_cache,
)


STREAMS = (
    {"stream_id": "s2025", "seed": 2025},
    {"stream_id": "s2026", "seed": 2026},
    {"stream_id": "s2027", "seed": 2027},
)
METHODS = ("packed_exact", "frequent_directions_ridge")
PASS_STREAM = "PASS_M24_EQUAL_MEMORY_CONTROLS_STREAM_TRAIN_ONLY"
PASS_STUDY = "PASS_M24_EQUAL_MEMORY_CONTROLS_TRAIN_ONLY"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_top = {
        "schema_version",
        "study_id",
        "dataset",
        "model_name",
        "checkpoint_sha256",
        "uses_test_set",
        "accuracy_based_selection",
        "num_classes",
        "num_tasks",
        "outer_validation_fraction",
        "statistics_dtype",
        "solver_dtype",
        "streams",
        "source_m23",
        "ranpac",
        "ridge_selection",
        "packed_exact",
        "frequent_directions",
        "budget",
        "gates",
    }
    if set(config) != expected_top or config.get("schema_version") != 1:
        raise ValueError("M24 config keys/schema mismatch")
    if config.get("study_id") != "srq-generalization-m24-equal-memory-controls-train-only-v1":
        raise ValueError("M24 study identity mismatch")
    if config.get("uses_test_set") is not False:
        raise ValueError("M24 must remain train-only")
    if config.get("accuracy_based_selection") is not False:
        raise ValueError("M24 cannot select width, rank or lambda using accuracy")
    if config.get("streams") != list(STREAMS):
        raise ValueError("M24 streams must be exactly s2025/s2026/s2027")
    if (
        config.get("dataset") != "CIFAR-100"
        or config.get("model_name") != "vit_base_patch16_224"
        or len(config.get("checkpoint_sha256", "")) != 64
        or config.get("num_classes") != 100
        or config.get("num_tasks") != 10
        or config.get("outer_validation_fraction") != 0.2
        or config.get("statistics_dtype") != "float32"
        or config.get("solver_dtype") != "float32"
    ):
        raise ValueError("M24 dataset/numerical protocol mismatch")

    source = config["source_m23"]
    if set(source) != {
        "filename",
        "sha256",
        "study_id",
        "required_status",
        "target_budget_field",
    }:
        raise ValueError("M24 source-M23 fields mismatch")
    if (
        source["filename"] != "srq_generalization_m23_equal_budget_multistream_train_only.zip"
        or len(source["sha256"]) != 64
        or source["study_id"] != "srq-generalization-m23-equal-budget-multistream-train-only-v1"
        or source["required_status"] != "PASS_M23_EQUAL_BUDGET_MULTISTREAM_TRAIN_ONLY"
        or source["target_budget_field"] != "target_p2b_total_persistent_bytes"
    ):
        raise ValueError("invalid M24 source-M23 contract")

    ranpac = config["ranpac"]
    if ranpac != {
        "path": "phase2_no_petl_random_relu",
        "feature_dimension": 768,
        "full_expand_dimension": 10000,
        "projection_distribution": "standard_normal",
        "activation": "relu",
        "encode_batch_size": 256,
        "evaluation_batch_size": 256,
    }:
        raise ValueError("invalid M24 RanPAC contract")

    selection = config["ridge_selection"]
    if set(selection) != {
        "policy",
        "candidate_lambdas",
        "fit_samples_per_class",
        "validation_samples_per_class",
        "metric",
        "tie_break",
    }:
        raise ValueError("M24 Ridge-selection fields mismatch")
    candidates = list(map(float, selection["candidate_lambdas"]))
    if (
        selection["policy"] != "per_representation_train_only_calibration"
        or selection["metric"] != "mean_squared_error"
        or selection["tie_break"] != "smallest_lambda"
        or candidates != sorted(set(candidates))
        or not candidates
        or min(candidates) <= 0
        or selection["fit_samples_per_class"] != 20
        or selection["validation_samples_per_class"] != 5
    ):
        raise ValueError("invalid M24 Ridge-selection contract")

    packed = config["packed_exact"]
    if packed != {
        "storage": "physical_blocked_upper_triangle_fp32",
        "block_size": 256,
        "projection_policy": "prefix_of_source_full_projection",
        "dimension_policy": "largest_integer_not_exceeding_target_total_bytes",
        "maximum_dimension": 10000,
        "ridge_policy": "same_train_only_grid_selected_for_packed_representation",
    }:
        raise ValueError("invalid M24 packed-Exact contract")
    directions = config["frequent_directions"]
    if directions != {
        "storage": "fp32_rank_by_dimension_summary",
        "algorithm": "deterministic_standard_frequent_directions",
        "shrinkage": "sigma_ell_squared",
        "rank_policy": "largest_integer_not_exceeding_target_total_bytes",
        "ridge_policy": "reuse_source_m23_full_random_relu_lambda",
        "solve": "woodbury",
    }:
        raise ValueError("invalid M24 Frequent-Directions contract")
    budget = config["budget"]
    if budget != {
        "target": "source_m23_full_width_p2b_final_total_persistent_tensor_bytes",
        "derive_width_and_rank_before_encoding_or_accuracy": True,
        "include_projection": True,
        "include_cross_counts_and_classifier": True,
        "element_size_bytes": 4,
    }:
        raise ValueError("invalid M24 budget contract")
    gates = config["gates"]
    if gates != {
        "maximum_solver_relative_residual": 0.0001,
        "require_source_identity_match": True,
        "require_physical_bytes_equal_symbolic_bytes": True,
        "require_maximal_width_and_rank_under_budget": True,
        "require_all_streams_complete": True,
        "accuracy_is_not_a_gate": True,
    }:
        raise ValueError("invalid M24 gate contract")
    return config


def _read_source_m23(path: str | Path, config: dict) -> dict:
    artifact = Path(path).resolve()
    source = config["source_m23"]
    if artifact.name != source["filename"]:
        raise ValueError("M24 source artifact filename mismatch")
    if _sha256_file(artifact) != source["sha256"]:
        raise ValueError("M24 source artifact SHA-256 mismatch")
    with zipfile.ZipFile(artifact) as archive:
        names = set(archive.namelist())
        required = {
            "config.json",
            "m23_results.json",
            "stream_2025_results.json",
            "stream_2026_results.json",
            "stream_2027_results.json",
        }
        if not required.issubset(names):
            raise ValueError("M24 source artifact is incomplete")
        source_config = json.loads(archive.read("config.json"))
        aggregate = json.loads(archive.read("m23_results.json"))
        archived_streams = {
            f"s{seed}": json.loads(archive.read(f"stream_{seed}_results.json"))
            for seed in (2025, 2026, 2027)
        }
    if (
        aggregate.get("study_id") != source["study_id"]
        or aggregate.get("status") != source["required_status"]
        or aggregate.get("uses_test_set") is not False
        or aggregate.get("streams") != [item["stream_id"] for item in STREAMS]
        or not all(aggregate.get("gates", {}).values())
    ):
        raise ValueError("M24 source M23 aggregate did not pass its locked protocol")
    if source_config.get("uses_test_set") is not False:
        raise ValueError("M24 source config is not train-only")
    for stream in STREAMS:
        stream_id = stream["stream_id"]
        embedded = aggregate.get("per_stream", {}).get(stream_id)
        archived = archived_streams[stream_id]
        if embedded != archived:
            raise ValueError(f"M24 source duplicate mismatch for {stream_id}")
        if (
            archived.get("status") != "PASS_M5_EQUAL_BUDGET_TRAIN_ONLY"
            or archived.get("uses_test_set") is not False
            or archived.get("stream_id") != stream_id
            or archived.get("provenance", {}).get("stream_seed") != stream["seed"]
            or not all(archived.get("gates", {}).values())
        ):
            raise ValueError(f"M24 source stream is invalid: {stream_id}")
    aggregate["_artifact_path"] = str(artifact)
    aggregate["_artifact_sha256"] = source["sha256"]
    aggregate["_archived_streams"] = archived_streams
    return aggregate


def derive_budget_lock(config: dict, source_m23: dict) -> dict:
    """Lock both controls using bytes only; call before any encoding."""

    field = config["source_m23"]["target_budget_field"]
    target = int(source_m23["budget_lock"][field])
    feature_dimension = int(config["ranpac"]["feature_dimension"])
    full_dimension = int(config["ranpac"]["full_expand_dimension"])
    classes = int(config["num_classes"])
    element_size = int(config["budget"]["element_size_bytes"])
    packed_dimension, packed_bytes = largest_packed_exact_dimension(
        target_total_bytes=target,
        feature_dimension=feature_dimension,
        classes=classes,
        maximum_dimension=int(config["packed_exact"]["maximum_dimension"]),
        element_size=element_size,
    )
    fd_rank, fd_bytes = largest_frequent_directions_rank(
        target_total_bytes=target,
        feature_dimension=feature_dimension,
        expanded_dimension=full_dimension,
        classes=classes,
        element_size=element_size,
    )
    packed_next = element_size * feature_dimension * (packed_dimension + 1)
    packed_next += packed_exact_backend_bytes(
        packed_dimension + 1, classes, element_size=element_size
    )
    fd_next = element_size * feature_dimension * full_dimension
    fd_next += frequent_directions_backend_bytes(
        full_dimension, classes, fd_rank + 1, element_size=element_size
    )
    return {
        "policy": "largest_integer_width_or_rank_not_exceeding_source_srq_bytes",
        "locked_before_representation_encoding_or_accuracy": True,
        "target_total_persistent_bytes": target,
        "packed_exact_dimension": packed_dimension,
        "packed_exact_total_persistent_bytes": packed_bytes,
        "packed_exact_underfill_bytes": target - packed_bytes,
        "packed_exact_next_dimension_bytes": packed_next,
        "frequent_directions_dimension": full_dimension,
        "frequent_directions_rank": fd_rank,
        "frequent_directions_total_persistent_bytes": fd_bytes,
        "frequent_directions_underfill_bytes": target - fd_bytes,
        "frequent_directions_next_rank_bytes": fd_next,
    }


def _projection(seed: int, feature_dimension: int, expanded_dimension: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn(
        feature_dimension,
        expanded_dimension,
        generator=generator,
        dtype=torch.float32,
    )
    return values.to(device=device)


def _evaluate_one(
    *,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    backend: object,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
) -> float:
    result = m5._evaluate(
        encoder=encoder,
        backends={"method": backend},
        features=features,
        labels=labels,
        indices=indices,
        batch_size=batch_size,
    )
    return float(result["method"])


def _run_method(
    *,
    name: str,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    backend: object,
    projection: torch.Tensor,
    expected_total_bytes: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    encode_batch_size: int,
    evaluation_batch_size: int,
    device: torch.device,
) -> dict:
    records = []
    update_seconds = 0.0
    encoding_seconds = 0.0
    maximum_update_peak_allocated = 0
    for task_index, training_indices in enumerate(training_parts):
        m5._sync(device)
        started = time.perf_counter()
        codes = m5._encode_indices(
            encoder, features, training_indices, encode_batch_size
        )
        m5._sync(device)
        encoding_seconds += time.perf_counter() - started
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        m5._sync(device)
        started = time.perf_counter()
        backend.update(codes, labels[training_indices])
        m5._sync(device)
        update_seconds += time.perf_counter() - started
        if device.type == "cuda":
            maximum_update_peak_allocated = max(
                maximum_update_peak_allocated,
                int(torch.cuda.max_memory_allocated(device)),
            )
        seen_validation = torch.cat(validation_parts[: task_index + 1])
        accuracy = _evaluate_one(
            encoder=encoder,
            backend=backend,
            features=features,
            labels=labels,
            indices=seen_validation,
            batch_size=evaluation_batch_size,
        )
        tensors = {"projection": projection}
        tensors.update(backend.persistent_tensors())
        actual_bytes = persistent_tensor_bytes(tensors)
        if task_index + 1 == len(training_parts) and actual_bytes != expected_total_bytes:
            raise AssertionError(
                f"{name} symbolic/physical state mismatch: "
                f"{expected_total_bytes} != {actual_bytes}"
            )
        residual = float(backend.diagnostics["solver_relative_residual"])
        diagnostics = {
            key: value
            for key, value in backend.diagnostics.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        records.append(
            {
                "task": task_index + 1,
                "validation_accuracy_percent": accuracy,
                "total_persistent_bytes": actual_bytes,
                "solver_relative_residual": residual,
                "diagnostics": diagnostics,
            }
        )
        print(
            f"M24 {name} task={task_index + 1}/{len(training_parts)} "
            f"validation={accuracy:.4f} state={actual_bytes} residual={residual:.3e}",
            flush=True,
        )
        del codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    accuracies = [record["validation_accuracy_percent"] for record in records]
    return {
        "method": name,
        "records": records,
        "validation_aia_percent": sum(accuracies) / len(accuracies),
        "final_validation_accuracy_percent": accuracies[-1],
        "final_total_persistent_bytes": records[-1]["total_persistent_bytes"],
        "maximum_solver_relative_residual": max(
            record["solver_relative_residual"] for record in records
        ),
        "analytic_update_seconds": update_seconds,
        "representation_encoding_seconds": encoding_seconds,
        "maximum_update_peak_allocated_bytes": maximum_update_peak_allocated,
        "final_diagnostics": records[-1]["diagnostics"],
    }


def _identity_matches_source(
    *,
    source_stream: dict,
    train_sha256: str,
    projection: torch.Tensor,
    class_order: list[int],
    training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    fit_indices: torch.Tensor,
    calibration_validation_indices: torch.Tensor,
) -> dict[str, bool]:
    provenance = source_stream["provenance"]
    return {
        "train_cache": train_sha256 == provenance["train_sha256"],
        "full_projection": m4._tensor_content_sha256(projection)
        == provenance["projection_sha256"],
        "class_order": class_order == provenance["class_order"],
        "training_indices": m5._sequence_sha256(training_parts)
        == provenance["training_indices_sha256"],
        "outer_validation_indices": m5._sequence_sha256(validation_parts)
        == provenance["outer_validation_indices_sha256"],
        "calibration_fit_indices": m5._sequence_sha256([fit_indices])
        == provenance["calibration_fit_indices_sha256"],
        "calibration_validation_indices": m5._sequence_sha256(
            [calibration_validation_indices]
        )
        == provenance["calibration_validation_indices_sha256"],
    }


def _current_hashes(config_path: Path) -> dict[str, str]:
    return {
        "config_sha256": _sha256_file(config_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "control_backend_sha256": _sha256_file(
            ROOT / "methods/analytic_ridge/equal_memory_controls.py"
        ),
    }


def _stream_result_is_reusable(
    payload: dict,
    *,
    stream: dict,
    source_sha256: str,
    current_hashes: dict[str, str],
) -> bool:
    provenance = payload.get("provenance", {})
    return (
        payload.get("status") == PASS_STREAM
        and payload.get("uses_test_set") is False
        and payload.get("accuracy_based_selection") is False
        and payload.get("stream_id") == stream["stream_id"]
        and provenance.get("stream_seed") == stream["seed"]
        and provenance.get("source_m23_artifact_sha256") == source_sha256
        and all(provenance.get(key) == value for key, value in current_hashes.items())
        and all(payload.get("gates", {}).values())
    )


def _run_one_stream(
    *,
    config: dict,
    config_path: Path,
    stream: dict,
    source_m23: dict,
    feature_cache_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    seed = int(stream["seed"])
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M24 refuses a visible test.pt")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("M24 feature-cache checkpoint SHA-256 mismatch")
    if int(train["features"].shape[1]) != config["ranpac"]["feature_dimension"]:
        raise ValueError("M24 feature dimension mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(
        range(config["num_classes"])
    ):
        raise ValueError("M24 training labels do not match locked classes")

    # The byte-only decisions occur before constructing a projection, encoding
    # a feature, calibrating Ridge, or measuring validation accuracy.
    budget_lock = derive_budget_lock(config, source_m23)
    print(
        f"M24 BUDGET LOCK {stream['stream_id']} "
        + json.dumps(budget_lock, sort_keys=True),
        flush=True,
    )

    source_stream = source_m23["_archived_streams"][stream["stream_id"]]
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
    feature_dimension = int(config["ranpac"]["feature_dimension"])
    full_dimension = int(config["ranpac"]["full_expand_dimension"])
    projection = _projection(seed, feature_dimension, full_dimension, device)
    identity = _identity_matches_source(
        source_stream=source_stream,
        train_sha256=_sha256_file(feature_cache_dir / "train.pt"),
        projection=projection,
        class_order=class_order,
        training_parts=training_parts,
        validation_parts=validation_parts,
        fit_indices=fit_indices,
        calibration_validation_indices=calibration_validation_indices,
    )
    if not all(identity.values()):
        raise AssertionError(f"M24/M23 stream identity mismatch: {identity}")

    packed_dimension = int(budget_lock["packed_exact_dimension"])
    packed_projection = projection[:, :packed_dimension].contiguous()
    full_encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ projection
    )
    packed_encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ packed_projection
    )
    packed_selection = m5._select_ridge(
        name="packed_exact_random_relu",
        encoder=packed_encoder,
        features=train["features"],
        labels=train["labels"],
        fit_indices=fit_indices,
        validation_indices=calibration_validation_indices,
        candidate_lambdas=list(map(float, selection["candidate_lambdas"])),
        num_classes=int(config["num_classes"]),
        batch_size=int(config["ranpac"]["encode_batch_size"]),
    )
    fd_ridge = float(
        source_stream["ridge_selection"]["full_random_relu"][
            "selected_ridge_lambda"
        ]
    )
    ridge_lock = {
        "packed_exact_random_relu": packed_selection,
        "frequent_directions_random_relu": {
            "representation": "full_random_relu",
            "selected_ridge_lambda": fd_ridge,
            "policy": "reused_from_verified_source_m23_stream",
            "source_stream_id": stream["stream_id"],
        },
    }
    print(
        f"M24 RIDGE LOCK {stream['stream_id']} packed="
        f"{packed_selection['selected_ridge_lambda']} fd={fd_ridge}",
        flush=True,
    )

    methods = {}
    numerical_failure = None
    try:
        packed_backend = PackedExactGramBackend(
            dimension=packed_dimension,
            ridge_lambda=float(packed_selection["selected_ridge_lambda"]),
            block_size=int(config["packed_exact"]["block_size"]),
            device=device,
            statistics_dtype=torch.float32,
            solver_dtype=torch.float32,
        )
        methods["packed_exact"] = _run_method(
            name="packed_exact",
            encoder=packed_encoder,
            backend=packed_backend,
            projection=packed_projection,
            expected_total_bytes=int(
                budget_lock["packed_exact_total_persistent_bytes"]
            ),
            features=train["features"],
            labels=train["labels"],
            training_parts=training_parts,
            validation_parts=validation_parts,
            encode_batch_size=int(config["ranpac"]["encode_batch_size"]),
            evaluation_batch_size=int(config["ranpac"]["evaluation_batch_size"]),
            device=device,
        )
        del packed_backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fd_backend = FrequentDirectionsRidgeBackend(
            dimension=full_dimension,
            ridge_lambda=fd_ridge,
            sketch_rank=int(budget_lock["frequent_directions_rank"]),
            device=device,
            statistics_dtype=torch.float32,
            solver_dtype=torch.float32,
        )
        methods["frequent_directions_ridge"] = _run_method(
            name="frequent_directions_ridge",
            encoder=full_encoder,
            backend=fd_backend,
            projection=projection,
            expected_total_bytes=int(
                budget_lock["frequent_directions_total_persistent_bytes"]
            ),
            features=train["features"],
            labels=train["labels"],
            training_parts=training_parts,
            validation_parts=validation_parts,
            encode_batch_size=int(config["ranpac"]["encode_batch_size"]),
            evaluation_batch_size=int(config["ranpac"]["evaluation_batch_size"]),
            device=device,
        )
        del fd_backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (RuntimeError, AssertionError, torch.linalg.LinAlgError) as error:
        numerical_failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    target = int(budget_lock["target_total_persistent_bytes"])
    residual_limit = float(config["gates"]["maximum_solver_relative_residual"])
    physical_match = numerical_failure is None and all(
        methods[name]["final_total_persistent_bytes"]
        == budget_lock[
            "packed_exact_total_persistent_bytes"
            if name == "packed_exact"
            else "frequent_directions_total_persistent_bytes"
        ]
        for name in METHODS
    )
    under_budget = numerical_failure is None and all(
        methods[name]["final_total_persistent_bytes"] <= target for name in METHODS
    )
    residual_pass = numerical_failure is None and all(
        methods[name]["maximum_solver_relative_residual"] <= residual_limit
        for name in METHODS
    )
    maximal = (
        budget_lock["packed_exact_next_dimension_bytes"] > target
        and budget_lock["frequent_directions_next_rank_bytes"] > target
    )
    gates = {
        "source_stream_identity_match": all(identity.values()),
        "budget_locked_before_encoding_or_accuracy": budget_lock[
            "locked_before_representation_encoding_or_accuracy"
        ],
        "physical_bytes_equal_symbolic_bytes": physical_match,
        "both_controls_within_source_srq_budget": under_budget,
        "width_and_rank_are_maximal_under_budget": maximal,
        "solver_residual": residual_pass,
        "numerical_success": numerical_failure is None,
        "accuracy_not_used_as_gate": True,
        "test_cache_absent": not (feature_cache_dir / "test.pt").exists(),
    }
    current_hashes = _current_hashes(config_path)
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
        **current_hashes,
        "source_m23_artifact_sha256": source_m23["_artifact_sha256"],
        "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
        "full_projection_sha256": m4._tensor_content_sha256(projection),
        "packed_projection_sha256": m4._tensor_content_sha256(packed_projection),
        "class_order": class_order,
        "training_indices_sha256": m5._sequence_sha256(training_parts),
        "outer_validation_indices_sha256": m5._sequence_sha256(validation_parts),
        "calibration_fit_indices_sha256": m5._sequence_sha256([fit_indices]),
        "calibration_validation_indices_sha256": m5._sequence_sha256(
            [calibration_validation_indices]
        ),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "stream_id": stream["stream_id"],
        "status": PASS_STREAM if passed else "FAIL_M24_EQUAL_MEMORY_CONTROLS_STREAM_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "frontend": "RanPAC Phase-2 random-ReLU analytic head",
            "data": "CIFAR-100 train split with held-out train validation",
            "packed_exact": "physical unique symmetric FP32 Gram entries",
            "frequent_directions": "standard deterministic covariance sketch",
            "workspace_not_counted_as_persistent_state": True,
        },
        "budget_lock": budget_lock,
        "ridge_selection": ridge_lock,
        "source_reference": {
            "stream_id": stream["stream_id"],
            "p2b_int8": {
                "validation_aia_percent": source_stream["summary"][
                    "validation_aia_percent"
                ]["p2b_int8"],
                "final_validation_accuracy_percent": source_stream["summary"][
                    "final_validation_accuracy_percent"
                ]["p2b_int8"],
                "final_total_persistent_bytes": source_stream["summary"][
                    "final_total_persistent_bytes"
                ]["p2b_int8"],
            },
        },
        "methods": methods,
        "numerical_failure": numerical_failure,
        "identity_checks": identity,
        "provenance": provenance,
        "gates": gates,
    }
    destination = output_dir / f"stream_{seed}_results.json"
    _write_json_atomic(destination, payload)
    print(
        json.dumps(
            {
                "stream_id": stream["stream_id"],
                "status": payload["status"],
                "gates": gates,
                "measurements": {
                    name: {
                        "aia": methods.get(name, {}).get("validation_aia_percent"),
                        "final": methods.get(name, {}).get(
                            "final_validation_accuracy_percent"
                        ),
                        "state": methods.get(name, {}).get(
                            "final_total_persistent_bytes"
                        ),
                    }
                    for name in METHODS
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise RuntimeError(f"M24 stream failed structural gates: {gates}")
    return payload


def _sample_stats(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("sample statistics require at least two streams")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def summarize_streams(
    *, config: dict, source_m23: dict, stream_payloads: dict[str, dict]
) -> dict:
    ordered = [stream_payloads[item["stream_id"]] for item in STREAMS]
    locks = [payload["budget_lock"] for payload in ordered]
    locks_identical = all(lock == locks[0] for lock in locks[1:])
    aggregate = {}
    for method in METHODS:
        fields = {
            "aia": "validation_aia_percent",
            "final_accuracy": "final_validation_accuracy_percent",
            "state_bytes": "final_total_persistent_bytes",
            "update_seconds": "analytic_update_seconds",
            "maximum_update_peak_allocated_bytes": "maximum_update_peak_allocated_bytes",
        }
        method_aggregate = {}
        for output_name, source_name in fields.items():
            mean, std = _sample_stats(
                [float(payload["methods"][method][source_name]) for payload in ordered]
            )
            method_aggregate[f"{output_name}_mean"] = mean
            method_aggregate[f"{output_name}_std"] = std
        aggregate[method] = method_aggregate

    comparisons = {}
    for method in METHODS:
        advantages = [
            float(payload["source_reference"]["p2b_int8"]["validation_aia_percent"])
            - float(payload["methods"][method]["validation_aia_percent"])
            for payload in ordered
        ]
        mean, std = _sample_stats(advantages)
        comparisons[method] = {
            "srq_aia_advantage_over_control_pp_per_stream": advantages,
            "srq_aia_advantage_over_control_pp_mean": mean,
            "srq_aia_advantage_over_control_pp_std": std,
            "srq_higher_aia_streams": sum(value > 0 for value in advantages),
            "control_higher_aia_streams": sum(value < 0 for value in advantages),
            "ties": sum(value == 0 for value in advantages),
        }

    source_reference = {
        method: source_m23["aggregate"][method]
        for method in (
            "p2b_int8",
            "byte_matched_exact",
            "countsketch_exact",
            "full_width_exact",
        )
    }
    all_streams_complete = all(
        payload.get("status") == PASS_STREAM
        and payload.get("uses_test_set") is False
        and all(payload.get("gates", {}).values())
        for payload in ordered
    )
    gates = {
        "all_three_streams_complete": all_streams_complete,
        "budget_lock_identical_across_streams": locks_identical,
        "source_m23_verified": source_m23["status"]
        == config["source_m23"]["required_status"],
        "accuracy_not_used_as_gate": True,
    }
    return {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": PASS_STUDY if all(gates.values()) else "FAIL_M24_EQUAL_MEMORY_CONTROLS_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "streams": [item["stream_id"] for item in STREAMS],
        "budget_lock": locks[0] if locks_identical else {"per_stream": locks},
        "source_m23_artifact_sha256": source_m23["_artifact_sha256"],
        "source_reference": source_reference,
        "per_stream": stream_payloads,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "gates": gates,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    source_m23 = _read_source_m23(args.source_m23_artifact, config)
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    per_stream_dir = output_dir / "m24_results"
    if getattr(args, "require_clean_git", False) and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M24 requires a clean source checkout")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("M24 requested CUDA but CUDA is unavailable")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M24 refuses a visible test.pt")

    current_hashes = _current_hashes(config_path)
    stream_payloads = {}
    for stream in STREAMS:
        destination = per_stream_dir / f"stream_{stream['seed']}_results.json"
        payload = None
        if destination.is_file():
            candidate = json.loads(destination.read_text(encoding="utf-8"))
            if _stream_result_is_reusable(
                candidate,
                stream=stream,
                source_sha256=source_m23["_artifact_sha256"],
                current_hashes=current_hashes,
            ):
                payload = candidate
                print(f"M24 RESUME SKIP {stream['stream_id']} {destination}", flush=True)
            else:
                print(
                    f"M24 RESUME REPLACE INCOMPLETE {stream['stream_id']} {destination}",
                    flush=True,
                )
        if payload is None:
            payload = _run_one_stream(
                config=config,
                config_path=config_path,
                stream=stream,
                source_m23=source_m23,
                feature_cache_dir=feature_cache_dir,
                output_dir=per_stream_dir,
                device=device,
            )
        stream_payloads[stream["stream_id"]] = payload

    combined = summarize_streams(
        config=config, source_m23=source_m23, stream_payloads=stream_payloads
    )
    destination = output_dir / "m24_results.json"
    _write_json_atomic(destination, combined)
    print(
        json.dumps(
            {
                "status": combined["status"],
                "aggregate": combined["aggregate"],
                "paired_comparisons": combined["paired_comparisons"],
                "gates": combined["gates"],
            },
            indent=2,
        ),
        flush=True,
    )
    if combined["status"] != PASS_STUDY:
        raise RuntimeError("M24 aggregate structural gates failed")
    return combined


def run_stream(args) -> dict:
    """Run or resume one preregistered stream for Colab handoff safety."""

    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    source_m23 = _read_source_m23(args.source_m23_artifact, config)
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if getattr(args, "require_clean_git", False) and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M24 requires a clean source checkout")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("M24 requested CUDA but CUDA is unavailable")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M24 refuses a visible test.pt")
    matches = [item for item in STREAMS if item["stream_id"] == args.stream_id]
    if len(matches) != 1:
        raise ValueError("M24 run-stream needs one preregistered stream_id")
    stream = matches[0]
    destination = output_dir / "m24_results" / f"stream_{stream['seed']}_results.json"
    current_hashes = _current_hashes(config_path)
    if destination.is_file():
        candidate = json.loads(destination.read_text(encoding="utf-8"))
        if _stream_result_is_reusable(
            candidate,
            stream=stream,
            source_sha256=source_m23["_artifact_sha256"],
            current_hashes=current_hashes,
        ):
            print(f"M24 RESUME SKIP {stream['stream_id']} {destination}", flush=True)
            return candidate
        print(f"M24 REPLACE INCOMPLETE {stream['stream_id']} {destination}", flush=True)
    return _run_one_stream(
        config=config,
        config_path=config_path,
        stream=stream,
        source_m23=source_m23,
        feature_cache_dir=feature_cache_dir,
        output_dir=output_dir / "m24_results",
        device=device,
    )


def summarize(args) -> dict:
    """Aggregate three verified per-stream JSON files without rerunning them."""

    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    source_m23 = _read_source_m23(args.source_m23_artifact, config)
    output_dir = Path(args.output_dir).resolve()
    current_hashes = _current_hashes(config_path)
    payloads = {}
    for stream in STREAMS:
        path = output_dir / "m24_results" / f"stream_{stream['seed']}_results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing M24 stream result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _stream_result_is_reusable(
            payload,
            stream=stream,
            source_sha256=source_m23["_artifact_sha256"],
            current_hashes=current_hashes,
        ):
            raise ValueError(f"M24 stream result is stale or invalid: {path}")
        payloads[stream["stream_id"]] = payload
    combined = summarize_streams(
        config=config, source_m23=source_m23, stream_payloads=payloads
    )
    destination = output_dir / "m24_results.json"
    _write_json_atomic(destination, combined)
    print(
        json.dumps(
            {
                "status": combined["status"],
                "aggregate": combined["aggregate"],
                "paired_comparisons": combined["paired_comparisons"],
                "gates": combined["gates"],
            },
            indent=2,
        ),
        flush=True,
    )
    if combined["status"] != PASS_STUDY:
        raise RuntimeError("M24 aggregate structural gates failed")
    return combined


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "run-stream", "summarize"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-m23-artifact", required=True)
    parser.add_argument("--feature-cache-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stream-id", choices=tuple(item["stream_id"] for item in STREAMS))
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"run", "run-stream"} and not args.feature_cache_dir:
        parser.error(f"{args.command} requires --feature-cache-dir")
    if args.command == "run-stream" and not args.stream_id:
        parser.error("run-stream requires --stream-id")
    if args.command != "run-stream" and args.stream_id:
        parser.error("--stream-id is only valid with run-stream")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "run-stream":
        run_stream(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
