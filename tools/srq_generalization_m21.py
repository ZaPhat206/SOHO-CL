"""M21: train-only controls for error accumulation and direct-Gram loading.

Part 1 (gram_load).  On the Priority-3 FLY-CL stream (CIFAR-100, width
10,000), the direct-INT8 Gram matrix is repaired with the smallest diagonal
load on a fixed grid for which the FP32 Cholesky factorization succeeds.  The
load carries no certificate.  Exact, SRQ-INT8 and the Weyl-certified repair
are rerun on the same stream and must reproduce Priority 3.

Part 2 (accumulation).  On RanPAC random-ReLU CIFAR-100 validation streams,
three arms advance in lockstep: Exact, recursive SRQ-INT8 (the locked
backend, whose stored factor carries the quantization error of every earlier
task), and a one-shot counterfactual that quantizes the exact factor of the
current system once with the same codec.  The one-shot arm needs the exact
statistics and is therefore a diagnostic, not a method.  The recursive arm
must reproduce M6/M20 and M7.

No unit reads a test split, and no accuracy value selects a method, load,
seed or hyperparameter.  Interpretation thresholds are fixed in the config.
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
    CompressedUpper,
    ExactGramBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from methods.analytic_ridge.backends import _relative_factor_residual  # noqa: E402
from methods.srq_fly_optimized.minimal_load_control import (  # noqa: E402
    MinimalLoadDirectInt8GramLearner,
)
from tools import srq_fly_d0 as d0  # noqa: E402
from tools import srq_fly_priority1_ablation as p1  # noqa: E402
from tools import srq_fly_priority3_direct_control as p3  # noqa: E402
from tools import srq_generalization_m4 as m4  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools import srq_generalization_m6 as m6  # noqa: E402
from tools import srq_generalization_m11 as m11  # noqa: E402
from tools import srq_generalization_m20 as m20  # noqa: E402
from tools.experiment_runner import (  # noqa: E402
    split,
    train_validation_indices,
    validate_cache,
)


STATUS_PASS = "PASS_M21_ACCUMULATION_GRAM_LOAD_TRAIN_ONLY"
STATUS_WARNING = "COMPLETE_M21_ACCUMULATION_GRAM_LOAD_WITH_WARNINGS"
TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name", "checkpoint_sha256",
    "train_cache_sha256", "uses_test_set", "accuracy_based_selection", "seed",
    "num_classes", "num_tasks", "statistics_dtype", "solver_dtype",
    "required_device_name_substring", "gram_load", "accumulation", "gates",
}
GRAM_LOAD_METHODS = (
    "exact_fly_10000",
    "srq_int8_p2b",
    "direct_int8_gram_weyl_repair",
    "direct_int8_gram_minimal_load",
)
REPRODUCED_GRAM_LOAD_METHODS = GRAM_LOAD_METHODS[:3]
ACCUMULATION_ARMS = ("exact", "recursive_int8", "one_shot_int8")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _sha256(path: str | Path, *, source: bool = False) -> str:
    return m20._sha256(path, source=source)


def validate_config(config: dict) -> dict:
    if set(config) != TOP_KEYS:
        raise ValueError(
            f"M21 config keys mismatch: missing={sorted(TOP_KEYS - set(config))}, "
            f"unexpected={sorted(set(config) - TOP_KEYS)}"
        )
    if config["schema_version"] != 1 or int(config["seed"]) != 2025:
        raise ValueError("M21 requires schema 1 and protocol seed 2025")
    if config["uses_test_set"] is not False or config["accuracy_based_selection"] is not False:
        raise ValueError("M21 must stay train-only and must not select from accuracy")
    if (
        config["dataset"] != "CIFAR-100"
        or config["model_name"] != "vit_base_patch16_224"
        or config["statistics_dtype"] != "float32"
        or config["solver_dtype"] != "float32"
        or int(config["num_classes"]) != 100
        or int(config["num_tasks"]) != 10
    ):
        raise ValueError("M21 no longer matches the locked CIFAR-100 protocol")

    gram = config["gram_load"]
    if set(gram) != {
        "priority3_config", "priority3_config_sha256", "methods", "minimal_load",
        "priority3_reference", "interpretation",
    }:
        raise ValueError("M21 gram_load fields mismatch")
    if tuple(gram["methods"]) != GRAM_LOAD_METHODS:
        raise ValueError("M21 gram_load methods changed")
    load = gram["minimal_load"]
    if set(load) != {
        "load_margin_multiplier", "load_grid_steps_per_doubling",
        "maximum_load_to_ridge_ratio", "error_chunk_size", "success_test",
        "monotone_success_assumed", "uses_labels_or_accuracy",
    } or (
        float(load["load_margin_multiplier"]) != 8.0
        or int(load["load_grid_steps_per_doubling"]) <= 0
        or float(load["maximum_load_to_ridge_ratio"]) <= 1.0
        or int(load["error_chunk_size"]) <= 0
        or load["success_test"] != "fp32_cholesky_of_symmetrized_loaded_system"
        or load["monotone_success_assumed"] is not True
        or load["uses_labels_or_accuracy"] is not False
    ):
        raise ValueError("M21 minimal-load policy changed")
    reference = gram["priority3_reference"]
    if set(reference) != {"artifact_filename", "artifact_sha256", "validation_average_accuracy"} or set(
        reference["validation_average_accuracy"]
    ) != set(REPRODUCED_GRAM_LOAD_METHODS):
        raise ValueError("M21 Priority-3 reference fields mismatch")
    interpretation = gram["interpretation"]
    if set(interpretation) != {"practical_equivalence_pp", "material_advantage_pp"} or not (
        0 < float(interpretation["practical_equivalence_pp"])
        < float(interpretation["material_advantage_pp"])
    ):
        raise ValueError("M21 gram_load interpretation thresholds are invalid")

    accumulation = config["accumulation"]
    if set(accumulation) != {
        "outer_validation_fraction", "ridge_lambda", "streams", "units", "source_m6",
        "ranpac", "p2b", "system_probe_count", "system_probe_seed", "reference",
        "interpretation",
    }:
        raise ValueError("M21 accumulation fields mismatch")
    if float(accumulation["ridge_lambda"]) != 1.0e6 or not 0 < float(
        accumulation["outer_validation_fraction"]
    ) < 1:
        raise ValueError("M21 accumulation must reuse the M6 ridge and split")
    streams = accumulation["streams"]
    identifiers = [stream["stream_id"] for stream in streams]
    if not streams or len(set(identifiers)) != len(identifiers):
        raise ValueError("M21 streams must be non-empty and unique")
    for stream in streams:
        if set(stream) != {
            "stream_id", "class_order_seed", "split_seed", "projection_seed",
            "m6_identity_locked",
        }:
            raise ValueError("M21 stream fields mismatch")
    locked = [stream for stream in streams if stream["m6_identity_locked"]]
    if (
        len(locked) != 1
        or streams[0] is not locked[0]
        or any(int(locked[0][key]) != 2025 for key in ("class_order_seed", "split_seed", "projection_seed"))
    ):
        raise ValueError("M21 needs exactly one first M6-locked stream with seeds 2025")
    units = accumulation["units"]
    unit_ids = [unit["unit_id"] for unit in units]
    if not units or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("M21 accumulation units must be non-empty and unique")
    for unit in units:
        if set(unit) != {"unit_id", "width", "stream_id"} or unit["stream_id"] not in identifiers:
            raise ValueError("M21 accumulation unit fields mismatch")
        if not 0 < int(unit["width"]) <= int(accumulation["ranpac"]["maximum_expand_dimension"]):
            raise ValueError("M21 accumulation width outside the projection")
    if set(accumulation["reference"]) != set(unit_ids):
        raise ValueError("M21 needs one reproduction reference per accumulation unit")
    for unit_id, row in accumulation["reference"].items():
        if set(row) != {
            "source", "exact_aia_percent", "exact_final_percent", "exact_bytes",
            "p2b_int8_aia_percent", "p2b_int8_final_percent", "p2b_int8_bytes",
            "m7_p2b_int8_relative_logit_error",
        }:
            raise ValueError(f"M21 reference fields mismatch for {unit_id}")
        curve = row["m7_p2b_int8_relative_logit_error"]
        if curve is not None and len(curve) != int(config["num_tasks"]):
            raise ValueError(f"M21 M7 reference curve length mismatch for {unit_id}")

    ranpac = accumulation["ranpac"]
    if set(ranpac) != {
        "path", "feature_dimension", "maximum_expand_dimension",
        "projection_distribution", "activation", "encode_batch_size",
        "evaluation_batch_size",
    } or (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or min(int(ranpac["encode_batch_size"]), int(ranpac["evaluation_batch_size"])) <= 0
    ):
        raise ValueError("invalid M21 RanPAC semantics")
    p2b = accumulation["p2b"]
    if set(p2b) != {
        "block_size", "group_size", "update_panel_size", "update_trailing_chunk_size",
        "first_update_backend", "quantization_backend", "quantization_batch_blocks",
    } or (
        p2b["update_trailing_chunk_size"] is not None
        or p2b["first_update_backend"] != "gram_cholesky"
        or p2b["quantization_backend"] != "streaming"
        or min(int(p2b[key]) for key in (
            "block_size", "group_size", "update_panel_size", "quantization_batch_blocks"
        )) <= 0
    ):
        raise ValueError("M21 must use the locked P2B storage and update settings")
    if int(accumulation["system_probe_count"]) <= 0:
        raise ValueError("M21 needs at least one system probe")
    rule = accumulation["interpretation"]
    if set(rule) != {
        "primary_width", "primary_endpoint", "dominant_minimum_ratio",
        "contributes_minimum_ratio",
    } or (
        rule["primary_endpoint"] != "final_relative_logit_error_recursive_over_one_shot"
        or not 1.0 <= float(rule["contributes_minimum_ratio"]) < float(rule["dominant_minimum_ratio"])
        or int(rule["primary_width"]) not in {int(unit["width"]) for unit in units}
    ):
        raise ValueError("M21 accumulation interpretation rule is invalid")

    gates = config["gates"]
    if set(gates) != {
        "maximum_solver_relative_residual", "maximum_reproduction_accuracy_difference_pp",
        "maximum_m7_logit_error_relative_difference",
        "maximum_task1_identity_relative_difference", "require_t4_device", "accuracy_gate",
    } or gates["accuracy_gate"] is not None:
        raise ValueError("M21 gate fields mismatch or an accuracy gate was added")
    return config


def read_config(path: str | Path) -> dict:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def unit_plan(config: dict) -> list[dict]:
    """Fast FLY-CL units first; one accumulation unit per width and stream."""
    plan = [
        {"unit_id": f"gram_{method}", "part": "gram_load", "method": method}
        for method in config["gram_load"]["methods"]
    ]
    plan.extend(
        {"unit_id": unit["unit_id"], "part": "accumulation",
         "width": int(unit["width"]), "stream_id": unit["stream_id"]}
        for unit in config["accumulation"]["units"]
    )
    return plan


def _unit_path(output_dir: Path, unit_id: str) -> Path:
    return output_dir / "units" / f"{unit_id}.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda" else platform.processor(),
    }


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# part 1: minimal diagonal load for the direct INT8 Gram matrix (FLY-CL)
# ---------------------------------------------------------------------------
def read_priority3_config(config: dict) -> dict:
    path = ROOT / config["gram_load"]["priority3_config"]
    if _sha256(path, source=True) != config["gram_load"]["priority3_config_sha256"]:
        raise ValueError("M21 Priority-3 config SHA-256 mismatch")
    return p3._read_config(path)


def load_fly_stream(p3_config: dict, *, feature_cache_dir: Path, code_cache_dir: Path, device: torch.device):
    return p1._load_stream(
        config=p3._stream_config(p3_config),
        feature_cache_dir=feature_cache_dir,
        code_cache_dir=code_cache_dir,
        representation=p3_config["representation"],
        device_name=str(device),
    )


def evaluate_minimal_load(
    *, p3_config: dict, config: dict, train: dict, code_cache,
    training_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    """The Priority-3 compressed-control loop with the minimal-load learner."""
    code_indices, code_values, _, projection = code_cache
    policy = config["gram_load"]["minimal_load"]
    learner = MinimalLoadDirectInt8GramLearner(
        load_margin_multiplier=float(policy["load_margin_multiplier"]),
        load_grid_steps_per_doubling=int(policy["load_grid_steps_per_doubling"]),
        maximum_load_to_ridge_ratio=float(policy["maximum_load_to_ridge_ratio"]),
        error_chunk_size=int(policy["error_chunk_size"]),
        **p3._common_kwargs(p3_config, int(train["features"].shape[1]), projection, device),
    )
    ridge = float(p3_config["fly_ridge_lambda"])
    stage_accuracy, task_diagnostics = [], []
    started = time.perf_counter()
    for task, indices in enumerate(training_parts):
        codes = d0._dense_codes(
            code_indices[indices], code_values[indices], learner.expand_dim,
            device=device, dtype=learner.statistics_dtype,
        )
        m20._sync(device)
        update_started = time.perf_counter()
        learner.update_codes(codes, train["labels"][indices])
        m20._sync(device)
        update_seconds = time.perf_counter() - update_started
        del codes
        accuracy = d0._stage_code_accuracy(
            learner.weights, learner.class_ids, validation_parts, task,
            code_indices, code_values, train["labels"], learner.expand_dim,
            int(p3_config["representation"]["evaluation_batch_size"]),
        )
        stage_accuracy.append(accuracy)
        diagnostics = learner.diagnostics
        row = {
            "task": task + 1,
            "validation_accuracy": accuracy,
            "update_seconds": update_seconds,
            "persistent_state_bytes": learner.persistent_state_bytes(),
            "solver_relative_residual": float(diagnostics["solver_relative_residual"]),
            "diagonal_loading": float(diagnostics["diagonal_loading"]),
            "loading_to_ridge_ratio": float(diagnostics["diagonal_loading"]) / ridge,
            "grid_base_load": float(diagnostics["grid_base_load"]),
            "grid_index": diagnostics["grid_index"],
            "cholesky_attempts": int(diagnostics["cholesky_attempts"]),
            "zero_load_succeeded": bool(diagnostics["zero_load_succeeded"]),
            "local_quantization_error_infinity_bound": float(
                diagnostics["local_quantization_error_infinity_bound"]
            ),
            "relative_local_storage_error": float(diagnostics["relative_local_storage_error"]),
        }
        task_diagnostics.append(row)
        print(
            f"TASK method=direct_int8_gram_minimal_load {task + 1}/{len(training_parts)} "
            f"AA={accuracy:.4f} loading/lambda={row['loading_to_ridge_ratio']:.4g} "
            f"attempts={row['cholesky_attempts']} residual={row['solver_relative_residual']:.3e}",
            flush=True,
        )
        _release(device)
    return {
        "method": "direct_int8_gram_minimal_load",
        "status": "complete",
        "uses_test_set": False,
        "exemplar_free": True,
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy,
        "persistent_state_bytes": learner.persistent_state_bytes(),
        "maximum_solver_relative_residual": max(
            row["solver_relative_residual"] for row in task_diagnostics
        ),
        "total_update_seconds": sum(row["update_seconds"] for row in task_diagnostics),
        "analytic_and_validation_seconds": time.perf_counter() - started,
        "task_diagnostics": task_diagnostics,
    }


def evaluate_gram_load_method(
    method: str, *, p3_config: dict, config: dict, train: dict, code_cache,
    training_parts: list[torch.Tensor], validation_parts: list[torch.Tensor],
    device: torch.device,
) -> dict:
    common = {
        "config": p3_config, "train": train, "code_cache": code_cache,
        "training_parts": training_parts, "validation_parts": validation_parts,
        "device": device,
    }
    if method == "exact_fly_10000":
        return p3._evaluate_exact(**common)
    if method in {"srq_int8_p2b", "direct_int8_gram_weyl_repair"}:
        return p3._evaluate_compressed(method=method, **common)
    if method == "direct_int8_gram_minimal_load":
        return evaluate_minimal_load(
            p3_config=p3_config, config=config, train=train, code_cache=code_cache,
            training_parts=training_parts, validation_parts=validation_parts, device=device,
        )
    raise ValueError(f"unknown M21 gram_load method {method}")


def run_gram_load_unit(
    config: dict, unit: dict, *, p3_config: dict, stream, output_dir: Path,
    device: torch.device,
) -> dict:
    precision = m20._lock_precision()
    train, class_order, training_parts, validation_parts, code_cache = stream
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = evaluate_gram_load_method(
        unit["method"], p3_config=p3_config, config=config, train=train,
        code_cache=code_cache, training_parts=training_parts,
        validation_parts=validation_parts, device=device,
    )
    payload = {
        "schema_version": 1,
        "status": "complete" if result.get("status") == "complete" else "failed",
        "study_id": config["study_id"],
        "uses_test_set": False,
        "unit_id": unit["unit_id"],
        "part": "gram_load",
        "method": unit["method"],
        "ridge_lambda": float(p3_config["fly_ridge_lambda"]),
        "class_order": class_order,
        "result": result,
        "unit_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else None,
        "precision_lock": precision,
        "environment": _environment(device),
    }
    _write_json_atomic(_unit_path(output_dir, unit["unit_id"]), payload)
    _release(device)
    return payload


# ---------------------------------------------------------------------------
# part 2: recursive versus one-shot quantization (RanPAC)
# ---------------------------------------------------------------------------
def _stream_by_id(config: dict, stream_id: str) -> dict:
    for stream in config["accumulation"]["streams"]:
        if stream["stream_id"] == stream_id:
            return stream
    raise KeyError(stream_id)


def prepare_ranpac_stream(
    config: dict, stream: dict, labels: torch.Tensor, width: int, device: torch.device
) -> dict:
    """The M6/M20 stream: seeded class order and split, projection prefix."""
    accumulation = config["accumulation"]
    num_classes = int(config["num_classes"])
    class_order = random.Random(int(stream["class_order_seed"])).sample(
        list(range(num_classes)), num_classes
    )
    task_indices = split(labels, class_order, int(config["num_tasks"]))
    training_parts, validation_parts = train_validation_indices(
        labels, task_indices, int(stream["split_seed"]),
        float(accumulation["outer_validation_fraction"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(stream["projection_seed"]))
    full_projection = torch.randn(
        int(accumulation["ranpac"]["feature_dimension"]),
        int(accumulation["ranpac"]["maximum_expand_dimension"]),
        generator=generator, dtype=torch.float32,
    )
    projection = full_projection[:, :width].contiguous()
    return {
        "class_order": class_order,
        "training_parts": training_parts,
        "validation_parts": validation_parts,
        "full_projection": full_projection,
        "projection": projection.to(device),
    }


def stream_identity(prepared: dict, source: dict, width: int) -> dict:
    provenance = source["provenance"]
    return {
        "class_order": prepared["class_order"] == provenance["class_order"],
        "training_indices": m6._sequence_sha256(prepared["training_parts"])
        == provenance["training_indices_sha256"],
        "validation_indices": m6._sequence_sha256(prepared["validation_parts"])
        == provenance["outer_validation_indices_sha256"],
        "full_projection": m4._tensor_content_sha256(prepared["full_projection"])
        == provenance["full_projection_sha256"],
        "projection_prefix": m4._tensor_content_sha256(
            prepared["full_projection"][:, :width].contiguous()
        ) == provenance["projection_prefix_sha256"][str(width)],
    }


def system_probes(config: dict, width: int, device: torch.device) -> torch.Tensor:
    """The fixed M7 Rademacher probes, scaled to unit norm."""
    accumulation = config["accumulation"]
    generator = torch.Generator(device="cpu").manual_seed(int(accumulation["system_probe_seed"]))
    signs = torch.randint(
        0, 2,
        (int(accumulation["ranpac"]["maximum_expand_dimension"]), int(accumulation["system_probe_count"])),
        generator=generator, dtype=torch.int8,
    )
    probes = signs[:width].to(device=device, dtype=torch.float32)
    return probes.mul_(2).sub_(1).div_(width**0.5)


def relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """A true relative Frobenius error (no floor on the reference norm)."""
    numerator = float(torch.linalg.vector_norm((candidate - reference).to(torch.float64)).item())
    denominator = float(torch.linalg.vector_norm(reference.to(torch.float64)).item())
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def one_shot_int8(
    gram: torch.Tensor, cross: torch.Tensor, *, ridge: float, p2b: dict
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Quantize the exact factor of ``gram + ridge I`` once and solve with it.

    The factorization repeats ExactGramBackend's operations and the codec is
    the one SquareRootBackend applies, so at the first task the result equals
    the recursive arm.  Returns the dequantized factor, the weights, the local
    factor error and the solver residual.
    """
    system = gram.to(torch.float32).clone()
    system.diagonal().add_(ridge)
    symmetric = (system + system.T) * 0.5
    del system
    lower, info = torch.linalg.cholesky_ex(symmetric)
    del symmetric
    if int(info.max().item()) != 0:
        raise RuntimeError("one-shot exact factorization failed")
    upper = lower.T
    _, local_error = CompressedUpper.from_upper_inplace_streaming(
        upper,
        block_size=int(p2b["block_size"]),
        group_size=int(p2b["group_size"]),
        mode="int8",
        maximum_batched_blocks=int(p2b["quantization_batch_blocks"]),
    )
    work_cross = cross.to(torch.float32)
    intermediate = torch.linalg.solve_triangular(upper.T, work_cross, upper=False)
    weights = torch.linalg.solve_triangular(upper, intermediate, upper=True)
    residual = _relative_factor_residual(upper, weights, work_cross)
    return upper, weights, float(local_error), residual


def _seen_predictions(
    *, encoder, features: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor,
    batch_size: int, weights: dict[str, torch.Tensor], class_ids: list[int],
) -> dict:
    """Logits and predictions with M5's batching, so accuracies reproduce M6/M20."""
    mapping = torch.tensor(class_ids, dtype=torch.long)
    logits = {name: [] for name in weights}
    predictions = {name: [] for name in weights}
    targets = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        codes = encoder(features[batch])
        for name, matrix in weights.items():
            values = codes.to(dtype=matrix.dtype) @ matrix
            predictions[name].append(mapping[values.argmax(1).cpu()])
            logits[name].append(values.detach().cpu())
        targets.append(labels[batch].to(torch.long).cpu())
        del codes
    return {
        "logits": {name: torch.cat(parts) for name, parts in logits.items()},
        "predictions": {name: torch.cat(parts) for name, parts in predictions.items()},
        "targets": torch.cat(targets),
    }


def compare_to_exact(
    *, logits: torch.Tensor, prediction: torch.Tensor, exact_logits: torch.Tensor,
    exact_prediction: torch.Tensor,
) -> dict:
    top2 = torch.topk(exact_logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    sample_linf = (logits - exact_logits).abs().amax(dim=1)
    return {
        "relative_logit_error": relative_error(logits, exact_logits),
        "prediction_agreement": float((prediction == exact_prediction).float().mean().item()),
        "margin_certified_fraction": float((2.0 * sample_linf < margin).float().mean().item()),
    }


def run_accumulation_unit(
    config: dict, unit: dict, *, train: dict, source: dict | None, output_dir: Path,
    device: torch.device,
) -> dict:
    precision = m20._lock_precision()
    accumulation = config["accumulation"]
    width = int(unit["width"])
    ridge = float(accumulation["ridge_lambda"])
    p2b = accumulation["p2b"]
    stream = _stream_by_id(config, unit["stream_id"])
    prepared = prepare_ranpac_stream(config, stream, train["labels"], width, device)
    identity = (
        stream_identity(prepared, source, width) if stream["m6_identity_locked"] else None
    )
    projection = prepared["projection"]
    encoder = lambda values: torch.relu(values.to(device=device, dtype=torch.float32) @ projection)
    probes = system_probes(config, width, device)
    common = m5._common_backend(width, ridge, device)
    exact = ExactGramBackend(**common)
    recursive = SquareRootBackend(
        storage_mode="int8", **common, **m20._p2b_kwargs({"p2b": p2b}),
    )

    started = time.perf_counter()
    records = []
    for task_id, train_indices in enumerate(prepared["training_parts"]):
        codes = m5._encode_indices(
            encoder, train["features"], train_indices, int(accumulation["ranpac"]["encode_batch_size"])
        )
        task_labels = train["labels"][train_indices]
        exact.update(codes, task_labels)
        exact_action = exact.gram @ probes + ridge * probes

        one_shot_factor, one_shot_weights, one_shot_local_error, one_shot_residual = one_shot_int8(
            exact.gram, exact.Q, ridge=ridge, p2b=p2b
        )
        one_shot_action = one_shot_factor.T @ (one_shot_factor @ probes)
        del one_shot_factor
        _release(device)

        m20._sync(device)
        update_started = time.perf_counter()
        recursive.update(codes, task_labels)
        m20._sync(device)
        recursive_update_seconds = time.perf_counter() - update_started
        del codes
        factor = recursive.factor.reconstruct_upper(dtype=torch.float32)
        recursive_action = factor.T @ (factor @ probes)
        del factor
        _release(device)

        if list(recursive.class_ids) != list(exact.class_ids):
            raise AssertionError("M21 class columns differ between arms")
        seen = torch.cat(prepared["validation_parts"][: task_id + 1])
        evaluation = _seen_predictions(
            encoder=encoder, features=train["features"], labels=train["labels"],
            indices=seen, batch_size=int(accumulation["ranpac"]["evaluation_batch_size"]),
            weights={"exact": exact.weights, "recursive_int8": recursive.weights,
                     "one_shot_int8": one_shot_weights},
            class_ids=list(exact.class_ids),
        )
        targets = evaluation["targets"]
        accuracy = {
            name: 100.0 * float((prediction == targets).sum().item()) / len(targets)
            for name, prediction in evaluation["predictions"].items()
        }
        arms = {}
        for name, weights, action, local_error, residual in (
            ("recursive_int8", recursive.weights, recursive_action,
             float(recursive.diagnostics["relative_local_factor_error"]),
             float(recursive.diagnostics["solver_relative_residual"])),
            ("one_shot_int8", one_shot_weights, one_shot_action, one_shot_local_error,
             one_shot_residual),
        ):
            arms[name] = {
                "accuracy_percent": accuracy[name],
                "relative_classifier_error": relative_error(weights, exact.weights),
                "relative_system_action_error": relative_error(action, exact_action),
                "relative_local_factor_error": local_error,
                "solver_relative_residual": residual,
                **compare_to_exact(
                    logits=evaluation["logits"][name],
                    prediction=evaluation["predictions"][name],
                    exact_logits=evaluation["logits"]["exact"],
                    exact_prediction=evaluation["predictions"]["exact"],
                ),
            }
        identity_difference = (
            relative_error(one_shot_weights, recursive.weights) if task_id == 0 else None
        )
        exact_tensors = {"projection": projection, **exact.persistent_tensors()}
        recursive_tensors = {"projection": projection, **recursive.persistent_tensors()}
        record = {
            "task": task_id + 1,
            "exact": {
                "accuracy_percent": accuracy["exact"],
                "solver_relative_residual": float(exact.diagnostics["solver_relative_residual"]),
                "total_persistent_bytes": persistent_tensor_bytes(exact_tensors),
            },
            "recursive_int8": {
                **arms["recursive_int8"],
                "update_seconds": recursive_update_seconds,
                "total_persistent_bytes": persistent_tensor_bytes(recursive_tensors),
            },
            "one_shot_int8": arms["one_shot_int8"],
            "task1_one_shot_minus_recursive_relative_weight_difference": identity_difference,
        }
        records.append(record)
        print(
            f"M21 {unit['unit_id']} task={task_id + 1}/{config['num_tasks']} "
            f"acc exact={accuracy['exact']:.3f} rec={accuracy['recursive_int8']:.3f} "
            f"one={accuracy['one_shot_int8']:.3f} | logit err rec="
            f"{arms['recursive_int8']['relative_logit_error']:.4f} one="
            f"{arms['one_shot_int8']['relative_logit_error']:.4f}",
            flush=True,
        )
        del evaluation, one_shot_weights, exact_action, one_shot_action, recursive_action
        _release(device)

    def aia(name: str) -> float:
        return sum(record[name]["accuracy_percent"] for record in records) / len(records)

    payload = {
        "schema_version": 1,
        "status": "complete",
        "study_id": config["study_id"],
        "uses_test_set": False,
        "unit_id": unit["unit_id"],
        "part": "accumulation",
        "width": width,
        "stream_id": unit["stream_id"],
        "stream": stream,
        "m6_identity_checks": identity,
        "records": records,
        "validation_aia_percent": {name: aia(name) for name in ACCUMULATION_ARMS},
        "final_validation_accuracy_percent": {
            name: records[-1][name]["accuracy_percent"] for name in ACCUMULATION_ARMS
        },
        "final_total_persistent_bytes": {
            "exact": records[-1]["exact"]["total_persistent_bytes"],
            "recursive_int8": records[-1]["recursive_int8"]["total_persistent_bytes"],
        },
        "unit_seconds": time.perf_counter() - started,
        "precision_lock": precision,
        "environment": _environment(device),
    }
    _write_json_atomic(_unit_path(output_dir, unit["unit_id"]), payload)
    del exact, recursive
    _release(device)
    return payload


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


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.inf if numerator > 0 else 1.0
    return numerator / denominator


def summarize_gram_load(config: dict, by_id: dict) -> dict:
    gram = config["gram_load"]
    results = {
        method: by_id[f"gram_{method}"]["result"]
        for method in gram["methods"] if f"gram_{method}" in by_id
    }
    tolerance = float(config["gates"]["maximum_reproduction_accuracy_difference_pp"])
    reference = gram["priority3_reference"]["validation_average_accuracy"]
    reproduction = {
        method: (
            abs(results[method]["validation_average_accuracy"] - float(reference[method]))
            if method in results and results[method].get("status") == "complete" else None
        )
        for method in REPRODUCED_GRAM_LOAD_METHODS
    }
    reproduced = all(value is not None and value <= tolerance for value in reproduction.values())
    minimal = results.get("direct_int8_gram_minimal_load")
    srq = results.get("srq_int8_p2b")
    exact = results.get("exact_fly_10000")
    weyl = results.get("direct_int8_gram_weyl_repair")
    verdict = None
    observations = None
    search_ok = False
    if minimal and srq and exact and weyl and minimal.get("status") == "complete":
        rows = minimal["task_diagnostics"]
        search_ok = len(rows) == int(config["num_tasks"]) and all(
            row["cholesky_attempts"] >= 1 and row["diagonal_loading"] >= 0.0 for row in rows
        )
        gap = srq["validation_average_accuracy"] - minimal["validation_average_accuracy"]
        rule = gram["interpretation"]
        if gap >= float(rule["material_advantage_pp"]):
            verdict = "SQUARE_ROOT_ADVANTAGE_MATERIAL_AGAINST_MINIMAL_LOAD"
        elif gap > float(rule["practical_equivalence_pp"]):
            verdict = "SQUARE_ROOT_ADVANTAGE_MODEST_AGAINST_MINIMAL_LOAD"
        elif gap >= -float(rule["practical_equivalence_pp"]):
            verdict = "MINIMAL_LOAD_PRACTICALLY_EQUIVALENT_TO_SQUARE_ROOT"
        else:
            verdict = "MINIMAL_LOAD_OUTPERFORMS_SQUARE_ROOT"
        ridge = float(by_id["gram_direct_int8_gram_weyl_repair"]["ridge_lambda"])
        weyl_rows = weyl.get("task_diagnostics", [])
        observations = {
            "srq_minus_minimal_load_aia_pp": gap,
            "minimal_load_minus_exact_aia_pp": minimal["validation_average_accuracy"]
            - exact["validation_average_accuracy"],
            "weyl_repair_minus_exact_aia_pp": weyl["validation_average_accuracy"]
            - exact["validation_average_accuracy"],
            "srq_minus_exact_aia_pp": srq["validation_average_accuracy"]
            - exact["validation_average_accuracy"],
            "minimal_loading_to_ridge_ratio_by_task": [row["loading_to_ridge_ratio"] for row in rows],
            "weyl_loading_to_ridge_ratio_by_task": [
                row["diagonal_loading"] / ridge for row in weyl_rows
            ],
            "minimal_load_cholesky_attempts_by_task": [row["cholesky_attempts"] for row in rows],
            "minimal_load_maximum_solver_relative_residual": minimal["maximum_solver_relative_residual"],
            "minimal_load_state_bytes": minimal["persistent_state_bytes"],
            "srq_state_bytes": srq["persistent_state_bytes"],
            "weyl_state_bytes": weyl["persistent_state_bytes"],
            "minimal_load_total_update_seconds": minimal["total_update_seconds"],
            "srq_total_update_seconds": srq["total_update_seconds"],
        }
    residual_limit = float(config["gates"]["maximum_solver_relative_residual"])
    residual_ok = all(
        method in results and results[method].get("status") == "complete"
        and results[method]["maximum_solver_relative_residual"] <= residual_limit
        for method in REPRODUCED_GRAM_LOAD_METHODS
    )
    return {
        "priority3_reproduction_abs_difference_pp": reproduction,
        "reproduces_priority3": reproduced,
        "minimal_load_search_consistent": search_ok,
        "reproduced_arms_solver_residual_ok": residual_ok,
        "verdict": verdict,
        "observations": observations,
        "validation_aia_percent": {
            method: result.get("validation_average_accuracy") for method, result in results.items()
        },
    }


def _unit_accumulation_summary(config: dict, unit: dict) -> dict:
    records = unit["records"]
    first, last = records[0], records[-1]
    aia = unit["validation_aia_percent"]
    row = {
        "width": unit["width"],
        "stream_id": unit["stream_id"],
        "aia_loss_pp": {
            "recursive_int8": aia["exact"] - aia["recursive_int8"],
            "one_shot_int8": aia["exact"] - aia["one_shot_int8"],
        },
        "one_shot_minus_recursive_aia_pp": aia["one_shot_int8"] - aia["recursive_int8"],
        "final_ratio_recursive_over_one_shot": {},
        "growth_task10_over_task1": {},
        "final": {},
    }
    for metric in (
        "relative_logit_error", "relative_system_action_error", "relative_classifier_error",
        "relative_local_factor_error",
    ):
        row["final_ratio_recursive_over_one_shot"][metric] = _ratio(
            last["recursive_int8"][metric], last["one_shot_int8"][metric]
        )
        row["growth_task10_over_task1"][metric] = {
            arm: _ratio(last[arm][metric], first[arm][metric])
            for arm in ("recursive_int8", "one_shot_int8")
        }
    for metric in (
        "relative_logit_error", "relative_system_action_error", "relative_classifier_error",
        "prediction_agreement", "margin_certified_fraction",
    ):
        row["final"][metric] = {
            arm: last[arm][metric] for arm in ("recursive_int8", "one_shot_int8")
        }
    return row


def summarize_accumulation(config: dict, by_id: dict, units: list[dict]) -> dict:
    accumulation = config["accumulation"]
    gates = config["gates"]
    per_unit = {}
    reproduction = {}
    m7_differences = {}
    identity_ok = True
    task1_ok = True
    residual_ok = True
    for item in accumulation["units"]:
        unit = by_id.get(item["unit_id"])
        if unit is None:
            continue
        per_unit[item["unit_id"]] = _unit_accumulation_summary(config, unit)
        reference = accumulation["reference"][item["unit_id"]]
        reproduction[item["unit_id"]] = {
            "exact_aia_difference_pp": abs(unit["validation_aia_percent"]["exact"] - reference["exact_aia_percent"]),
            "exact_final_difference_pp": abs(unit["final_validation_accuracy_percent"]["exact"] - reference["exact_final_percent"]),
            "exact_bytes_equal": unit["final_total_persistent_bytes"]["exact"] == reference["exact_bytes"],
            "p2b_int8_aia_difference_pp": abs(unit["validation_aia_percent"]["recursive_int8"] - reference["p2b_int8_aia_percent"]),
            "p2b_int8_final_difference_pp": abs(unit["final_validation_accuracy_percent"]["recursive_int8"] - reference["p2b_int8_final_percent"]),
            "p2b_int8_bytes_equal": unit["final_total_persistent_bytes"]["recursive_int8"] == reference["p2b_int8_bytes"],
        }
        curve = reference["m7_p2b_int8_relative_logit_error"]
        if curve is not None:
            m7_differences[item["unit_id"]] = max(
                abs(record["recursive_int8"]["relative_logit_error"] - expected) / expected
                for record, expected in zip(unit["records"], curve)
            )
        stream = _stream_by_id(config, item["stream_id"])
        if stream["m6_identity_locked"]:
            identity_ok &= bool(unit["m6_identity_checks"]) and all(unit["m6_identity_checks"].values())
        difference = unit["records"][0]["task1_one_shot_minus_recursive_relative_weight_difference"]
        task1_ok &= difference is not None and difference <= float(gates["maximum_task1_identity_relative_difference"])
        residual_ok &= all(
            record[arm]["solver_relative_residual"] <= float(gates["maximum_solver_relative_residual"])
            for record in unit["records"] for arm in ACCUMULATION_ARMS
        )

    tolerance = float(gates["maximum_reproduction_accuracy_difference_pp"])
    reproduction_ok = len(reproduction) == len(accumulation["units"]) and all(
        row["exact_aia_difference_pp"] <= tolerance and row["exact_final_difference_pp"] <= tolerance
        and row["p2b_int8_aia_difference_pp"] <= tolerance
        and row["p2b_int8_final_difference_pp"] <= tolerance
        and row["exact_bytes_equal"] and row["p2b_int8_bytes_equal"]
        for row in reproduction.values()
    )
    m7_ok = all(
        value <= float(gates["maximum_m7_logit_error_relative_difference"])
        for value in m7_differences.values()
    ) and len(m7_differences) == sum(
        accumulation["reference"][item["unit_id"]]["m7_p2b_int8_relative_logit_error"] is not None
        for item in accumulation["units"]
    )

    rule = accumulation["interpretation"]
    primary = [
        per_unit[item["unit_id"]] for item in accumulation["units"]
        if int(item["width"]) == int(rule["primary_width"]) and item["unit_id"] in per_unit
    ]
    primary_complete = len(primary) == sum(
        int(item["width"]) == int(rule["primary_width"]) for item in accumulation["units"]
    )
    logit_verdict = None
    accuracy_verdict = None
    primary_summary = None
    if primary and primary_complete:
        ratios = [row["final_ratio_recursive_over_one_shot"]["relative_logit_error"] for row in primary]
        gaps = [row["one_shot_minus_recursive_aia_pp"] for row in primary]
        if min(ratios) >= float(rule["dominant_minimum_ratio"]):
            logit_verdict = "ACCUMULATION_DOMINATES_FINAL_LOGIT_ERROR"
        elif min(ratios) >= float(rule["contributes_minimum_ratio"]):
            logit_verdict = "ACCUMULATION_CONTRIBUTES_TO_FINAL_LOGIT_ERROR"
        else:
            logit_verdict = "ACCUMULATION_NOT_SUPPORTED_BY_FINAL_LOGIT_ERROR"
        if all(gap > 0 for gap in gaps):
            accuracy_verdict = "ONE_SHOT_HIGHER_AIA_ON_EVERY_STREAM"
        elif statistics.fmean(gaps) > 0:
            accuracy_verdict = "ONE_SHOT_HIGHER_MEAN_AIA_NOT_EVERY_STREAM"
        else:
            accuracy_verdict = "ACCUMULATION_ACCURACY_COST_NOT_SUPPORTED"
        primary_summary = {
            "width": int(rule["primary_width"]),
            "final_logit_error_ratio": _describe(ratios),
            "final_system_error_ratio": _describe([
                row["final_ratio_recursive_over_one_shot"]["relative_system_action_error"] for row in primary
            ]),
            "one_shot_minus_recursive_aia_pp": _describe(gaps),
            "recursive_aia_loss_pp": _describe([row["aia_loss_pp"]["recursive_int8"] for row in primary]),
            "one_shot_aia_loss_pp": _describe([row["aia_loss_pp"]["one_shot_int8"] for row in primary]),
            "recursive_logit_error_growth": _describe([
                row["growth_task10_over_task1"]["relative_logit_error"]["recursive_int8"] for row in primary
            ]),
            "one_shot_logit_error_growth": _describe([
                row["growth_task10_over_task1"]["relative_logit_error"]["one_shot_int8"] for row in primary
            ]),
        }
    return {
        "per_unit": per_unit,
        "reproduction": reproduction,
        "reproduces_m6_m20": reproduction_ok,
        "m7_recursive_logit_error_max_relative_difference": m7_differences,
        "reproduces_m7_logit_error": m7_ok,
        "m6_identity_for_locked_stream": identity_ok,
        "task1_one_shot_equals_recursive": task1_ok and bool(per_unit),
        "solver_residual_ok": residual_ok,
        "primary": primary_summary,
        "logit_error_verdict": logit_verdict,
        "accuracy_verdict": accuracy_verdict,
    }


def summarize(config: dict, units: list[dict], *, config_path: Path | None = None) -> dict:
    plan = unit_plan(config)
    by_id = {unit["unit_id"]: unit for unit in units}
    complete = len(by_id) == len(plan) and all(
        item["unit_id"] in by_id and by_id[item["unit_id"]].get("status") == "complete"
        for item in plan
    )
    gram = summarize_gram_load(config, by_id)
    accumulation = summarize_accumulation(config, by_id, units)
    required = str(config["required_device_name_substring"]).lower()
    device_ok = all(required in unit["environment"]["device_name"].lower() for unit in units)
    gates = {
        "all_units_complete": complete,
        "gram_load_reproduces_priority3": gram["reproduces_priority3"],
        "gram_load_reproduced_arms_solver_residual": gram["reproduced_arms_solver_residual_ok"],
        "minimal_load_search_consistent": gram["minimal_load_search_consistent"],
        "m6_identity_for_locked_stream": accumulation["m6_identity_for_locked_stream"],
        "accumulation_reproduces_m6_m20": accumulation["reproduces_m6_m20"],
        "recursive_logit_error_reproduces_m7": accumulation["reproduces_m7_logit_error"],
        "task1_one_shot_equals_recursive": accumulation["task1_one_shot_equals_recursive"],
        "accumulation_solver_residual": accumulation["solver_residual_ok"],
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
        "minimal_load_module_sha256": _sha256(
            ROOT / "methods/srq_fly_optimized/minimal_load_control.py", source=True
        ),
        "locked_backend_sha256": _sha256(ROOT / "methods/analytic_ridge/backends.py", source=True),
        "gram_load": gram,
        "accumulation": accumulation,
        "gates": gates,
        "units": units,
    }


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    m20.assert_train_only_cache(feature_cache_dir)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M21 requires a clean source checkout")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("M21 requested CUDA but CUDA is unavailable")

    plan = unit_plan(config)
    pending = [item for item in plan if not _unit_path(output_dir, item["unit_id"]).is_file()]
    if args.max_new_units is not None:
        pending = pending[: args.max_new_units]
    if pending:
        if _sha256(feature_cache_dir / "train.pt") != config["train_cache_sha256"]:
            raise ValueError("M21 train cache SHA-256 mismatch")
    for item in pending:
        print(f"M21 START UNIT {item['unit_id']}", flush=True)
        if item["part"] == "gram_load":
            p3_config = read_priority3_config(config)
            stream = load_fly_stream(
                p3_config, feature_cache_dir=feature_cache_dir,
                code_cache_dir=Path(args.code_cache_dir).resolve(), device=device,
            )
            run_gram_load_unit(config, item, p3_config=p3_config, stream=stream,
                               output_dir=output_dir, device=device)
            del stream
        else:
            source = m11._load_source(
                {"source_m6": config["accumulation"]["source_m6"]},
                Path(args.source_m6_artifact).resolve(),
            )
            train, _, metadata = validate_cache(
                feature_cache_dir,
                argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
                load_test=False,
            )
            if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
                raise ValueError("M21 feature-cache checkpoint SHA-256 mismatch")
            run_accumulation_unit(config, item, train=train, source=source,
                                  output_dir=output_dir, device=device)
            del train
        _release(device)
        print(f"M21 CHECKPOINT {item['unit_id']}: COMPLETE", flush=True)

    units = [
        json.loads(_unit_path(output_dir, item["unit_id"]).read_text(encoding="utf-8"))
        for item in plan if _unit_path(output_dir, item["unit_id"]).is_file()
    ]
    progress = {"study_id": config["study_id"], "completed_units": len(units), "total_units": len(plan)}
    _write_json_atomic(output_dir / "m21_progress.json", progress)
    if len(units) == len(plan):
        result = summarize(config, units, config_path=config_path)
        _write_json_atomic(output_dir / "m21_results.json", result)
        print("M21 STATUS:", result["status"], flush=True)
        print("M21 GATES:", json.dumps(result["gates"], indent=2), flush=True)
        print("M21 GRAM-LOAD VERDICT:", result["gram_load"]["verdict"], flush=True)
        print("M21 ACCUMULATION LOGIT VERDICT:", result["accumulation"]["logit_error_verdict"], flush=True)
        print("M21 ACCUMULATION ACCURACY VERDICT:", result["accumulation"]["accuracy_verdict"], flush=True)
    else:
        print(f"M21 PROGRESS: {len(units)}/{len(plan)} units", flush=True)
    return progress


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--code-cache-dir", required=True)
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
