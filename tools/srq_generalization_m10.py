"""Controlled train-only GACL adapter gate for the generic SRQ backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import (  # noqa: E402
    DenseSquareRootBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from methods.frontends import (  # noqa: E402
    GACLAnalyticLearner,
    GACLInverseRLSReference,
    gacl_linear_projection,
)
from tools.experiment_runner import validate_cache  # noqa: E402
from tools.twa_fly_pilot import _sequence_sha256, _sha256_file, _tensor_content_sha256  # noqa: E402


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
    "statistics_dtype",
    "solver_dtype",
    "development_subset",
    "stream",
    "gacl",
    "p2b",
    "gates",
}
SUBSET_KEYS = {
    "fit_samples_per_class",
    "validation_samples_per_class",
    "selection_seed_offset",
}
STREAM_KEYS = {
    "semantics",
    "num_tasks",
    "disjoint_class_ratio_percent",
    "blurry_sample_ratio_percent",
    "varying_nm",
    "mini_batch_size",
    "online_iter",
    "evaluation",
}
GACL_KEYS = {
    "upstream_repository",
    "upstream_commit",
    "upstream_gacl_py_sha256",
    "upstream_sampler_sha256",
    "upstream_model_sha256",
    "upstream_script_sha256",
    "scope",
    "expansion_dim",
    "projection_initializer",
    "activation",
    "projection_seed",
    "gamma",
    "encode_batch_size",
}
P2B_KEYS = {
    "block_size",
    "group_size",
    "update_panel_size",
    "update_trailing_chunk_size",
    "first_update_backend",
    "quantization_backend",
    "quantization_batch_blocks",
}
GATE_KEYS = {
    "maximum_inverse_primal_weight_relative_error",
    "maximum_inverse_primal_logit_relative_error",
    "minimum_inverse_primal_prediction_agreement",
    "minimum_p2b_total_state_reduction_fraction",
    "maximum_p2b_validation_aia_loss_pp",
    "maximum_solver_relative_residual",
}


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M10 config keys/schema mismatch")
    if config["uses_test_set"] is not False:
        raise ValueError("M10 must remain train-only")
    if config["accuracy_based_selection"] is not False:
        raise ValueError("M10 cannot select a method using outer accuracy")
    if config["seed"] != 2025:
        raise ValueError("M10 locks seed 2025")
    if config["num_classes"] <= 1:
        raise ValueError("invalid class count")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("M10 controlled run locks FP32 state and solves")

    subset = config["development_subset"]
    if set(subset) != SUBSET_KEYS or min(
        subset["fit_samples_per_class"], subset["validation_samples_per_class"]
    ) <= 0:
        raise ValueError("invalid M10 development subset")

    stream = config["stream"]
    if set(stream) != STREAM_KEYS:
        raise ValueError("M10 stream fields mismatch")
    if (
        stream["semantics"] != "controlled_fixed_si_blurry_port"
        or stream["varying_nm"] is not False
        or stream["online_iter"] != 1
        or stream["evaluation"] != "task_boundary_seen_class_train_validation"
        or stream["num_tasks"] <= 1
        or stream["mini_batch_size"] <= 0
        or not 0 <= stream["disjoint_class_ratio_percent"] <= 100
        or not 0 <= stream["blurry_sample_ratio_percent"] <= 100
    ):
        raise ValueError("invalid controlled Si-Blurry semantics")

    gacl = config["gacl"]
    if set(gacl) != GACL_KEYS:
        raise ValueError("M10 GACL fields mismatch")
    if (
        gacl["scope"] != "cached_vit_b_features_not_official_deit_checkpoint"
        or gacl["projection_initializer"]
        != "pytorch_linear_default_kaiming_uniform"
        or gacl["activation"] != "relu"
        or gacl["projection_seed"] != 2025
        or gacl["gamma"] <= 0
        or min(gacl[name] for name in ("expansion_dim", "encode_batch_size")) <= 0
        or len(gacl["upstream_commit"]) != 40
        or any(
            len(gacl[name]) != 64
            for name in (
                "upstream_gacl_py_sha256",
                "upstream_sampler_sha256",
                "upstream_model_sha256",
                "upstream_script_sha256",
            )
        )
    ):
        raise ValueError("invalid locked GACL control")

    p2b = config["p2b"]
    if set(p2b) != P2B_KEYS:
        raise ValueError("M10 P2B fields mismatch")
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
        raise ValueError("M10 no longer matches frozen P2B")
    if p2b["update_trailing_chunk_size"] is not None and p2b[
        "update_trailing_chunk_size"
    ] <= 0:
        raise ValueError("invalid P2B trailing chunk size")

    gates = config["gates"]
    if set(gates) != GATE_KEYS or min(map(float, gates.values())) < 0:
        raise ValueError("invalid M10 gates")
    for key in (
        "minimum_inverse_primal_prediction_agreement",
        "minimum_p2b_total_state_reduction_fraction",
    ):
        if not 0 <= float(gates[key]) <= 1:
            raise ValueError(f"invalid fraction gate: {key}")
    return config


def development_subset_indices(
    labels: torch.Tensor,
    *,
    num_classes: int,
    fit_per_class: int,
    validation_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose disjoint per-class fit/validation subsets without test data."""

    labels = labels.cpu().to(torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    fit_parts, validation_parts = [], []
    required = int(fit_per_class) + int(validation_per_class)
    for class_id in range(int(num_classes)):
        candidates = torch.where(labels == class_id)[0]
        if len(candidates) < required:
            raise ValueError(f"class {class_id} lacks M10 development samples")
        order = torch.randperm(len(candidates), generator=generator)
        selected = candidates[order[:required]]
        fit_parts.append(selected[:fit_per_class])
        validation_parts.append(selected[fit_per_class:])
    return torch.cat(fit_parts).sort().values, torch.cat(validation_parts).sort().values


def fixed_si_blurry_tasks(
    labels: torch.Tensor,
    allowed_indices: torch.Tensor,
    *,
    num_tasks: int,
    disjoint_ratio_percent: int,
    blurry_sample_ratio_percent: int,
    seed: int,
) -> tuple[list[torch.Tensor], dict]:
    """Port the fixed-N/M branch of the upstream GACL OnlineSampler.

    The function intentionally rejects a remainder in the redistributed pool;
    upstream silently drops that remainder. The locked M10 sample counts are
    divisible, so M10 keeps every selected training example exactly once.
    """

    labels = labels.cpu().to(torch.long)
    allowed = allowed_indices.cpu().to(torch.long).sort().values
    classes = sorted(map(int, torch.unique(labels[allowed]).tolist()))
    if classes != list(range(len(classes))):
        raise ValueError("M10 expects contiguous class IDs")
    if len(classes) % num_tasks:
        raise ValueError("class count must be divisible by task count")
    disjoint_num = len(classes) * int(disjoint_ratio_percent) // 100
    disjoint_num = (disjoint_num // num_tasks) * num_tasks
    blurry_num = len(classes) - disjoint_num
    blurry_num = (blurry_num // num_tasks) * num_tasks
    if disjoint_num + blurry_num != len(classes):
        raise ValueError("locked stream would omit classes")

    generator = torch.Generator().manual_seed(int(seed))
    class_order = torch.randperm(len(classes), generator=generator)
    disjoint_classes = class_order[:disjoint_num].reshape(num_tasks, -1).tolist()
    blurry_classes = class_order[
        disjoint_num : disjoint_num + blurry_num
    ].reshape(num_tasks, -1).tolist()
    disjoint_owner = {
        class_id: task_id
        for task_id, task_classes in enumerate(disjoint_classes)
        for class_id in task_classes
    }
    blurry_owner = {
        class_id: task_id
        for task_id, task_classes in enumerate(blurry_classes)
        for class_id in task_classes
    }
    disjoint_indices = [[] for _ in range(num_tasks)]
    blurry_indices = [[] for _ in range(num_tasks)]
    for index in allowed.tolist():
        class_id = int(labels[index])
        if class_id in disjoint_owner:
            disjoint_indices[disjoint_owner[class_id]].append(index)
        else:
            blurry_indices[blurry_owner[class_id]].append(index)

    blurred: list[int] = []
    for task_id in range(num_tasks):
        count = len(blurry_indices[task_id]) * int(blurry_sample_ratio_percent) // 100
        blurred.extend(blurry_indices[task_id][:count])
        blurry_indices[task_id] = blurry_indices[task_id][count:]
    if len(blurred) % num_tasks:
        raise ValueError("M10 refuses the upstream sampler's dropped remainder")
    if blurred:
        shuffled = torch.tensor(blurred, dtype=torch.long)
        shuffled = shuffled[torch.randperm(len(shuffled), generator=generator)]
        redistributed = shuffled.tolist()
    else:
        redistributed = []
    per_task = len(redistributed) // num_tasks
    for task_id in range(num_tasks):
        start = task_id * per_task
        blurry_indices[task_id].extend(redistributed[start : start + per_task])

    task_indices: list[torch.Tensor] = []
    for task_id in range(num_tasks):
        combined = torch.tensor(
            disjoint_indices[task_id] + blurry_indices[task_id], dtype=torch.long
        )
        combined = combined[torch.randperm(len(combined), generator=generator)]
        task_indices.append(combined)

    flattened = torch.cat(task_indices)
    if len(flattened) != len(allowed) or set(flattened.tolist()) != set(allowed.tolist()):
        raise AssertionError("Si-Blurry port did not partition the selected samples")
    task_class_sets = [set(map(int, labels[part].tolist())) for part in task_indices]
    repeated_classes = set()
    for left in range(num_tasks):
        for right in range(left + 1, num_tasks):
            repeated_classes.update(task_class_sets[left] & task_class_sets[right])
    metadata = {
        "class_order": class_order.tolist(),
        "disjoint_classes": disjoint_classes,
        "blurry_classes": blurry_classes,
        "task_sample_counts": [len(part) for part in task_indices],
        "task_class_counts": [len(values) for values in task_class_sets],
        "redistributed_sample_count": len(redistributed),
        "repeated_class_count": len(repeated_classes),
        "partition_exact": True,
    }
    return task_indices, metadata


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(expected).item()), 1.0)
    return float(torch.linalg.vector_norm(actual - expected).item()) / denominator


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_update(owner, codes: torch.Tensor, labels: torch.Tensor, device: torch.device) -> float:
    _synchronize(device)
    started = time.perf_counter()
    owner.update_codes(codes, labels)
    _synchronize(device)
    return time.perf_counter() - started


def _encode_selected(
    projection: torch.Tensor,
    features: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(indices), batch_size):
        batch = features[indices[start : start + batch_size]].to(
            device=device, dtype=projection.dtype
        )
        parts.append(torch.relu(batch @ projection))
    return torch.cat(parts)


def _quadratic_bytes(backend) -> int:
    tensors = backend.persistent_tensors()
    names = (
        ["factor"]
        if "factor" in tensors
        else [name for name in tensors if name.startswith("factor_")]
    )
    return persistent_tensor_bytes({name: tensors[name] for name in names})


def _learner_state(learner: GACLAnalyticLearner) -> dict:
    return {
        "total_persistent_bytes": learner.persistent_state_bytes(),
        "backend_persistent_bytes": learner.backend.persistent_state_bytes(),
        "quadratic_persistent_bytes": _quadratic_bytes(learner.backend),
        "solver_relative_residual": float(
            learner.backend.diagnostics["solver_relative_residual"]
        ),
    }


def _evaluate(
    *,
    reference: GACLInverseRLSReference,
    learners: dict[str, GACLAnalyticLearner],
    codes: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    if any(learner.class_ids != reference.class_ids for learner in learners.values()):
        raise AssertionError("GACL class-column alignment diverged")
    class_to_column = {
        class_id: column for column, class_id in enumerate(reference.class_ids)
    }
    targets = torch.tensor(
        [class_to_column[int(value)] for value in labels.cpu().tolist()],
        device=codes.device,
        dtype=torch.long,
    )
    logits = {"reference_inverse_rls": reference.predict_logits_from_codes(codes)}
    logits.update(
        {name: learner.predict_logits_from_codes(codes) for name, learner in learners.items()}
    )
    predictions = {name: value.argmax(1) for name, value in logits.items()}
    accuracy = {
        name: 100.0 * float((prediction == targets).float().mean().item())
        for name, prediction in predictions.items()
    }
    return {
        "sample_count": len(labels),
        "seen_class_count": len(reference.class_ids),
        "accuracy_percent": accuracy,
        "inverse_primal_relative_logit_error": _relative_error(
            logits["reference_inverse_rls"], logits["fp32_square_root"]
        ),
        "inverse_primal_prediction_agreement": float(
            (
                predictions["reference_inverse_rls"]
                == predictions["fp32_square_root"]
            ).float().mean().item()
        ),
        "fp16_primal_relative_logit_error": _relative_error(
            logits["fp16_square_root"], logits["fp32_square_root"]
        ),
        "p2b_primal_relative_logit_error": _relative_error(
            logits["p2b_int8"], logits["fp32_square_root"]
        ),
        "p2b_primal_prediction_agreement": float(
            (
                predictions["p2b_int8"] == predictions["fp32_square_root"]
            ).float().mean().item()
        ),
    }


def _run_stream(
    *,
    config: dict,
    features: torch.Tensor,
    labels: torch.Tensor,
    fit_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    task_indices: list[torch.Tensor],
    stream_metadata: dict,
    device: torch.device,
) -> dict:
    gacl = config["gacl"]
    p2b = config["p2b"]
    dimension = int(gacl["expansion_dim"])
    gamma = float(gacl["gamma"])
    projection = gacl_linear_projection(
        int(features.shape[1]),
        dimension,
        seed=int(gacl["projection_seed"]),
        dtype=torch.float32,
        device=device,
    )

    def frontend(backend) -> GACLAnalyticLearner:
        return GACLAnalyticLearner(
            feature_dim=int(features.shape[1]),
            backend=backend,
            seed=int(gacl["projection_seed"]),
            projection=projection,
        )

    common = dict(
        dimension=dimension,
        ridge_lambda=gamma,
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
    )
    learners = {
        "fp32_square_root": frontend(
            DenseSquareRootBackend(
                **common,
                update_backend="blocked_qr",
                update_panel_size=int(p2b["update_panel_size"]),
                update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
            )
        ),
        "fp16_square_root": frontend(
            SquareRootBackend(**common, storage_mode="float16", **p2b)
        ),
        "p2b_int8": frontend(
            SquareRootBackend(**common, storage_mode="int8", **p2b)
        ),
    }
    reference = GACLInverseRLSReference(
        dimension=dimension,
        gamma=gamma,
        device=device,
        dtype=torch.float32,
    )

    selected = torch.cat((fit_indices, validation_indices)).sort().values
    codes = _encode_selected(
        projection,
        features,
        selected,
        batch_size=int(gacl["encode_batch_size"]),
        device=device,
    )
    positions = torch.full((len(labels),), -1, dtype=torch.long)
    positions[selected] = torch.arange(len(selected))
    validation_positions = positions[validation_indices]
    if bool((validation_positions < 0).any()):
        raise AssertionError("validation code lookup failed")

    update_seconds = {
        "reference_inverse_rls": 0.0,
        **{name: 0.0 for name in learners},
    }
    maximum_inverse_primal_weight_error = 0.0
    mixed_old_new_batches = 0
    repeated_only_batches = 0
    total_updates = 0
    seen_before: set[int] = set()
    records = []
    mini_batch_size = int(config["stream"]["mini_batch_size"])

    for task_id, indices in enumerate(task_indices):
        task_updates = 0
        task_new_classes: set[int] = set()
        task_exposed_classes: set[int] = set()
        for start in range(0, len(indices), mini_batch_size):
            batch_indices = indices[start : start + mini_batch_size]
            batch_positions = positions[batch_indices]
            if bool((batch_positions < 0).any()):
                raise AssertionError("training code lookup failed")
            batch_codes = codes[batch_positions.to(codes.device)]
            batch_labels = labels[batch_indices].to(device=device, dtype=torch.long)
            batch_classes = set(map(int, batch_labels.cpu().tolist()))
            old = batch_classes & seen_before
            new = batch_classes - seen_before
            task_exposed_classes.update(old)
            task_new_classes.update(new)
            if old and new:
                mixed_old_new_batches += 1
            if old and not new:
                repeated_only_batches += 1

            update_seconds["reference_inverse_rls"] += _timed_update(
                reference, batch_codes, batch_labels, device
            )
            for name, learner in learners.items():
                update_seconds[name] += _timed_update(
                    learner, batch_codes, batch_labels, device
                )
            seen_before.update(batch_classes)
            total_updates += 1
            task_updates += 1
            inverse_primal_error = _relative_error(
                reference.weights, learners["fp32_square_root"].weights
            )
            maximum_inverse_primal_weight_error = max(
                maximum_inverse_primal_weight_error, inverse_primal_error
            )

        seen_tensor = torch.tensor(reference.class_ids, dtype=torch.long)
        mask = torch.isin(labels[validation_indices].cpu(), seen_tensor)
        task_validation_indices = validation_indices[mask]
        task_validation_positions = validation_positions[mask]
        evaluation = _evaluate(
            reference=reference,
            learners=learners,
            codes=codes[task_validation_positions.to(codes.device)],
            labels=labels[task_validation_indices],
        )
        states = {name: _learner_state(learner) for name, learner in learners.items()}
        reference_state = {
            "backend_persistent_bytes": reference.persistent_state_bytes(),
            "total_persistent_bytes": reference.persistent_state_bytes()
            + persistent_tensor_bytes({"projection": projection}),
            "inverse_symmetry_relative_error": float(
                reference.diagnostics["inverse_symmetry_relative_error"]
            ),
        }
        records.append(
            {
                "task_id": task_id + 1,
                "sample_count": len(indices),
                "mini_batch_updates": task_updates,
                "new_class_count_within_task": len(task_new_classes),
                "previously_exposed_class_count_within_task": len(task_exposed_classes),
                "evaluation": evaluation,
                "reference_state": reference_state,
                "state": states,
            }
        )
        print(
            f"TASK {task_id + 1}/{len(task_indices)} updates={task_updates} "
            f"seen={evaluation['seen_class_count']} "
            f"primal={evaluation['accuracy_percent']['fp32_square_root']:.4f} "
            f"p2b={evaluation['accuracy_percent']['p2b_int8']:.4f}",
            flush=True,
        )

    method_names = ["reference_inverse_rls", *learners]
    aia = {
        name: sum(record["evaluation"]["accuracy_percent"][name] for record in records)
        / len(records)
        for name in method_names
    }
    final_accuracy = {
        name: records[-1]["evaluation"]["accuracy_percent"][name]
        for name in method_names
    }
    primal_total = records[-1]["state"]["fp32_square_root"][
        "total_persistent_bytes"
    ]
    p2b_total = records[-1]["state"]["p2b_int8"]["total_persistent_bytes"]
    state_reduction = 1.0 - p2b_total / primal_total
    maximum_solver_residual = max(
        record["state"][name]["solver_relative_residual"]
        for record in records
        for name in learners
    )
    summary = {
        "validation_aia_percent": aia,
        "final_validation_accuracy_percent": final_accuracy,
        "maximum_inverse_primal_weight_relative_error": (
            maximum_inverse_primal_weight_error
        ),
        "maximum_inverse_primal_logit_relative_error": max(
            record["evaluation"]["inverse_primal_relative_logit_error"]
            for record in records
        ),
        "minimum_inverse_primal_prediction_agreement": min(
            record["evaluation"]["inverse_primal_prediction_agreement"]
            for record in records
        ),
        "p2b_validation_aia_loss_pp": aia["fp32_square_root"] - aia["p2b_int8"],
        "p2b_total_state_reduction_fraction": state_reduction,
        "final_persistent_state": {
            "reference_inverse_rls": records[-1]["reference_state"],
            **records[-1]["state"],
        },
        "update_seconds": update_seconds,
        "total_mini_batch_updates": total_updates,
        "mixed_exposed_unexposed_mini_batches": mixed_old_new_batches,
        "repeated_only_mini_batches": repeated_only_batches,
        "maximum_solver_relative_residual": maximum_solver_residual,
    }
    thresholds = config["gates"]
    gates = {
        "stream_partition_exact": bool(stream_metadata["partition_exact"]),
        "generalized_class_reappearance": stream_metadata["repeated_class_count"] > 0,
        "mixed_exposed_unexposed_minibatch": mixed_old_new_batches > 0,
        "matched_minibatch_update_frequency": (
            reference.update_count == total_updates
            and all(
                learner.backend.total_rows == reference.total_rows
                for learner in learners.values()
            )
        ),
        "inverse_primal_weight_equivalence": maximum_inverse_primal_weight_error
        <= thresholds["maximum_inverse_primal_weight_relative_error"],
        "inverse_primal_logit_equivalence": summary[
            "maximum_inverse_primal_logit_relative_error"
        ]
        <= thresholds["maximum_inverse_primal_logit_relative_error"],
        "inverse_primal_prediction_agreement": summary[
            "minimum_inverse_primal_prediction_agreement"
        ]
        >= thresholds["minimum_inverse_primal_prediction_agreement"],
        "p2b_total_state_reduction": state_reduction
        >= thresholds["minimum_p2b_total_state_reduction_fraction"],
        "p2b_validation_aia_retention": summary["p2b_validation_aia_loss_pp"]
        <= thresholds["maximum_p2b_validation_aia_loss_pp"],
        "solver_residual": maximum_solver_residual
        <= thresholds["maximum_solver_relative_residual"],
    }
    return {
        "projection_sha256": _tensor_content_sha256(projection),
        "records": records,
        "summary": summary,
        "gates": gates,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M10 requires a clean source checkout")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M10 refuses a visible test.pt")
    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(
        range(config["num_classes"])
    ):
        raise ValueError("training labels do not match locked classes")

    subset = config["development_subset"]
    fit_indices, validation_indices = development_subset_indices(
        train["labels"],
        num_classes=int(config["num_classes"]),
        fit_per_class=int(subset["fit_samples_per_class"]),
        validation_per_class=int(subset["validation_samples_per_class"]),
        seed=int(config["seed"]) + int(subset["selection_seed_offset"]),
    )
    stream = config["stream"]
    task_indices, stream_metadata = fixed_si_blurry_tasks(
        train["labels"],
        fit_indices,
        num_tasks=int(stream["num_tasks"]),
        disjoint_ratio_percent=int(stream["disjoint_class_ratio_percent"]),
        blurry_sample_ratio_percent=int(stream["blurry_sample_ratio_percent"]),
        seed=int(config["seed"]),
    )
    result = _run_stream(
        config=config,
        features=train["features"],
        labels=train["labels"],
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        task_indices=task_indices,
        stream_metadata=stream_metadata,
        device=torch.device(args.device),
    )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    provenance = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_sha256": _sha256_file(config_path),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "generic_backend_sha256": _sha256_file(ROOT / "methods/analytic_ridge/backends.py"),
        "gacl_frontend_sha256": _sha256_file(ROOT / "methods/frontends/gacl.py"),
        "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
        "fit_indices_sha256": _sequence_sha256([fit_indices]),
        "validation_indices_sha256": _sequence_sha256([validation_indices]),
        "task_indices_sha256": _sequence_sha256(task_indices),
        "upstream_repository": config["gacl"]["upstream_repository"],
        "upstream_commit": config["gacl"]["upstream_commit"],
        "upstream_source_hashes": {
            key: config["gacl"][key]
            for key in (
                "upstream_gacl_py_sha256",
                "upstream_sampler_sha256",
                "upstream_model_sha256",
                "upstream_script_sha256",
            )
        },
    }
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M10_GACL_CONTROLLED_TRAIN_ONLY"
        if all(result["gates"].values())
        else "FAIL_M10_GACL_CONTROLLED_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "adapter": "GACL inverse-RLS equation to primal additive-Ridge backend",
            "generalized_stream": "fixed-N/M Si-Blurry port with repeated and new classes",
            "official_end_to_end_reproduction": False,
            "backbone": "cached ViT-B/16 features, not upstream DeiT-small-384 checkpoint",
            "augmentation_reproduced": False,
            "online_iter": int(stream["online_iter"]),
            "compression_frequency": "every mini-batch for every backend",
        },
        "stream": stream_metadata,
        "provenance": provenance,
        **result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "m10_results.json.tmp"
    destination = output_dir / "m10_results.json"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
    if payload["status"] != "PASS_M10_GACL_CONTROLLED_TRAIN_ONLY":
        raise RuntimeError("M10 controlled GACL gate failed; do not relax gates")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
