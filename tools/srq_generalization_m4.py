"""Train-only RanPAC Phase-2 gate for the reusable analytic Ridge backend."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from methods.analytic_ridge import (
    DenseSquareRootBackend,
    ExactGramBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from methods.frontends import RanPACAnalyticLearner
from tools.experiment_runner import split, train_validation_indices, validate_cache
from tools.twa_fly_pilot import _sequence_sha256, _sha256_file, _tensor_content_sha256


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
    "ranpac",
    "ridge_selection",
    "p2b",
    "gates",
}
RANPAC_KEYS = {
    "upstream_repository",
    "upstream_commit",
    "upstream_ranpac_py_sha256",
    "path",
    "expand_dim",
    "projection_distribution",
    "activation",
    "projection_seed",
    "encode_batch_size",
    "evaluation_batch_size",
}
SELECTION_KEYS = {
    "policy",
    "candidate_lambdas",
    "fit_samples_per_class",
    "validation_samples_per_class",
    "metric",
    "tie_break",
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
    "require_reference_exact_tensor_identity",
    "maximum_fp32_system_relative_error",
    "maximum_fp32_weight_relative_error",
    "maximum_fp32_logit_relative_error",
    "minimum_fp32_prediction_agreement",
    "minimum_srq_quadratic_state_reduction_fraction",
    "maximum_srq_validation_aia_loss_pp",
    "maximum_solver_relative_residual",
}


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M4 config keys/schema mismatch")
    if config["uses_test_set"] is not False:
        raise ValueError("M4 must remain train-only")
    if config["accuracy_based_selection"] is not False:
        raise ValueError("M4 cannot select a method using outer accuracy")
    if config["seed"] != 2025:
        raise ValueError("new protocols require seed 2025")
    if (
        config["num_classes"] <= 1
        or config["num_tasks"] <= 0
        or config["num_classes"] % config["num_tasks"]
        or not 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("invalid M4 class/task split")
    if config["statistics_dtype"] != "float32" or config["solver_dtype"] != "float32":
        raise ValueError("M4 locks FP32 statistics and solves")

    ranpac = config["ranpac"]
    if set(ranpac) != RANPAC_KEYS:
        raise ValueError("M4 RanPAC fields mismatch")
    if (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or ranpac["projection_seed"] != 2025
        or min(
            ranpac[name]
            for name in ("expand_dim", "encode_batch_size", "evaluation_batch_size")
        )
        <= 0
        or len(ranpac["upstream_commit"]) != 40
        or len(ranpac["upstream_ranpac_py_sha256"]) != 64
    ):
        raise ValueError("invalid locked RanPAC semantics")

    selection = config["ridge_selection"]
    if set(selection) != SELECTION_KEYS:
        raise ValueError("M4 Ridge-selection fields mismatch")
    lambdas = list(map(float, selection["candidate_lambdas"]))
    if (
        selection["policy"] != "single_fixed_train_only_calibration"
        or selection["metric"] != "mean_squared_error"
        or selection["tie_break"] != "smallest_lambda"
        or not lambdas
        or lambdas != sorted(set(lambdas))
        or min(lambdas) <= 0
        or min(
            selection[name]
            for name in ("fit_samples_per_class", "validation_samples_per_class")
        )
        <= 0
    ):
        raise ValueError("invalid fixed train-only Ridge selection")

    p2b = config["p2b"]
    if set(p2b) != P2B_KEYS:
        raise ValueError("M4 P2B fields mismatch")
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
        raise ValueError("M4 no longer matches frozen P2B")
    if p2b["update_trailing_chunk_size"] is not None and p2b[
        "update_trailing_chunk_size"
    ] <= 0:
        raise ValueError("invalid trailing chunk size")

    gates = config["gates"]
    if set(gates) != GATE_KEYS:
        raise ValueError("M4 gate fields mismatch")
    if gates["require_reference_exact_tensor_identity"] is not True:
        raise ValueError("M4 requires reference/generic Exact identity")
    if not 0 <= gates["minimum_fp32_prediction_agreement"] <= 1:
        raise ValueError("invalid prediction agreement gate")
    if not 0 <= gates["minimum_srq_quadratic_state_reduction_fraction"] < 1:
        raise ValueError("invalid state-reduction gate")
    if min(
        gates[name]
        for name in (
            "maximum_fp32_system_relative_error",
            "maximum_fp32_weight_relative_error",
            "maximum_fp32_logit_relative_error",
            "maximum_srq_validation_aia_loss_pp",
            "maximum_solver_relative_residual",
        )
    ) < 0:
        raise ValueError("invalid nonnegative M4 tolerance")
    return config


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(expected).item()), 1.0)
    return float(torch.linalg.vector_norm(actual - expected).item()) / denominator


def _calibration_indices(
    labels: torch.Tensor,
    outer_training_parts: list[torch.Tensor],
    *,
    seed: int,
    fit_per_class: int,
    validation_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    allowed = torch.cat(outer_training_parts).cpu()
    allowed_labels = labels[allowed].cpu()
    generator = torch.Generator().manual_seed(seed + 404)
    fit_parts, validation_parts = [], []
    required = fit_per_class + validation_per_class
    for class_id in sorted(map(int, torch.unique(allowed_labels).tolist())):
        class_indices = allowed[allowed_labels == class_id]
        if len(class_indices) < required:
            raise ValueError(f"class {class_id} lacks calibration samples")
        permutation = torch.randperm(len(class_indices), generator=generator)
        selected = class_indices[permutation[:required]]
        fit_parts.append(selected[:fit_per_class])
        validation_parts.append(selected[fit_per_class:])
    return torch.cat(fit_parts), torch.cat(validation_parts)


def _encode_batches(
    frontend: RanPACAnalyticLearner,
    features: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(indices), batch_size):
        parts.append(frontend.encode(features[indices[start : start + batch_size]]))
    return torch.cat(parts)


def _select_fixed_ridge(
    *,
    frontend: RanPACAnalyticLearner,
    features: torch.Tensor,
    labels: torch.Tensor,
    fit_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    candidate_lambdas: list[float],
    num_classes: int,
    batch_size: int,
) -> dict:
    fit_codes = _encode_batches(frontend, features, fit_indices, batch_size)
    validation_codes = _encode_batches(
        frontend, features, validation_indices, batch_size
    )
    fit_targets = torch.nn.functional.one_hot(
        labels[fit_indices].to(device=fit_codes.device, dtype=torch.long),
        num_classes=num_classes,
    ).to(fit_codes.dtype)
    validation_targets = torch.nn.functional.one_hot(
        labels[validation_indices].to(
            device=validation_codes.device, dtype=torch.long
        ),
        num_classes=num_classes,
    ).to(validation_codes.dtype)

    # The dual eigensystem evaluates the same Ridge predictors without forming
    # or factorizing one 10k x 10k primal system for every lambda candidate.
    kernel = fit_codes @ fit_codes.T
    kernel = (kernel + kernel.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    eigenvalues = eigenvalues.clamp_min_(0)
    target_coordinates = eigenvectors.T @ fit_targets
    validation_kernel = validation_codes @ fit_codes.T
    scores = []
    for ridge in candidate_lambdas:
        dual = eigenvectors @ (target_coordinates / (eigenvalues[:, None] + ridge))
        prediction = validation_kernel @ dual
        mse = float(torch.mean((prediction - validation_targets) ** 2).item())
        if not torch.isfinite(torch.tensor(mse)):
            raise RuntimeError(f"non-finite Ridge calibration score at {ridge}")
        scores.append({"ridge_lambda": ridge, "validation_mse": mse})
    selected = min(scores, key=lambda item: (item["validation_mse"], item["ridge_lambda"]))
    result = {
        "selected_ridge_lambda": float(selected["ridge_lambda"]),
        "scores": scores,
        "fit_samples": len(fit_indices),
        "validation_samples": len(validation_indices),
    }
    del fit_codes, validation_codes, kernel, eigenvalues, eigenvectors
    del target_coordinates, validation_kernel, fit_targets, validation_targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _common_backend(config: dict, ridge_lambda: float, device) -> dict:
    return {
        "dimension": int(config["ranpac"]["expand_dim"]),
        "ridge_lambda": ridge_lambda,
        "device": device,
        "statistics_dtype": torch.float32,
        "solver_dtype": torch.float32,
    }


def _new_methods(config: dict, feature_dim: int, ridge_lambda: float, device):
    ranpac = config["ranpac"]
    p2b = config["p2b"]
    common = _common_backend(config, ridge_lambda, device)
    seed = int(ranpac["projection_seed"])
    exact = RanPACAnalyticLearner(
        feature_dim=feature_dim, backend=ExactGramBackend(**common), seed=seed
    )
    projection = exact.projection
    fp32 = RanPACAnalyticLearner(
        feature_dim=feature_dim,
        backend=DenseSquareRootBackend(
            update_backend="blocked_qr",
            update_panel_size=int(p2b["update_panel_size"]),
            update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
            **common,
        ),
        seed=seed,
        projection=projection,
    )
    compressed = {}
    for name, storage_mode in (("fp16_square_root", "float16"), ("p2b_int8", "int8")):
        compressed[name] = RanPACAnalyticLearner(
            feature_dim=feature_dim,
            backend=SquareRootBackend(
                storage_mode=storage_mode,
                block_size=int(p2b["block_size"]),
                group_size=int(p2b["group_size"]),
                update_panel_size=int(p2b["update_panel_size"]),
                update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
                first_update_backend=p2b["first_update_backend"],
                quantization_backend=p2b["quantization_backend"],
                quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
                **common,
            ),
            seed=seed,
            projection=projection,
        )
    return {
        "generic_exact": exact,
        "fp32_square_root": fp32,
        **compressed,
    }


def _reference_update(
    state: dict, codes: torch.Tensor, labels: torch.Tensor, ridge_lambda: float
) -> float:
    class_ids = sorted(set(state["class_ids"]) | set(map(int, labels.cpu().tolist())))
    old_columns = {value: index for index, value in enumerate(state["class_ids"])}
    new_columns = {value: index for index, value in enumerate(class_ids)}
    cross = torch.zeros(
        (codes.shape[1], len(class_ids)), device=codes.device, dtype=codes.dtype
    )
    counts = torch.zeros(len(class_ids), device=codes.device, dtype=codes.dtype)
    for class_id, old_column in old_columns.items():
        new_column = new_columns[class_id]
        cross[:, new_column] = state["Q"][:, old_column]
        counts[new_column] = state["counts"][old_column]
    columns = torch.tensor(
        [new_columns[int(value)] for value in labels.cpu().tolist()],
        device=codes.device,
    )
    targets = torch.nn.functional.one_hot(columns, num_classes=len(class_ids)).to(
        codes.dtype
    )
    state["gram"].add_(codes.T @ codes)
    cross.add_(codes.T @ targets)
    counts.add_(targets.sum(0))
    system = state["gram"].clone()
    system.diagonal().add_(ridge_lambda)
    symmetric = (system + system.T) * 0.5
    factor, info = torch.linalg.cholesky_ex(symmetric)
    if int(info.max().item()) != 0:
        raise RuntimeError("reference RanPAC Ridge system is not positive definite")
    weights = torch.cholesky_solve(cross, factor)
    residual = _relative_error(symmetric @ weights, cross)
    state.update(
        class_ids=class_ids,
        Q=cross,
        counts=counts,
        weights=weights,
        total_rows=state["total_rows"] + len(codes),
    )
    return residual


def _reference_persistent_bytes(state: dict, projection: torch.Tensor) -> int:
    return persistent_tensor_bytes(
        {
            "projection": projection,
            "gram": state["gram"],
            "Q": state["Q"],
            "counts": state["counts"],
            "weights": state["weights"],
        }
    )


def _quadratic_bytes(learner: RanPACAnalyticLearner) -> int:
    tensors = {
        name: tensor
        for name, tensor in learner.backend.persistent_tensors().items()
        if name == "gram" or name == "factor" or name.startswith("factor.")
    }
    return persistent_tensor_bytes(tensors)


def _evaluate_seen(
    *,
    methods: dict[str, RanPACAnalyticLearner],
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    reference_weights: torch.Tensor,
    reference_class_ids: list[int],
) -> dict:
    correct = {"reference_original": 0, **{name: 0 for name in methods}}
    exact_agree = 0
    fp32_agree = 0
    exact_diff2 = 0.0
    exact_ref2 = 0.0
    fp32_diff2 = 0.0
    fp32_ref2 = 0.0
    total = 0
    frontend = methods["generic_exact"]
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        codes = frontend.encode(features[batch_indices])
        target = labels[batch_indices].to(device=codes.device)
        reference_logits = codes @ reference_weights
        reference_columns = reference_logits.argmax(1).cpu().tolist()
        reference_prediction = torch.tensor(
            [reference_class_ids[column] for column in reference_columns],
            device=target.device,
        )
        correct["reference_original"] += int(
            (reference_prediction == target).sum().item()
        )
        logits_by_method = {}
        for name, learner in methods.items():
            logits = learner.predict_logits_from_codes(codes)
            logits_by_method[name] = logits
            columns = logits.argmax(1).cpu().tolist()
            prediction = torch.tensor(
                [learner.class_ids[column] for column in columns], device=target.device
            )
            correct[name] += int((prediction == target).sum().item())
        exact_logits = logits_by_method["generic_exact"]
        fp32_logits = logits_by_method["fp32_square_root"]
        exact_agree += int(
            (reference_logits.argmax(1) == exact_logits.argmax(1)).sum().item()
        )
        fp32_agree += int(
            (reference_logits.argmax(1) == fp32_logits.argmax(1)).sum().item()
        )
        exact_diff2 += float(torch.sum((exact_logits - reference_logits) ** 2).item())
        exact_ref2 += float(torch.sum(reference_logits**2).item())
        fp32_diff2 += float(torch.sum((fp32_logits - reference_logits) ** 2).item())
        fp32_ref2 += float(torch.sum(reference_logits**2).item())
        total += len(batch_indices)
    return {
        "accuracy_percent": {
            name: 100.0 * value / total for name, value in correct.items()
        },
        "reference_generic_exact_prediction_agreement": exact_agree / total,
        "reference_generic_exact_relative_logit_error": exact_diff2**0.5
        / max(exact_ref2**0.5, 1.0),
        "reference_fp32_prediction_agreement": fp32_agree / total,
        "reference_fp32_relative_logit_error": fp32_diff2**0.5
        / max(fp32_ref2**0.5, 1.0),
    }


def _run_stream(
    *,
    config: dict,
    features: torch.Tensor,
    labels: torch.Tensor,
    training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    ridge_selection: dict,
    device,
) -> dict:
    dimension = int(config["ranpac"]["expand_dim"])
    ridge_lambda = float(ridge_selection["selected_ridge_lambda"])
    methods = _new_methods(config, int(features.shape[1]), ridge_lambda, device)
    projection = methods["generic_exact"].projection
    reference = {
        "gram": torch.zeros((dimension, dimension), device=device),
        "Q": torch.zeros((dimension, 0), device=device),
        "counts": torch.zeros(0, device=device),
        "class_ids": [],
        "weights": None,
        "total_rows": 0,
    }
    records = []
    for task_id, train_indices in enumerate(training_parts):
        codes = _encode_batches(
            methods["generic_exact"],
            features,
            train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        task_labels = labels[train_indices]
        reference_residual = _reference_update(
            reference, codes, task_labels, ridge_lambda
        )
        for learner in methods.values():
            learner.update_codes(codes, task_labels)

        exact = methods["generic_exact"].backend
        fp32 = methods["fp32_square_root"].backend
        exact_system = exact.gram.clone()
        exact_system.diagonal().add_(ridge_lambda)
        fp32_system_error = _relative_error(fp32.factor.T @ fp32.factor, exact_system)
        fp32_weight_error = _relative_error(fp32.weights, reference["weights"])
        seen_validation = torch.cat(validation_parts[: task_id + 1])
        evaluation = _evaluate_seen(
            methods=methods,
            features=features,
            labels=labels,
            indices=seen_validation,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
            reference_weights=reference["weights"],
            reference_class_ids=reference["class_ids"],
        )
        exact_identity = {
            "gram": torch.equal(reference["gram"], exact.gram),
            "cross": torch.equal(reference["Q"], exact.Q),
            "counts": torch.equal(reference["counts"], exact.counts),
            "weights": torch.equal(reference["weights"], exact.weights),
            "class_ids": reference["class_ids"] == exact.class_ids,
            "persistent_bytes": _reference_persistent_bytes(reference, projection)
            == methods["generic_exact"].persistent_state_bytes(),
        }
        state = {
            name: {
                "total_persistent_bytes": learner.persistent_state_bytes(),
                "quadratic_persistent_bytes": _quadratic_bytes(learner),
                "solver_relative_residual": float(
                    learner.backend.diagnostics["solver_relative_residual"]
                ),
            }
            for name, learner in methods.items()
        }
        record = {
            "task": task_id + 1,
            "training_samples": len(train_indices),
            "seen_validation_samples": len(seen_validation),
            "reference_solver_relative_residual": reference_residual,
            "reference_generic_exact_identity": exact_identity,
            "fp32_system_relative_error": fp32_system_error,
            "fp32_weight_relative_error": fp32_weight_error,
            "evaluation": evaluation,
            "state": state,
        }
        records.append(record)
        print(
            f"TASK {task_id + 1}/{len(training_parts)} "
            f"exact_identity={all(exact_identity.values())} "
            f"fp32_agreement={evaluation['reference_fp32_prediction_agreement']:.6f} "
            f"p2b_acc={evaluation['accuracy_percent']['p2b_int8']:.4f}",
            flush=True,
        )
        del codes, exact_system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    method_names = ["reference_original", *methods]
    aia = {
        name: sum(record["evaluation"]["accuracy_percent"][name] for record in records)
        / len(records)
        for name in method_names
    }
    final_accuracy = {
        name: records[-1]["evaluation"]["accuracy_percent"][name]
        for name in method_names
    }
    exact_quadratic = records[-1]["state"]["generic_exact"][
        "quadratic_persistent_bytes"
    ]
    p2b_quadratic = records[-1]["state"]["p2b_int8"][
        "quadratic_persistent_bytes"
    ]
    reduction = 1.0 - p2b_quadratic / exact_quadratic
    p2b_aia_loss = aia["generic_exact"] - aia["p2b_int8"]
    maximum_solver_residual = max(
        [record["reference_solver_relative_residual"] for record in records]
        + [
            record["state"][name]["solver_relative_residual"]
            for record in records
            for name in methods
        ]
    )
    summary = {
        "selected_ridge_lambda": ridge_lambda,
        "validation_aia_percent": aia,
        "final_validation_accuracy_percent": final_accuracy,
        "maximum_fp32_system_relative_error": max(
            record["fp32_system_relative_error"] for record in records
        ),
        "maximum_fp32_weight_relative_error": max(
            record["fp32_weight_relative_error"] for record in records
        ),
        "maximum_fp32_logit_relative_error": max(
            record["evaluation"]["reference_fp32_relative_logit_error"]
            for record in records
        ),
        "minimum_fp32_prediction_agreement": min(
            record["evaluation"]["reference_fp32_prediction_agreement"]
            for record in records
        ),
        "p2b_validation_aia_loss_pp": p2b_aia_loss,
        "p2b_quadratic_state_reduction_fraction": reduction,
        "final_persistent_bytes": {
            name: records[-1]["state"][name] for name in methods
        },
        "maximum_solver_relative_residual": maximum_solver_residual,
    }
    thresholds = config["gates"]
    gates = {
        "reference_generic_exact_tensor_identity": all(
            all(record["reference_generic_exact_identity"].values())
            for record in records
        ),
        "fp32_system_error": summary["maximum_fp32_system_relative_error"]
        <= thresholds["maximum_fp32_system_relative_error"],
        "fp32_weight_error": summary["maximum_fp32_weight_relative_error"]
        <= thresholds["maximum_fp32_weight_relative_error"],
        "fp32_logit_error": summary["maximum_fp32_logit_relative_error"]
        <= thresholds["maximum_fp32_logit_relative_error"],
        "fp32_prediction_agreement": summary["minimum_fp32_prediction_agreement"]
        >= thresholds["minimum_fp32_prediction_agreement"],
        "srq_quadratic_state_reduction": reduction
        >= thresholds["minimum_srq_quadratic_state_reduction_fraction"],
        "srq_validation_aia_loss": p2b_aia_loss
        <= thresholds["maximum_srq_validation_aia_loss_pp"],
        "solver_residual": maximum_solver_residual
        <= thresholds["maximum_solver_relative_residual"],
    }
    return {"records": records, "summary": summary, "gates": gates}


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M4 requires a clean source checkout")
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M4 refuses a visible test.pt")
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
    fit_indices, calibration_validation_indices = _calibration_indices(
        train["labels"],
        training_parts,
        seed=config["seed"],
        fit_per_class=int(selection["fit_samples_per_class"]),
        validation_per_class=int(selection["validation_samples_per_class"]),
    )

    device = torch.device(args.device)
    # The frontend only needs a dimension/dtype/device owner during calibration;
    # a factor backend avoids allocating an unused 10k x 10k Gram matrix here.
    selection_backend = DenseSquareRootBackend(
        dimension=int(config["ranpac"]["expand_dim"]),
        ridge_lambda=1.0,
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        update_backend="blocked_qr",
        update_panel_size=int(config["p2b"]["update_panel_size"]),
    )
    selection_frontend = RanPACAnalyticLearner(
        feature_dim=int(train["features"].shape[1]),
        backend=selection_backend,
        seed=int(config["ranpac"]["projection_seed"]),
    )
    ridge_result = _select_fixed_ridge(
        frontend=selection_frontend,
        features=train["features"],
        labels=train["labels"],
        fit_indices=fit_indices,
        validation_indices=calibration_validation_indices,
        candidate_lambdas=list(map(float, selection["candidate_lambdas"])),
        num_classes=int(config["num_classes"]),
        batch_size=int(config["ranpac"]["encode_batch_size"]),
    )
    projection_sha256 = _tensor_content_sha256(selection_frontend.projection)
    del selection_backend, selection_frontend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"RIDGE LOCKED train-only lambda={ridge_result['selected_ridge_lambda']}",
        flush=True,
    )

    numerical_failure = None
    try:
        result = _run_stream(
            config=config,
            features=train["features"],
            labels=train["labels"],
            training_parts=training_parts,
            validation_parts=validation_parts,
            ridge_selection=ridge_result,
            device=device,
        )
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        numerical_failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        result = {
            "records": [],
            "summary": {
                "selected_ridge_lambda": ridge_result["selected_ridge_lambda"],
                "numerical_failure": numerical_failure,
            },
            "gates": {"numerical_success": False},
        }
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
        "generic_backend_sha256": _sha256_file(
            ROOT / "methods/analytic_ridge/backends.py"
        ),
        "ranpac_frontend_sha256": _sha256_file(
            ROOT / "methods/frontends/ranpac.py"
        ),
        "train_sha256": _sha256_file(feature_cache_dir / "train.pt"),
        "projection_sha256": projection_sha256,
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
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M4_RANPAC_TRAIN_ONLY"
        if all(result["gates"].values())
        else "FAIL_M4_RANPAC_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "frontend": "RanPAC Phase-2 random-ReLU analytic head",
            "petl_reproduced": False,
            "original_per_task_ridge_schedule_reproduced": False,
            "fixed_ridge_reason": "identical additive-Ridge system across backends",
        },
        "ridge_selection": ridge_result,
        "numerical_failure": numerical_failure,
        "provenance": provenance,
        **result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "m4_results.json.tmp"
    destination = output_dir / "m4_results.json"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
    if payload["status"] != "PASS_M4_RANPAC_TRAIN_ONLY":
        raise RuntimeError("M4 RanPAC train-only gate failed")
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


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
