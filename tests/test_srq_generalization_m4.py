"""M4 gates for the RanPAC analytic-head generalization study."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from methods.analytic_ridge import ExactGramBackend
from methods.frontends import RanPACAnalyticLearner
from tools import srq_generalization_m4 as m4


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "srq_generalization_m4_ranpac_train_only.json"
NOTEBOOK = ROOT / "notebooks" / "srq_generalization_m4_ranpac_colab.ipynb"


def _config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["num_classes"] = 6
    config["num_tasks"] = 3
    config["ranpac"].update(
        expand_dim=24,
        encode_batch_size=7,
        evaluation_batch_size=8,
    )
    config["ridge_selection"].update(
        candidate_lambdas=[0.1, 1.0, 10.0, 100.0],
        fit_samples_per_class=3,
        validation_samples_per_class=2,
    )
    config["p2b"].update(
        block_size=6,
        group_size=5,
        update_panel_size=7,
        quantization_batch_blocks=3,
    )
    config["gates"].update(
        maximum_fp32_system_relative_error=1e-5,
        maximum_fp32_weight_relative_error=1e-5,
        maximum_fp32_logit_relative_error=1e-5,
        minimum_fp32_prediction_agreement=1.0,
        maximum_srq_validation_aia_loss_pp=100.0,
        maximum_solver_relative_residual=1e-4,
    )
    return config


def _stream():
    config = _config()
    generator = torch.Generator().manual_seed(811)
    features = torch.randn(72, 9, generator=generator)
    labels = torch.tensor([index % 6 for index in range(72)])
    training_parts, validation_parts = [], []
    for task in range(3):
        indices = torch.where((labels == 2 * task) | (labels == 2 * task + 1))[0]
        training_parts.append(indices[:16])
        validation_parts.append(indices[16:])
    return config, features, labels, training_parts, validation_parts


def test_m4_config_locks_train_only_controlled_ranpac_semantics():
    config = m4._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["ranpac"]["path"] == "phase2_no_petl_random_relu"
    assert config["ranpac"]["upstream_commit"] == (
        "cf4b301d18b0c27db030f4371b72b768005ae58a"
    )
    assert config["ridge_selection"]["policy"] == (
        "single_fixed_train_only_calibration"
    )


def test_m4_rejects_test_use_and_relaxed_reference_identity(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["uses_test_set"] = True
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        m4._read_config(path)
    config["uses_test_set"] = False
    config["gates"]["require_reference_exact_tensor_identity"] = False
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="requires reference"):
        m4._read_config(path)


def test_m4_calibration_is_a_disjoint_subset_of_outer_training():
    config, _, labels, training_parts, validation_parts = _stream()
    fit, validation = m4._calibration_indices(
        labels,
        training_parts,
        seed=config["seed"],
        fit_per_class=3,
        validation_per_class=2,
    )
    outer_training = set(torch.cat(training_parts).tolist())
    outer_validation = set(torch.cat(validation_parts).tolist())
    assert set(fit.tolist()).isdisjoint(set(validation.tolist()))
    assert set(fit.tolist()) | set(validation.tolist()) <= outer_training
    assert (set(fit.tolist()) | set(validation.tolist())).isdisjoint(outer_validation)


def test_m4_dual_selection_uses_fixed_candidate_and_is_deterministic():
    config, features, labels, training_parts, _ = _stream()
    fit, validation = m4._calibration_indices(
        labels,
        training_parts,
        seed=config["seed"],
        fit_per_class=3,
        validation_per_class=2,
    )
    frontend = RanPACAnalyticLearner(
        feature_dim=features.shape[1],
        backend=ExactGramBackend(dimension=24, ridge_lambda=1.0),
        seed=2025,
    )
    first = m4._select_fixed_ridge(
        frontend=frontend,
        features=features,
        labels=labels,
        fit_indices=fit,
        validation_indices=validation,
        candidate_lambdas=config["ridge_selection"]["candidate_lambdas"],
        num_classes=6,
        batch_size=7,
    )
    second = m4._select_fixed_ridge(
        frontend=frontend,
        features=features,
        labels=labels,
        fit_indices=fit,
        validation_indices=validation,
        candidate_lambdas=config["ridge_selection"]["candidate_lambdas"],
        num_classes=6,
        batch_size=7,
    )
    assert first == second
    assert first["selected_ridge_lambda"] in config["ridge_selection"][
        "candidate_lambdas"
    ]


def test_m4_synthetic_stream_matches_reference_and_reduces_quadratic_state():
    config, features, labels, training_parts, validation_parts = _stream()
    result = m4._run_stream(
        config=config,
        features=features,
        labels=labels,
        training_parts=training_parts,
        validation_parts=validation_parts,
        ridge_selection={"selected_ridge_lambda": 100.0},
        device=torch.device("cpu"),
    )
    assert all(result["gates"].values())
    assert all(
        all(record["reference_generic_exact_identity"].values())
        for record in result["records"]
    )
    assert result["summary"]["minimum_fp32_prediction_agreement"] == 1.0
    assert result["summary"]["p2b_quadratic_state_reduction_fraction"] >= 0.7


def test_m4_notebook_is_source_locked_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "--extract-train-only" in code
    assert "test.pt').exists()" in code
    assert "--require-clean-git" in code
    assert "m4_results.json" in code
    assert "files.download(archive)" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
