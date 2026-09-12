"""M13-N train-only numerical-audit protocol and math tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from tools import srq_generalization_m13n as m13n


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m13n_numerical_audit.json"


def test_m13n_config_locks_nonpredictive_train_only_audit():
    config = m13n._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["computes_accuracy"] is False
    assert config["uses_accuracy_for_selection"] is False
    assert config["source_m13"]["archive_opening_permitted"] is False
    assert config["source_m13"]["required_status_disclosure"] == (
        "FAIL_M13_LORANPAC_TRAIN_ONLY"
    )
    assert config["gates"]["accuracy_gate"] is None
    assert config["train_identity"]["training_indices_sha256"] == (
        "ff06d5687dc8069c599b73c29c435cd84e2be160fc878fcfde8e37a565f326c9"
    )
    assert config["train_identity"]["validation_indices_sha256"] == (
        "979e3bea647ed5f51b8a354c8adb13be9fb82739a7773e3c941080bd848cb313"
    )
    assert config["protocol_recovery"][
        "numerical_metrics_observed_before_recovery"
    ] is False
    assert config["protocol_recovery"]["scientific_choices_changed"] is False
    assert {(unit["width"], unit["budget_target"]) for unit in config["units"]} == {
        (10000, "p2b_int8"),
        (10000, "adaptive_int8_fp16"),
        (20000, "p2b_int8"),
        (20000, "adaptive_int8_fp16"),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("uses_test_set", True),
        ("computes_accuracy", True),
        ("uses_accuracy_for_selection", True),
    ),
)
def test_m13n_rejects_predictive_or_test_boundary_mutation(tmp_path, field, value):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[field] = value
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only numerical audit"):
        m13n._read_config(path)


def test_m13n_source_verification_hashes_raw_bytes_without_opening_container(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload = b"not a zip and intentionally uninterpretable\x00\xff"
    path = tmp_path / config["source_m13"]["artifact_filename"]
    path.write_bytes(payload)
    config["source_m13"]["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
    identity = m13n._verify_source_artifact(config, path)
    assert identity["sha256_matches"] is True
    assert identity["container_opened"] is False
    assert identity["reported_source_status"] == "FAIL_M13_LORANPAC_TRAIN_ONLY"


def test_m13n_runner_has_no_archive_parser_or_prediction_path():
    source = inspect.getsource(m13n)
    assert "zipfile" not in source
    assert "tarfile" not in source
    assert ".argmax(" not in source
    assert "predict_logits" not in source


def test_m13n_rank_contracts_are_independently_rederived_from_bytes():
    config = m13n._read_config(CONFIG)
    contracts = {
        (unit["width"], unit["budget_target"]): m13n._rank_contract(
            config, unit, 4000
        )
        for unit in config["units"]
    }
    assert all(item["maximum_rank_matches_lock"] for item in contracts.values())
    assert all(
        item["task1_effective_rank_matches_lock"] for item in contracts.values()
    )
    assert contracts[(20000, "adaptive_int8_fp16")][
        "derived_task1_effective_rank"
    ] == 3000


def test_m13n_qr_diagnostic_isolates_basis_nonorthogonality():
    generator = torch.Generator().manual_seed(13200)
    exact, _ = torch.linalg.qr(
        torch.randn(96, 28, generator=generator), mode="reduced"
    )
    distorted = exact * torch.linspace(0.88, 1.12, 28).unsqueeze(0)
    singular_values = torch.linspace(25.0, 2.0, 28)
    cross = torch.randn(96, 7, generator=generator)
    audit = m13n._audit_basis(
        distorted, singular_values, cross, official_ridge=0.0, matched_ridge=10.0
    )
    assert audit["raw"]["orthogonality"]["raw_frobenius"] > 0.1
    assert audit["improvement_factor"]["orthogonality_raw_frobenius"] > 100
    assert audit["improvement_factor"]["maximum_solver_relative_residual"] > 100
    assert audit["qr_reorthogonalized_diagnostic"][
        "changes_represented_truncated_system"
    ] is True
    assert audit["qr_reorthogonalized_diagnostic"]["is_proposed_method_change"] is False


def test_m13n_spectral_metric_matches_exact_symmetric_eigendecomposition():
    basis = torch.tensor(
        [[1.1, 0.0], [0.0, 0.9], [0.0, 0.0]], dtype=torch.float64
    )
    metrics = m13n._orthogonality_metrics(basis)
    # Eigenvalues of U^T U - I are 0.21 and -0.19.
    assert metrics["spectral_norm"] == pytest.approx(0.21, abs=1e-14)
    assert metrics["raw_frobenius"] == pytest.approx(
        math.sqrt(0.21**2 + 0.19**2), abs=1e-14
    )


def test_m13n_small_fp64_oracle_meets_locked_precision_bound():
    config = m13n._read_config(CONFIG)
    oracle = m13n._run_small_oracle(config)
    normalized = oracle["results"]["float64"]["raw"]["orthogonality"][
        "frobenius_over_sqrt_rank"
    ]
    assert normalized <= config["gates"][
        "maximum_small_oracle_fp64_normalized_orthogonality"
    ]
    assert m13n._all_finite(oracle)
