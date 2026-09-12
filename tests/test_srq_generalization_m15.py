from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import zipfile

import pytest
import torch

from tools import srq_generalization_m15 as m15


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "srq_generalization_m15_loranpac_task1_closure.json"


def test_m15_config_locks_post_m14_nonpredictive_audit():
    config = m15._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["computes_predictive_metrics"] is False
    assert config["uses_accuracy_for_selection"] is False
    assert config["seed"] == 2025
    assert config["diagnostic_seeds"] == [4105, 4101]
    assert config["widths"] == [10000, 20000]
    assert config["budget_targets"] == ["p2b_int8", "adaptive_int8_fp16"]
    assert config["source_m14"]["required_status"] == (
        "FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY"
    )
    assert config["source_m14"]["locked_task1_solver_threshold"] == 1e-3
    assert config["gates"]["accuracy_gate"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uses_test_set", True),
        ("computes_predictive_metrics", True),
        ("diagnostic_seeds", [4101, 4105]),
        ("widths", [10000]),
    ],
)
def test_m15_rejects_boundary_or_design_mutation(tmp_path, field, value):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        m15._read_config(path)


def test_m15_source_requires_immutable_m14_fail(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = config["source_m14"]
    units = []
    for seed in range(4101, 4107):
        for width in (10000, 20000):
            for budget, method in m15.METHOD_BY_BUDGET.items():
                residual = 5e-4
                if seed == 4105 and width == 10000 and budget == "p2b_int8":
                    residual = source["observed_maximum_task1_solver_residual"]
                units.append({
                    "identity": {
                        "class_order_seed": seed,
                        "projection_seed": seed,
                        "split_seed": seed,
                        "width": width,
                        "method": method,
                        "budget_target": budget,
                    },
                    "uses_test_set": False,
                    "records": [{
                        "matched_ridge_solver_relative_residual": residual,
                        "official_ridge0_solver_relative_residual": residual,
                    }],
                })
    payload = {
        "status": source["required_status"],
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "gates": {"task1_solver_residual": False, "all_units_complete": True},
        "summary": {
            "maximum_task1_solver_relative_residual": source[
                "observed_maximum_task1_solver_residual"
            ],
            "completed_units": 60,
            "expected_units": 60,
        },
        "units": units,
        "provenance": {
            "git_commit": source["source_commit"],
            "loranpac_backend_sha256": "a" * 64,
        },
    }
    raw = (json.dumps(payload) + "\n").encode()
    artifact = tmp_path / source["artifact_filename"]
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr(source["result_member"], raw)
    source["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    source["result_sha256"] = hashlib.sha256(raw).hexdigest()
    loaded, identity = m15._load_m14_source(config, artifact)
    assert loaded["gates"]["task1_solver_residual"] is False
    assert identity["reported_status"] == source["required_status"]
    payload["gates"]["task1_solver_residual"] = True
    raw = (json.dumps(payload) + "\n").encode()
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr(source["result_member"], raw)
    source["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    source["result_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        m15._load_m14_source(config, artifact)


def test_system_preserving_qr_keeps_truncated_gram_and_closes_solver():
    generator = torch.Generator().manual_seed(1515)
    seed_matrix = torch.randn(18, 6, generator=generator, dtype=torch.float64)
    q0, _ = torch.linalg.qr(seed_matrix, mode="reduced")
    triangular0 = torch.eye(6, dtype=torch.float64)
    triangular0[0, 1] = 0.07
    triangular0[2, 4] = -0.04
    basis = q0 @ triangular0
    singular_values = torch.linspace(4.0, 1.0, 6, dtype=torch.float64)
    cross = torch.randn(18, 3, generator=generator, dtype=torch.float64)
    q, triangular = torch.linalg.qr(basis, mode="reduced")
    raw, raw_weights = m15._raw_formula_metrics(
        basis, singular_values, cross, 0.25, torch.float64
    )
    audit = m15._system_preserving_qr_metrics(
        basis=basis,
        singular_values=singular_values,
        cross=cross,
        q_prefix=q,
        triangular_prefix=triangular,
        ridge_lambda=0.25,
        dtype=torch.float64,
        raw_weights=raw_weights,
    )
    original = basis @ torch.diag(singular_values.square()) @ basis.T
    factor = triangular * singular_values.unsqueeze(0)
    reconstructed = q @ (factor @ factor.T) @ q.T
    assert torch.allclose(original, reconstructed, atol=1e-11, rtol=1e-11)
    assert raw["normwise_backward_error"] > 1e-4
    assert audit["factor_reconstruction_relative_error"] < 1e-12
    assert audit["core_normwise_backward_error"] < 1e-14


def test_same_factor_fp64_does_not_silently_replace_the_svd():
    basis = torch.tensor(
        [[1.0, 0.03], [0.0, 0.999], [0.0, 0.0]], dtype=torch.float32
    )
    singular_values = torch.tensor([3.0, 1.0], dtype=torch.float32)
    cross = torch.tensor([[1.0], [2.0], [0.0]], dtype=torch.float32)
    fp32, _ = m15._raw_formula_metrics(
        basis, singular_values, cross, 1.0, torch.float32
    )
    fp64, _ = m15._raw_formula_metrics(
        basis, singular_values, cross, 1.0, torch.float64
    )
    assert fp32["legacy_relative_residual"] > 0
    assert fp64["legacy_relative_residual"] > 0
    assert fp64["legacy_relative_residual"] == pytest.approx(
        fp32["legacy_relative_residual"], rel=2e-5
    )


def test_m15_runner_has_no_prediction_path_and_output_guard_rejects_one():
    source = inspect.getsource(m15)
    assert ".argmax(" not in source
    assert "predict_logits" not in source
    assert "load_test=False" in source
    assert m15._predictive_fields_absent([{"solver_residual": 1e-5}])
    assert not m15._predictive_fields_absent([{"validation_accuracy": 92.0}])
