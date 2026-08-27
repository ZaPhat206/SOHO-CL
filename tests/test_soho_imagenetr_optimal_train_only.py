import json
from pathlib import Path

import pytest
import torch

from tools import soho_imagenetr_optimal_train_only as optimal


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_locks_fair_seeds_and_original_fly():
    protocol = optimal._read_protocol(
        ROOT / "configs/soho_imagenetr_optimal_train_only.json"
    )
    assert optimal._verify_method_identity(protocol) == protocol["method_identity"]
    assert protocol["outer_confirmation"]["held_out_test_authorized"] is False
    assert protocol["fly_fidelity"] == {
        "expand_dim": 10000,
        "synaptic_degree": 300,
        "coding_level": 0.3,
        "ridge_lower": 6,
        "ridge_upper": 10,
    }
    assert len(protocol["outer_confirmation"]["replicates"]) == 5


def test_fixed_ridge_soho_uses_declared_lambda_and_discloses_replay():
    learner = optimal.FixedRidgeSOHO(
        4,
        16,
        0.5,
        4,
        True,
        0.25,
        4,
        -2,
        2,
        seed=2025,
        device="cpu",
        replay_chunk_size=8,
        gcv_sample_size=8,
        fixed_ridge=10.0,
    )
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(12, 4, generator=generator)
    labels = torch.tensor([0, 1, 2, 3] * 3)
    learner.update(features, labels)
    assert learner.last_ridge == 10.0
    assert learner.diagnostics["ridge_policy"] == "fixed_train_validation_selected"
    assert torch.isfinite(learner.weights).all()
    assert learner.predict(features).shape == (12,)
    audit = optimal.base._state_audit(learner, "soho_replay_fidelity", 12)
    assert audit["exemplar_free"] is False
    assert audit["historical_feature_rows"] == 12


def test_ridge_near_tie_prefers_larger_regularization():
    results = [
        {"valid": True, "mean_inner_aia": 80.00, "config": {"ridge_lambda": 100.0}},
        {"valid": True, "mean_inner_aia": 79.98, "config": {"ridge_lambda": 1000.0}},
        {"valid": True, "mean_inner_aia": 79.70, "config": {"ridge_lambda": 10000.0}},
    ]
    selected, best = optimal._select_near_tie(
        results, 0.05, lambda item: -float(item["config"]["ridge_lambda"])
    )
    assert best == pytest.approx(80.0)
    assert selected["config"]["ridge_lambda"] == 1000.0


def test_five_seed_ci_is_serializable_and_paired():
    summary = optimal._mean_ci([0.1, 0.2, 0.3, 0.4, 0.5])
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(0.3)
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]
    json.dumps(summary)


def test_colab_notebook_pins_sources_and_never_materializes_test():
    notebook = json.loads(
        (ROOT / "notebooks/soho_imagenetr_optimal_train_only_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    protocol = ROOT / "configs/soho_imagenetr_optimal_train_only.json"
    runner = ROOT / "tools/soho_imagenetr_optimal_train_only.py"
    assert optimal.base._sha256_file(protocol) in source
    assert optimal.base._sha256_file(runner) in source
    assert "--extract-train-only" in source
    assert "extract-test" not in source
    assert "test.pt absent" in source
    assert "held_out_test_authorized" in source
