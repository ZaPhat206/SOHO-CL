import copy
import json
from pathlib import Path

import pytest
import torch

from tools import soho_matched_selection as matched


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "soho_matched_selection_final.json"
LOCKED_HPARAMS = ROOT / "configs" / "soho_matched_selected_hyperparameters.json"


def test_v2_protocol_locks_equal_split_replicates_and_four_method_roles():
    protocol = matched._read_protocol(PROTOCOL)
    assert tuple(protocol["matched_final_methods"]) == matched.METHODS
    assert len(matched._fly_candidates(protocol)) == 18
    assert protocol["selection"]["development_replicates"] == [
        {"class_order_seed": 2025, "projection_seed": 4201},
        {"class_order_seed": 2026, "projection_seed": 4202},
        {"class_order_seed": 2027, "projection_seed": 4203},
    ]
    assert protocol["final_evaluation"]["replicates"][0] == {
        "class_order_seed": 3031, "projection_seed": 5031,
    }


def test_fly_grid_and_near_tie_prefer_smaller_projection_then_sparser_code():
    protocol = matched._read_protocol(PROTOCOL)
    candidates = matched._fly_candidates(protocol)
    assert {item["synaptic_degree"] for item in candidates} == {100, 300, 500}
    assert {item["coding_level"] for item in candidates} == {0.1, 0.2, 0.3, 0.4, 0.45, 0.5}
    results = [
        {"valid": True, "mean_inner_aia": 90.04,
         "config": {"synaptic_degree": 300, "coding_level": 0.3}},
        {"valid": True, "mean_inner_aia": 90.00,
         "config": {"synaptic_degree": 100, "coding_level": 0.2}},
        {"valid": True, "mean_inner_aia": 89.98,
         "config": {"synaptic_degree": 100, "coding_level": 0.1}},
    ]
    selected, best = matched._select_fly_near_tie(results, 0.05)
    assert best == pytest.approx(90.04)
    assert selected["config"] == {"synaptic_degree": 100, "coding_level": 0.2}


def test_candidate_protocol_does_not_mutate_official_fly_control():
    protocol = matched._read_protocol(PROTOCOL)
    original = copy.deepcopy(protocol["fly_fixed"])
    configured = matched._protocol_with_fly(
        protocol, {"synaptic_degree": 100, "coding_level": 0.2}
    )
    assert protocol["fly_fixed"] == original
    assert configured["fly_fixed"]["synaptic_degree"] == 100
    assert configured["fly_fixed"]["coding_level"] == 0.2
    assert original["synaptic_degree"] == 300
    assert original["coding_level"] == 0.3


def test_test_loader_mapping_is_converted_to_deterministic_loader_sequence():
    first, second = object(), object()
    assert matched._ordered_test_loaders({1: second, 0: first}) == [first, second]
    assert matched._ordered_test_loaders([first, second]) == [first, second]


def test_tuned_fly_uses_candidate_without_changing_method_fidelity(monkeypatch):
    captured = {}

    def fake_evaluate(method, protocol, dataset, seed, stream, train_parts,
                      test_parts, soho_config, raw_ridge, device_name, uses_test_set):
        captured.update({
            "method": method,
            "fly": copy.deepcopy(protocol["fly_fixed"]),
            "uses_test_set": uses_test_set,
        })
        return {"status": "complete", "method": method}

    monkeypatch.setattr(matched.base, "_evaluate", fake_evaluate)
    protocol = matched._read_protocol(PROTOCOL)
    result = matched._evaluate_method(
        "flycl_validation_tuned", protocol,
        {"synaptic_degree": 100, "coding_level": 0.2},
        {"num_classes": 2}, 3, {}, [], [],
        {"density": 0.3, "coding_level": 0.3, "use_etf": True},
        0.1, "cpu",
    )
    assert captured["method"] == "flycl_fidelity"
    assert captured["fly"]["synaptic_degree"] == 100
    assert captured["fly"]["coding_level"] == 0.2
    assert captured["uses_test_set"] is True
    assert result["method"] == "flycl_validation_tuned"


def test_locked_test_hyperparameters_are_train_only_and_inside_declared_grids():
    protocol = matched._read_protocol(PROTOCOL)
    manifest = json.loads(LOCKED_HPARAMS.read_text(encoding="utf-8"))
    assert manifest["uses_test_set"] is False
    assert manifest["test_tuning_allowed"] is False
    assert manifest["selection_protocol_sha256"] == matched.base._sha256_file(PROTOCOL)
    assert manifest["selection_runner_sha256"] == matched.base._sha256_file(
        ROOT / "tools" / "soho_matched_selection.py"
    )
    soho_candidates = matched.base._soho_candidates(protocol)
    fly_candidates = matched._fly_candidates(protocol)
    raw_grid = set(map(float, protocol["selection"]["raw_ridge_grid"]))
    assert set(manifest["selected"]) == set(matched.DATASET_KEYS)
    for selected in manifest["selected"].values():
        assert selected["soho_config"] in soho_candidates
        assert selected["fly_validation_tuned_config"] in fly_candidates
        assert float(selected["raw_ridge_lambda"]) in raw_grid
