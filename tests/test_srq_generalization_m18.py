import copy
import json
from pathlib import Path

import pytest
import torch

from methods.analytic_ridge import ExactGramBackend, SquareRootBackend
from tools import srq_generalization_m18 as m18


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/srq_generalization_m18_fly20k_adaptive_locked_test.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _toy_config():
    config = _config()
    config["dataset"].update(
        num_classes=6,
        num_tasks=3,
        expected_train_samples=60,
        expected_test_samples=30,
    )
    config["backbone"]["feature_dim"] = 4
    config["representation"].update(
        expand_dim=32,
        synaptic_degree=4,
        coding_level=0.25,
        evaluation_batch_size=16,
    )
    config["ridge"]["lambda"] = 10.0
    config["square_root_backend"].update(
        block_size=8,
        group_size=8,
        update_panel_size=8,
        quantization_batch_blocks=4,
    )
    return config


def _toy_inputs():
    generator = torch.Generator().manual_seed(18)
    train_labels = torch.arange(60) % 6
    test_labels = torch.arange(30) % 6
    labels = torch.cat((train_labels, test_labels))
    active = 8
    code_indices = torch.stack(
        [torch.randperm(32, generator=generator)[:active] for _ in range(90)]
    ).to(torch.int16)
    code_values = torch.randn(90, active, generator=generator)
    training_parts, test_parts = [], []
    for task in range(3):
        class_ids = torch.tensor([2 * task, 2 * task + 1])
        training_parts.append(
            torch.nonzero(torch.isin(train_labels, class_ids)).flatten()
        )
        test_parts.append(
            torch.nonzero(torch.isin(test_labels, class_ids)).flatten() + 60
        )
    projection = torch.randn(32, 4, generator=generator).to_sparse_csc()
    stream = {
        "features": torch.randn(90, 4, generator=generator),
        "labels": labels,
    }
    return stream, code_indices, code_values, projection, training_parts, test_parts


def test_m18_locked_config_and_source_identity():
    config = m18._read_config(CONFIG_PATH)
    assert config["representation"]["frontend"] == "fly_wta"
    assert config["representation"]["expand_dim"] == 20_000
    assert config["methods"] == list(m18.METHODS)
    assert config["ridge"]["lambda"] == 1_000_000.0
    assert config["ridge"]["width_20000_retuned"] is False
    assert config["evaluation"]["accuracy_gate"] is None
    assert config["integrity_gates"]["accuracy_gate"] is None
    assert m18._verify_source_identity(config) == config["source_identity"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["representation"].__setitem__("expand_dim", 10_000),
        lambda value: value["ridge"].__setitem__("lambda", 2_000_000.0),
        lambda value: value["replicates"].pop(),
        lambda value: value["evaluation"].__setitem__("accuracy_gate", 0.25),
        lambda value: value["adaptive"].__setitem__(
            "selection_signal", "validation_accuracy"
        ),
    ),
)
def test_m18_config_rejects_design_drift(tmp_path, mutation):
    config = _config()
    mutation(config)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        m18._read_config(path)


def test_m18_backend_construction_is_exact_int8_and_adaptive():
    config = _toy_config()
    exact = m18._backend(config, m18.METHODS[0], torch.device("cpu"))
    int8 = m18._backend(config, m18.METHODS[1], torch.device("cpu"))
    adaptive = m18._backend(config, m18.METHODS[2], torch.device("cpu"))
    assert isinstance(exact, ExactGramBackend)
    assert isinstance(int8, SquareRootBackend) and int8.storage_mode == "int8"
    assert isinstance(adaptive, SquareRootBackend)
    assert adaptive.storage_mode == "adaptive_int8_fp16"
    assert adaptive.adaptive_budget_fraction == 0.25


def test_m18_paired_toy_run_shares_codes_and_accounts_state():
    config = _toy_config()
    stream, code_indices, code_values, projection, training_parts, test_parts = (
        _toy_inputs()
    )
    result = m18._evaluate_replicate(
        config=config,
        stream=stream,
        code_indices=code_indices,
        code_values=code_values,
        projection=projection,
        training_parts=training_parts,
        test_parts=test_parts,
        device=torch.device("cpu"),
        replicate_index=0,
    )
    assert result["status"] == "complete"
    assert set(result["methods"]) == set(m18.METHODS)
    exact = result["methods"][m18.METHODS[0]]
    int8 = result["methods"][m18.METHODS[1]]
    adaptive = result["methods"][m18.METHODS[2]]
    assert len(exact["accuracy_matrix"]) == 3
    assert all(
        left > middle > right
        for left, middle, right in zip(
            exact["persistent_state_bytes_by_task"],
            adaptive["persistent_state_bytes_by_task"],
            int8["persistent_state_bytes_by_task"],
        )
    )
    for record in adaptive["diagnostics_by_task"]:
        assert (
            record["factor_all_int8_bytes"]
            <= record["factor_persistent_bytes"]
            <= record["factor_budget_ceiling_bytes"]
        )
        assert record["used_extra_bytes"] <= record["extra_budget_bytes"]
    for method in m18.METHODS:
        assert result["methods"][method]["maximum_solver_relative_residual"] < 2e-5


def test_m18_summary_has_structural_not_accuracy_gate():
    config = _toy_config()
    stream, code_indices, code_values, projection, training_parts, test_parts = (
        _toy_inputs()
    )
    unit = m18._evaluate_replicate(
        config=config,
        stream=stream,
        code_indices=code_indices,
        code_values=code_values,
        projection=projection,
        training_parts=training_parts,
        test_parts=test_parts,
        device=torch.device("cpu"),
        replicate_index=0,
    )
    seed_results = [
        {
            "replicate_index": index,
            "class_order_seed": 3031 + index,
            "projection_seed": 5031 + index,
            "status": "complete",
            "methods": copy.deepcopy(unit["methods"]),
            "paired_diagnostics": copy.deepcopy(unit["paired_diagnostics"]),
        }
        for index in range(6)
    ]
    summary, gates = m18._summarize_results(config, seed_results)
    assert summary["methods"][m18.METHODS[0]]["average_incremental_accuracy"]["n"] == 6
    assert gates["all_six_paired_replicates_complete"] is True
    assert gates["no_seed_excluded"] is True
    assert gates["adaptive_budget_conformance"] is True
    assert gates["accuracy_gate"] is None


def test_m18_visible_test_cache_is_rejected_before_authorization(tmp_path):
    (tmp_path / "test.pt").write_bytes(b"visible")
    with pytest.raises(RuntimeError, match="test.pt"):
        m18._validate_cache(_config(), tmp_path, require_test=False)
