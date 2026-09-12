"""M14 multi-seed LoRanPAC protocol and runner gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools import srq_generalization_m14 as m14


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m14_loranpac_multiseed_train_only.json"


def test_m14_config_locks_fresh_paired_seeds_and_scale_aware_gates():
    config = m14._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["methods"] == list(m14.METHODS)
    assert [item["class_order_seed"] for item in config["replicates"]] == list(
        range(4101, 4107)
    )
    gates = config["integrity_gates"]
    assert gates["maximum_normalized_orthogonality_residual"] == 1.5e-3
    assert gates["maximum_task1_spectral_orthogonality_residual"] == 3.5e-3
    assert gates["maximum_task1_solver_relative_residual"] == 1e-3
    assert gates["accuracy_gate"] is None


def test_m14_rejects_test_use_accuracy_gate_and_seed_mutation(tmp_path):
    original = json.loads(CONFIG.read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(original))
    broken["uses_test_set"] = True
    path = tmp_path / "test.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        m14._read_config(path)

    broken = json.loads(json.dumps(original))
    broken["evaluation"]["accuracy_gate"] = 0.0
    path = tmp_path / "accuracy.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="accuracy"):
        m14._read_config(path)

    broken = json.loads(json.dumps(original))
    broken["replicates"][0]["projection_seed"] += 1
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="design"):
        m14._read_config(path)

    broken = json.loads(json.dumps(original))
    broken["p2b"]["group_size"] = 128
    path = tmp_path / "p2b.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="P2B"):
        m14._read_config(path)


def test_m14_rank_contract_uses_exact_paired_backend_bytes():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["num_classes"] = 6
    config["ranpac"]["feature_dimension"] = 9
    width, rank = 24, 5
    projection_bytes = 4 * 9 * width
    base = 8 * width * 6 + 4 * 6
    per_rank = 4 * (width + 1)
    target = projection_bytes + base + rank * per_rank + 17
    contract = m14._rank_contract_for_target(
        config, width=width, budget="p2b_int8", target_total_bytes=target
    )
    assert contract["derived_max_rank"] == rank
    assert contract["final_budget_underfill_bytes"] == 17
    assert contract["target_total_persistent_bytes"] == target
    assert contract["target_source"] == "paired_backend_measured_final_state"


def test_m14_orthogonality_reports_rank_normalized_and_task1_spectral():
    basis = torch.eye(8)[:, :5].clone()
    metrics = m14._orthogonality_metrics(basis, spectral=True)
    assert metrics["raw_frobenius"] == 0.0
    assert metrics["frobenius_over_sqrt_rank"] == 0.0
    assert metrics["spectral_norm"] == 0.0
    without_spectral = m14._orthogonality_metrics(basis, spectral=False)
    assert without_spectral["spectral_norm"] is None


def test_m14_small_loranpac_unit_is_train_only_and_byte_matched():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["num_classes"] = 6
    config["num_tasks"] = 2
    config["ranpac"]["feature_dimension"] = 9
    config["ranpac"]["encode_batch_size"] = 8
    config["ranpac"]["evaluation_batch_size"] = 8
    config["selected_ridge_by_width"] = {"24": 10.0}
    replicate = {"class_order_seed": 4101, "projection_seed": 4101, "split_seed": 4101}
    generator = torch.Generator().manual_seed(1414)
    train = {
        "features": torch.randn(72, 9, generator=generator),
        "labels": torch.arange(72) % 6,
    }
    projection = torch.randn(9, 24, generator=generator)
    order = list(range(6))
    training_parts = [torch.arange(0, 24), torch.arange(24, 48)]
    validation_parts = [torch.arange(48, 60), torch.arange(60, 72)]
    projection_bytes = projection.numel() * projection.element_size()
    base = 8 * 24 * 6 + 4 * 6
    per_rank = 4 * (24 + 1)
    target = projection_bytes + base + 5 * per_rank + 13
    result = m14._run_loranpac_unit(
        config=config, sources={}, train=train, replicate=replicate, width=24,
        method="loranpac_p2b_budget", projection=projection, order=order,
        training_parts=training_parts, validation_parts=validation_parts,
        device=torch.device("cpu"), target_total_bytes=target,
    )
    assert result["uses_test_set"] is False
    assert result["rank_contract"]["derived_max_rank"] == 5
    assert result["final_total_persistent_bytes"] == target - 13
    assert len(result["records"]) == 2
    assert result["records"][0]["orthogonality"]["spectral_norm"] is not None
    assert result["records"][1]["orthogonality"]["spectral_norm"] is None
