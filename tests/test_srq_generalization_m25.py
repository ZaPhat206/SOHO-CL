"""Small contract tests for the M25 cross-backbone experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import srq_generalization_m25 as m25


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m25_ranpac_resnet_cifar_train_only.json"


def _payload(seed: int, aia: float = 90.0) -> dict:
    summary = {
        "validation_aia_percent": {name: aia + index for index, name in enumerate(m25.METHODS)},
        "final_validation_accuracy_percent": {name: aia + index + 1 for index, name in enumerate(m25.METHODS)},
        "final_total_persistent_bytes": {name: 1000 - index * 100 for index, name in enumerate(m25.METHODS)},
        "analytic_update_seconds": {name: 1.0 + index for index, name in enumerate(m25.METHODS)},
        "paired_aia_difference_from_exact_pp": {"exact": 0.0, "p2b_int8": 1.0, "adaptive_int8_fp16": 2.0},
    }
    return {
        "schema_version": 1,
        "study_id": "srq-generalization-m25-ranpac-resnet-cifar-train-only-v1",
        "stream_id": f"s{seed}",
        "status": "PASS_M25_RANPAC_RESNET_CIFAR_TRAIN_ONLY",
        "uses_test_set": False,
        "summary": summary,
        "gates": {
            "train_only_cache": True,
            "solver_residual": True,
            "compressed_state_smaller_than_exact": True,
            "accuracy_not_used_as_gate": True,
        },
        "provenance": {"train_sha256": "a" * 64},
    }


def test_m25_config_locks_resnet_and_six_streams():
    config = m25._read_config(CONFIG)
    assert config["backbone"]["feature_dimension"] == 2048
    assert config["ranpac"]["full_expand_dimension"] == 10000
    assert [item["seed"] for item in config["streams"]] == list(range(2025, 2031))
    assert config["uses_test_set"] is False
    assert config["integrity_gates"]["accuracy_gate"] is None


def test_m25_rejects_test_enabled_or_stream_count_changed(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["uses_test_set"] = True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        m25._read_config(bad)
    config["uses_test_set"] = False
    config["streams"] = config["streams"][:-1]
    bad.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="six"):
        m25._read_config(bad)


def test_m25_aggregate_reports_sample_standard_deviation():
    config = m25._read_config(CONFIG)
    payloads = {
        f"s{seed}": _payload(seed, 90.0 + index)
        for index, seed in enumerate(range(2025, 2031))
    }
    result = m25._aggregate(config, payloads)
    assert result["status"] == "PASS_M25_RANPAC_RESNET_CIFAR_TRAIN_ONLY"
    assert result["gates"]["all_six_streams_complete"] is True
    assert result["gates"]["train_only"] is True
    assert result["aggregate"]["exact"]["validation_aia_percent"]["mean"] == pytest.approx(92.5)
    assert result["aggregate"]["exact"]["validation_aia_percent"]["sample_standard_deviation"] == pytest.approx(
        1.8708286933869707
    )


def test_m25_cache_loader_refuses_test_pt(tmp_path):
    config = m25._read_config(CONFIG)
    (tmp_path / "test.pt").write_bytes(b"sentinel")
    with pytest.raises(RuntimeError, match="test.pt"):
        m25._load_train_cache(config, tmp_path)
