"""Contract tests for the train-only M23 multistream wrapper."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import srq_generalization_m23 as m23
from tools import srq_generalization_m5 as m5


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "srq_generalization_m23_equal_budget_multistream_train_only.json"


def _payload(methods=None, *, seed=2025):
    methods = methods or m23.METHODS
    records = []
    for task in range(1, 3):
        records.append(
            {
                "task": task,
                "accuracy_percent": {name: 80.0 + task for name in methods},
                "state": {
                    name: {
                        "total_persistent_bytes": 100 + index,
                        "solver_relative_residual": 0.0,
                    }
                    for index, name in enumerate(methods)
                },
            }
        )
    summary = {
        "validation_aia_percent": {name: 81.0 for name in methods},
        "final_validation_accuracy_percent": {name: 82.0 for name in methods},
        "final_total_persistent_bytes": {name: 100 + index for index, name in enumerate(methods)},
        "analytic_update_seconds": {name: 1.0 + index for index, name in enumerate(methods)},
        "representation_encoding_seconds": {name: 0.1 for name in methods},
        "p2b_validation_aia_loss_pp": 0.0,
        "best_equal_or_lower_state_aia_advantage_over_p2b_pp": 0.0,
        "pareto_dominators_of_p2b": [],
        "maximum_solver_relative_residual": 0.0,
    }
    return {
        "schema_version": 1,
        "study_id": "srq-generalization-m23-equal-budget-multistream-train-only-v1",
        "stream_id": f"s{seed}",
        "status": "PASS_M5_EQUAL_BUDGET_TRAIN_ONLY",
        "uses_test_set": False,
        "gates": {"valid": True},
        "budget_lock": m5._derive_budget_lock(m23._read_config(CONFIG)),
        "summary": summary,
        "groups": [{"records": records}],
        "provenance": {"stream_seed": seed},
        "ridge_selection": {},
    }


def test_m23_config_has_only_stream_generalization_and_locks_budget():
    config = m23._read_config(CONFIG)
    m5_config = json.loads(
        (ROOT / "configs" / "srq_generalization_m5_equal_budget_train_only.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["streams"] == list(m23.STREAMS)
    assert "seed" not in config
    assert "projection_seed" not in config["ranpac"]
    assert "seed" not in config["countsketch"]
    # All method-defining fields are byte-for-byte inherited from M5.
    for key in set(m5_config) - {"seed", "study_id"}:
        if key == "ranpac":
            expected = {k: v for k, v in m5_config[key].items() if k != "projection_seed"}
            assert config[key] == expected
        elif key == "countsketch":
            expected = {k: v for k, v in m5_config[key].items() if k != "seed"}
            assert config[key] == expected
        else:
            assert config[key] == m5_config[key]
    assert m5._derive_budget_lock(config) == {
        "policy": "largest_integer_dimension_not_exceeding_target_bytes",
        "locked_before_representation_encoding_or_accuracy": True,
        "target_p2b_total_persistent_bytes": 91880088,
        "full_expand_dimension": 10000,
        "byte_matched_exact_dimension": 4333,
        "byte_matched_exact_total_persistent_bytes": 91877332,
        "byte_matched_exact_underfill_bytes": 2756,
        "byte_matched_exact_underfill_fraction": 2.9995617766495826e-05,
        "countsketch_dimension": 3809,
        "countsketch_total_persistent_bytes": 91851524,
        "countsketch_underfill_bytes": 28564,
        "countsketch_underfill_fraction": 0.00031088346367278184,
        "raw_ridge_total_persistent_bytes": 2974096,
    }


def test_m23_rejects_test_use_and_wrong_streams(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["uses_test_set"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        m23._read_config(path)
    config["uses_test_set"] = False
    config["streams"][1]["seed"] = 7
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="streams"):
        m23._read_config(path)


def test_m23_aggregate_uses_sample_std_and_all_streams():
    config = m23._read_config(CONFIG)
    payloads = {f"s{seed}": _payload(seed=seed) for seed in (2025, 2026, 2027)}
    for index, seed in enumerate((2025, 2026, 2027)):
        payloads[f"s{seed}"]["summary"]["validation_aia_percent"]["p2b_int8"] = 80.0 + index
        payloads[f"s{seed}"]["summary"]["final_validation_accuracy_percent"]["p2b_int8"] = 81.0 + index
    result = m23.summarize_streams(
        config=config, stream_payloads=payloads, original_m5_artifact=Path("does-not-exist.zip")
    )
    assert result["status"] == "FAIL_M23_EQUAL_BUDGET_MULTISTREAM_TRAIN_ONLY"
    assert result["gates"]["all_three_streams_complete"] is True
    assert result["gates"]["stream_s2025_matches_original_m5_artifact"] is False
    assert result["gates"]["budget_lock_identical_across_streams"] is True
    assert result["aggregate"]["p2b_int8"]["aia_mean"] == pytest.approx(81.0)
    assert result["aggregate"]["p2b_int8"]["aia_std"] == pytest.approx(1.0)
    assert result["aggregate"]["p2b_int8"]["final_accuracy_std"] == pytest.approx(1.0)


def test_m23_stream_completion_requires_train_only_and_all_gates():
    payload = _payload(seed=2025)
    assert m23._stream_result_is_complete(payload, m23.STREAMS[0])
    payload = copy.deepcopy(payload)
    payload["uses_test_set"] = True
    assert not m23._stream_result_is_complete(payload, m23.STREAMS[0])


def test_m23_never_changes_m5_source():
    # This is a source-level guard for the explicit M23 non-modification rule.
    assert m5.ROOT / "tools" / "srq_generalization_m5.py"
    assert "srq_generalization_m5.py" not in (ROOT / "tools" / "srq_generalization_m23.py").read_text(
        encoding="utf-8"
    ).split("from tools import srq_generalization_m5", 1)[-1]
