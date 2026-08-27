import json
from pathlib import Path

import pytest
import torch

from tools import soho_imagenetr_gcv_diagnostic as diagnostic


ROOT = Path(__file__).resolve().parents[1]


def test_locked_protocol_and_method_identity_are_valid():
    protocol = diagnostic._read_protocol(
        ROOT / "configs/soho_imagenetr_gcv_diagnostic.json"
    )
    assert diagnostic._verify_method_identity(protocol) == protocol["method_identity"]
    assert protocol["diagnostic"]["uses_test_set"] is False
    assert protocol["diagnostic"]["held_out_test_authorized"] is False


def test_soho_dual_evaluation_reuses_one_state_without_test_data():
    generator = torch.Generator().manual_seed(29)
    features = torch.randn(24, 4, generator=generator)
    labels = torch.tensor([0, 1, 2, 3] * 6)
    protocol = {
        "backbone": {"feature_dim": 4},
        "dataset": {"num_classes": 4},
        "soho": {
            "expand_dim": 16,
            "olda_dim": 4,
            "density": 0.5,
            "coding_level": 0.25,
            "use_etf": True,
            "gcv_ridge_lower": -1,
            "gcv_ridge_upper": 2,
            "posthoc_fixed_ridge": 1.0,
            "replay_chunk_size": 8,
            "gcv_sample_size": 8,
        },
        "diagnostic": {"probe_rows": 4, "seed": 2025},
    }
    fit_parts = [torch.arange(0, 8), torch.arange(8, 16)]
    validation_parts = [torch.arange(16, 20), torch.arange(20, 24)]
    result = diagnostic._soho_dual_evaluation(
        protocol,
        {"features": features, "labels": labels},
        fit_parts,
        validation_parts,
        projection_seed=2025,
        device_name="cpu",
    )
    assert set(result["methods"]) == {
        "soho_current_gcv",
        "soho_fixed_1000_posthoc",
    }
    for method in result["methods"].values():
        assert method["uses_test_set"] is False
        assert method["state_audit"]["exemplar_free"] is False
        assert len(method["accuracy_matrix"]) == 2
    assert result["stage_diagnostics"][0]["probe_support_turnover"] is None
    assert 0 <= result["stage_diagnostics"][1]["probe_support_turnover"] <= 1


def test_summary_keeps_posthoc_control_out_of_authorization():
    def method(aia, final):
        return {
            "average_incremental_accuracy": aia,
            "final_accuracy": final,
            "forgetting": 1.0,
            "stage_accuracy": [aia, final],
            "selected_ridge_by_task": [0.1, 1000.0],
        }

    replicates = []
    for index in range(6):
        replicates.append(
            {
                "methods": {
                    "soho_current_gcv": method(70 + index, 71 + index),
                    "soho_fixed_1000_posthoc": method(75 + index, 72 + index),
                    "flycl_fidelity": method(74 + index, 71.5 + index),
                },
                "stage_diagnostics": [
                    {"probe_support_turnover": None},
                    {"probe_support_turnover": 0.5},
                ],
            }
        )
    protocol = {
        "study_id": "test",
        "diagnostic": {"replicates": [{"x": i} for i in range(6)]},
        "soho": {},
        "fly_original": {},
    }
    summary = diagnostic._summarize(replicates, protocol)
    assert summary["status"] == "POSTHOC_DIAGNOSTIC_COMPLETE"
    assert summary["uses_test_set"] is False
    assert summary["held_out_test_authorized"] is False
    assert summary["paired_differences"][
        "soho_fixed_1000_posthoc_minus_soho_current_gcv"
    ]["average_incremental_accuracy"]["mean"] == pytest.approx(5.0)
    json.dumps(summary)


def test_colab_notebook_is_train_only_and_pins_the_diagnostic_sources():
    notebook = json.loads(
        (ROOT / "notebooks/soho_imagenetr_gcv_diagnostic_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    protocol_path = ROOT / "configs/soho_imagenetr_gcv_diagnostic.json"
    runner_path = ROOT / "tools/soho_imagenetr_gcv_diagnostic.py"
    assert diagnostic.base._sha256_file(protocol_path) in source
    assert diagnostic.base._sha256_file(runner_path) in source
    assert "--extract-train-only" in source
    assert "extract-test" not in source
    assert "held_out_test_authorized" in source
    assert "test.pt absent" in source
