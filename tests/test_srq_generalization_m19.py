import json
from pathlib import Path

import pytest
import torch

from tools import srq_generalization_m19 as m19


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m19_panel_system_benchmark.json"


def _tiny_config():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config.update({
        "required_device_name_substring": "",
        "widths": [24],
        "panel_sizes": [4, 8],
        "rows_per_update": 16,
        "num_classes": 5,
        "probe_rows": 8,
        "warmup_repetitions": 0,
        "measured_repetitions": 2,
    })
    config["storage"] = {
        "mode": "int8", "block_size": 8, "group_size": 4,
        "quantization_batch_blocks": 2,
    }
    config["selection"]["reference_panel_size"] = 8
    config["gates"]["maximum_solver_relative_residual"] = 1e-3
    config["gates"]["maximum_relative_logit_drift_from_reference_panel"] = 1e-2
    return m19._validate_config(config)


def test_m19_locked_protocol_is_real_scale_synthetic_and_test_free():
    config = m19._read_config(CONFIG)
    assert config["uses_test_set"] is False and config["synthetic_only"] is True
    assert config["seed"] == 2025
    assert config["widths"] == [10000, 20000]
    assert config["panel_sizes"] == [64, 128, 256, 512, 1024]
    assert config["rows_per_update"] == 5000
    assert tuple(config["timed_stages"]) == m19.STAGES
    assert config["selection"]["uses_accuracy"] is False


def test_m19_rejects_accuracy_selection_and_consuming_contract():
    config = _tiny_config()
    config["selection"]["uses_accuracy"] = True
    with pytest.raises(ValueError, match="accuracy"):
        m19._validate_config(config)
    config = _tiny_config()
    config["update"]["preserve_update_rows"] = False
    with pytest.raises(ValueError, match="non-consuming"):
        m19._validate_config(config)


def test_m19_prior_factor_is_complete_int8_and_has_positive_diagonal():
    config = _tiny_config()
    factor = m19._synthetic_prior_factor(config, 24, torch.device("cpu"))
    decoded = factor.reconstruct_upper(dtype=torch.float32)
    assert factor.mode == "int8" and factor.dimension == 24
    assert torch.all(decoded.diagonal() > 0)
    assert torch.count_nonzero(torch.triu(decoded, diagonal=1)) > 0
    assert torch.count_nonzero(torch.tril(decoded, diagonal=-1)) == 0


def test_m19_atomic_units_time_real_stages_and_summarize(tmp_path):
    config = _tiny_config()
    units = []
    for panel in config["panel_sizes"]:
        unit = m19.run_unit(
            config, width=24, panel=panel, output_dir=tmp_path,
            device="cpu",
        )
        units.append(unit)
        assert set(unit["stage_seconds"]) == set(m19.STAGES)
        assert all(unit["stage_seconds"][stage]["median"] > 0 for stage in m19.STAGES)
        assert unit["state_bytes_identical_across_repetitions"]
        assert unit["predictions_identical_across_repetitions"]
        assert m19._unit_path(tmp_path, 24, panel).is_file()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = m19.summarize(config, units, config_path=config_path)
    assert result["status"] == m19.STATUS_PASS
    assert result["selection"]["selected_panel_size"] in {4, 8}
    assert result["bottleneck"]["stage"] in m19.STAGES
    assert all(result["gates"].values())


def test_m19_warning_preserves_result_when_prediction_gate_fails(tmp_path):
    config = _tiny_config()
    base = {
        "status": "complete", "width": 24,
        "stage_seconds": {stage: {"median": 1.0} for stage in m19.STAGES},
        "total_stage_seconds": {"median": 4.0},
        "analytic_persistent_tensor_bytes": 100,
        "factor_persistent_tensor_bytes": 50,
        "state_bytes_identical_across_repetitions": True,
        "predictions_identical_across_repetitions": True,
        "maximum_relative_logit_drift_across_repetitions": 0.0,
        "maximum_solver_relative_residual": 1e-6,
        "environment": {"device_name": "cpu"},
        "probe_logits": [[1.0, 0.0], [0.0, 1.0]],
    }
    first = dict(base, panel_size=4, predictions=[0, 1])
    second = dict(base, panel_size=8, predictions=[1, 1])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = m19.summarize(config, [first, second], config_path=path)
    assert result["status"] == m19.STATUS_WARNING
    assert result["gates"]["predictions_identical_across_panels"] is False
