import argparse
import json
from pathlib import Path

import pytest

from tools import srq_fly_priority5_memory as priority5


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_fly_priority5_cifar100_whole_process_memory.json"


def test_locked_config_and_train_only_contract(tmp_path):
    config = priority5._read_config(CONFIG)
    assert config["methods"] == list(priority5.METHODS)
    assert config["representation"]["expand_dim"] == 10000
    assert config["p2b_backend"]["quantization_backend"] == "streaming"
    source = (ROOT / "tools/srq_fly_priority5_memory.py").read_text(encoding="utf-8")
    assert "datasets.CIFAR100" in source
    assert "train=True" in source
    assert "train=False" not in source
    assert "test.pt" not in source

    broken = dict(config)
    broken["methods"] = list(reversed(config["methods"]))
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="locked"):
        priority5._read_config(path)


def test_priority5_composes_repository_transform_list():
    source = (ROOT / "tools/srq_fly_priority5_memory.py").read_text(
        encoding="utf-8"
    )
    assert "transforms.Compose([" in source
    assert "*build_transform(is_cifar=True, data_augmentation=\"vit\")" in source


def test_monitor_attributes_process_and_stage_peaks(tmp_path, monkeypatch):
    marker = tmp_path / "stage.json"
    marker.write_text(json.dumps({"stage": "analytic_update"}), encoding="utf-8")

    class Process:
        pid = 77

        def __init__(self):
            self.calls = 0

        def poll(self):
            self.calls += 1
            return None if self.calls <= 3 else 0

    class Sampler:
        def __init__(self):
            self.calls = 0

        def sample(self, pid):
            assert pid == 77
            self.calls += 1
            return 1000 + self.calls * 100, 200 + self.calls * 50

    monkeypatch.setattr(priority5.time, "sleep", lambda _: None)
    result = priority5._monitor_worker(Process(), marker, Sampler(), 0.02)
    assert result["peak_worker_process_bytes"] == 450
    assert result["peak_device_bytes"] == 1500
    assert result["worker_sample_count"] == 4
    assert result["stage_peaks"]["analytic_update"]["process_bytes"] == 400


def _method(method, state, analytic_peak, predictions):
    return {
        "method": method, "status": "complete", "uses_test_set": False,
        "test_features_materialized": False, "train_samples": 50000,
        "class_order": list(range(100)), "projection_sha256": "same",
        "persistent_state_bytes": state, "solver_relative_residual": 1e-6,
        "torch_cuda_stages": {
            "analytic_update": {"peak_allocated_bytes": analytic_peak}
        },
        "probe_predictions": predictions,
        "probe_logits": [[1.0, 0.0], [0.0, 1.0]],
    }


def _monitor(method, analytic_peak, process_peak):
    return {
        "method": method, "worker_sample_count": 100,
        "peak_worker_process_bytes": process_peak,
        "observed_stages": [
            "backbone_load", "feature_extraction", "analytic_update", "final_probe"
        ],
        "stage_peaks": {
            "analytic_update": {"process_bytes": analytic_peak}
        },
    }


def test_summary_separates_persistent_allocator_and_nvml_metrics():
    config = priority5._read_config(CONFIG)
    results = [
        _method("exact_fly_10000", 400, 1000, [0, 1]),
        _method("srq_fly_p2b_10000", 80, 800, [0, 1]),
    ]
    monitors = [
        _monitor("exact_fly_10000", 1200, 1400),
        _monitor("srq_fly_p2b_10000", 900, 1100),
    ]
    summary = priority5._summarize(config, results, monitors)
    assert summary["status"] == "PASS_PRIORITY5_MEMORY"
    assert summary["comparisons"]["srq_state_fraction_of_exact"] == 0.2
    assert summary["comparisons"]["srq_analytic_torch_peak_ratio"] == 0.8
    assert summary["comparisons"]["srq_analytic_nvml_process_peak_ratio"] == 0.75
    assert summary["comparisons"]["whole_process_nvml_process_peak_ratio"] == pytest.approx(1100 / 1400)


def test_summary_does_not_gate_on_whole_process_peak():
    config = priority5._read_config(CONFIG)
    results = [
        _method("exact_fly_10000", 400, 1000, [0, 1]),
        _method("srq_fly_p2b_10000", 80, 800, [0, 1]),
    ]
    monitors = [
        _monitor("exact_fly_10000", 1200, 1400),
        _monitor("srq_fly_p2b_10000", 900, 1500),
    ]
    summary = priority5._summarize(config, results, monitors)
    assert summary["status"] == "PASS_PRIORITY5_MEMORY"
    assert "whole_process_peak_reduced" not in summary["gates"]
