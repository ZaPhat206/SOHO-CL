"""Protocol and decision gates for SRQ-FLY Priority 2C."""

import json
from pathlib import Path

import pytest
import torch

from tools import srq_fly_priority2c_memory_benchmark as priority2c


ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict:
    payload = json.loads(
        (ROOT / "configs/srq_fly_priority2c_implicit_ridge_memory.json").read_text()
    )
    payload.update(
        feature_dim=8,
        expand_dim=24,
        synaptic_degree=4,
        block_size=6,
        group_size=4,
        update_panel_size=6,
        rows_per_task=10,
        num_classes=4,
        probe_rows=4,
        warmup_repetitions=0,
        measured_repetitions=3,
    )
    return payload


def test_priority2c_config_is_strict_and_keeps_batch64(tmp_path):
    path = tmp_path / "config.json"
    payload = _small_config()
    path.write_text(json.dumps(payload))
    assert priority2c._read_config(path)["quantization_batch_blocks"] == 64
    payload["quantization_batch_blocks"] = 16
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="batch 64"):
        priority2c._read_config(path)


def test_priority2c_passes_only_for_faithful_faster_lower_peak_candidate(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_small_config()))
    reference = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    def fake_run_one(**kwargs):
        label = kwargs["label"]
        if label == "exact_fly":
            seconds, peak, state = 0.4, 1600.0, 4000
        elif label == "priority2b_batch64":
            seconds, peak, state = 0.8, 1200.0, 1000
        else:
            seconds, peak, state = 0.75, 900.0, 1000
        profile = kwargs["profile_stages"]
        return {
            "method": priority2c.METHODS[label],
            "total_update_seconds": seconds,
            "peak_cuda_allocated_bytes": peak,
            "peak_cuda_reserved_bytes": peak + 10,
            "persistent_state_bytes": state,
            "solver_relative_residual": 1e-8,
            "profiled_task_stage_seconds": [{}] if profile else None,
            "profiled_task_stage_cuda_memory": [{}] if profile else None,
        }, reference.clone()

    monkeypatch.setattr(priority2c, "_run_one", fake_run_one)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fake_git(command, **kwargs):
        return "" if "status" in command else "f" * 40 + "\n"

    monkeypatch.setattr(priority2c.subprocess, "check_output", fake_git)
    result = priority2c.run(
        config_path=config_path,
        output_dir=tmp_path / "output",
        device="cuda",
        require_clean_git=True,
    )
    assert result["status"] == "PASS_REVIEW_PRIORITY2C"
    assert result["uses_test_set"] is False and result["synthetic_only"] is True
    assert all(result["gates"].values())
    assert result["candidate_backend"]["first_update_backend"] == "implicit_ridge_qr"
    assert (tmp_path / "output/priority2c_memory_results.json").is_file()
