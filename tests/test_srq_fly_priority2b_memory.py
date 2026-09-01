"""Protocol and selection gates for SRQ-FLY Priority 2B."""

import json
from pathlib import Path

import pytest
import torch

from tools import srq_fly_priority2b_memory_benchmark as priority2b


ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict:
    payload = json.loads(
        (ROOT / "configs/srq_fly_priority2b_quantization_memory.json").read_text()
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
        quantization_batch_grid=[1, 4],
    )
    return payload


def test_priority2b_config_is_strict_and_synthetic(tmp_path):
    path = tmp_path / "config.json"
    payload = _small_config()
    path.write_text(json.dumps(payload))
    assert priority2b._read_config(path)["quantization_batch_grid"] == [1, 4]
    payload["unknown"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown fields"):
        priority2b._read_config(path)


def test_priority2b_resume_requires_source_config_and_profile_identity(
    tmp_path, monkeypatch
):
    worker_config = tmp_path / "worker.json"
    worker_config.write_text(json.dumps({"study": "resume-test"}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result_path = output_dir / "rep_02_eager_quant.json"
    probe_path = output_dir / "rep_02_eager_quant.probe.pt"
    cached = {
        "status": "complete",
        "method": "optimized_eager_quant_blocked_qr_srq_int8",
        "config_sha256": priority2b._sha256(worker_config),
        "source_identity": priority2b._expected_worker_source(),
        "profiled_task_stage_seconds": None,
    }
    result_path.write_text(json.dumps(cached))
    expected_probe = torch.tensor([[2.0, 3.0]])
    torch.save(expected_probe, probe_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("valid cached worker should resume")

    monkeypatch.setattr(priority2b.subprocess, "run", must_not_run)
    result, probe = priority2b._run_one(
        worker_config=worker_config,
        worker_method="optimized_eager_quant_blocked_qr_srq_int8",
        label="eager_quant",
        repetition=2,
        output_dir=output_dir,
        device="cuda",
        profile_stages=False,
    )
    assert result == cached
    assert torch.equal(probe, expected_probe)


def test_priority2b_selects_lowest_peak_candidate_subject_to_gates(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_small_config()))
    probe = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    def fake_run_one(**kwargs):
        label = kwargs["label"]
        if label == "exact_fly":
            seconds, peak, state = 0.8, 100.0, 4000
            method = "exact_fly_dense"
        elif label == "eager_quant":
            seconds, peak, state = 1.0, 120.0, 1000
            method = "optimized_eager_quant_blocked_qr_srq_int8"
        elif label.endswith("_1"):
            seconds, peak, state = 1.05, 104.0, 1000
            method = "optimized_streaming_quant_blocked_qr_srq_int8"
        else:
            seconds, peak, state = 1.02, 106.0, 1000
            method = "optimized_streaming_quant_blocked_qr_srq_int8"
        profile = kwargs["profile_stages"]
        stage_memory = None
        stage_seconds = None
        if profile:
            stage_memory = [{
                "factor_quantization": {
                    "before_allocated_bytes": 20,
                    "peak_allocated_bytes": int(peak),
                }
            }]
            stage_seconds = [{"factor_quantization": seconds}]
        return {
            "method": method,
            "total_update_seconds": seconds,
            "peak_cuda_allocated_bytes": peak,
            "peak_cuda_reserved_bytes": peak + 10,
            "persistent_state_bytes": state,
            "serialized_checkpoint_bytes": state,
            "solver_relative_residual": 1e-8,
            "profiled_task_stage_seconds": stage_seconds,
            "profiled_task_stage_cuda_memory": stage_memory,
        }, probe.clone()

    monkeypatch.setattr(priority2b, "_run_one", fake_run_one)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fake_git(command, **kwargs):
        return "" if "status" in command else "f" * 40 + "\n"

    monkeypatch.setattr(priority2b.subprocess, "check_output", fake_git)
    result = priority2b.run(
        config_path=config_path,
        output_dir=tmp_path / "output",
        device="cuda",
        require_clean_git=True,
    )
    assert result["status"] == "PASS_REVIEW_PRIORITY2B"
    assert result["uses_test_set"] is False and result["synthetic_only"] is True
    assert result["selected_candidate"]["quantization_batch_blocks"] == 1
    by_label = {row["label"]: row for row in result["summaries"]}
    assert by_label["streaming_quant_batch_1"]["gates"][
        "median_peak_allocated_ratio_to_exact"
    ]
    assert not by_label["streaming_quant_batch_4"]["gates"][
        "median_peak_allocated_ratio_to_exact"
    ]
    assert (tmp_path / "output/priority2b_memory_results.json").is_file()
