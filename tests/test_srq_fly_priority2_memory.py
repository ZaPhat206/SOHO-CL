"""Protocol and selection gates for SRQ-FLY Priority 2A."""

import json
from pathlib import Path

import pytest
import torch

from tools import srq_fly_priority2_memory_benchmark as priority2


ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict:
    payload = json.loads(
        (ROOT / "configs/srq_fly_priority2a_memory.json").read_text()
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
        trailing_chunk_sizes=[5, 7],
    )
    return payload


def test_priority2_config_is_strict_and_train_free(tmp_path):
    path = tmp_path / "config.json"
    payload = _small_config()
    path.write_text(json.dumps(payload))
    assert priority2._read_config(path)["trailing_chunk_sizes"] == [5, 7]
    payload["unknown"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown fields"):
        priority2._read_config(path)


def test_priority2_worker_resume_requires_locked_identity_and_profile_mode(
    tmp_path, monkeypatch
):
    worker_config = tmp_path / "worker.json"
    worker_config.write_text(json.dumps({"study": "resume-test"}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result_path = output_dir / "rep_03_exact_fly.json"
    probe_path = output_dir / "rep_03_exact_fly.probe.pt"
    source_identity = {
        "runner": priority2._sha256(
            ROOT / "tools/srq_fly_system_benchmark.py"
        ),
        "optimized_learner": priority2._sha256(
            ROOT / "methods/srq_fly_optimized/learner.py"
        ),
        "optimized_storage": priority2._sha256(
            ROOT / "methods/srq_fly_optimized/storage.py"
        ),
    }
    cached = {
        "status": "complete",
        "method": "exact_fly_dense",
        "config_sha256": priority2._sha256(worker_config),
        "source_identity": source_identity,
        "profiled_task_stage_seconds": None,
    }
    result_path.write_text(json.dumps(cached))
    expected_probe = torch.tensor([[1.0, 2.0]])
    torch.save(expected_probe, probe_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("valid resumable worker should not be relaunched")

    monkeypatch.setattr(priority2.subprocess, "run", must_not_run)
    result, probe = priority2._run_one(
        worker_config=worker_config,
        worker_method="exact_fly_dense",
        label="exact_fly",
        repetition=3,
        output_dir=output_dir,
        device="cuda",
        profile_stages=False,
    )
    assert result == cached
    assert torch.equal(probe, expected_probe)


def test_priority2_selects_lowest_peak_candidate_subject_to_all_gates(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_small_config()))
    probe = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    def fake_run_one(**kwargs):
        label = kwargs["label"]
        if label == "exact_fly":
            seconds, peak, state = 1.0, 100.0, 4000
            method = "exact_fly_dense"
        elif label == "unchunked_blocked_qr":
            seconds, peak, state = 1.4, 120.0, 1000
            method = "optimized_blocked_qr_srq_int8"
        elif label.endswith("_5"):
            seconds, peak, state = 1.3, 104.0, 1000
            method = "optimized_chunked_blocked_qr_srq_int8"
        else:
            seconds, peak, state = 1.2, 106.0, 1000
            method = "optimized_chunked_blocked_qr_srq_int8"
        row = {
            "method": method,
            "total_update_seconds": seconds,
            "peak_cuda_allocated_bytes": peak,
            "peak_cuda_reserved_bytes": peak + 10,
            "persistent_state_bytes": state,
            "serialized_checkpoint_bytes": state,
            "solver_relative_residual": 1e-8,
            "profiled_task_stage_seconds": [] if kwargs["profile_stages"] else None,
            "profiled_task_stage_cuda_memory": [] if kwargs["profile_stages"] else None,
        }
        return row, probe.clone()

    monkeypatch.setattr(priority2, "_run_one", fake_run_one)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fake_git(command, **kwargs):
        return "" if "status" in command else "f" * 40 + "\n"

    monkeypatch.setattr(priority2.subprocess, "check_output", fake_git)
    result = priority2.run(
        config_path=config_path,
        output_dir=tmp_path / "output",
        device="cuda",
        require_clean_git=True,
    )
    assert result["status"] == "PASS_REVIEW_PRIORITY2A"
    assert result["uses_test_set"] is False and result["synthetic_only"] is True
    assert result["selected_candidate"]["chunk_size"] == 5
    by_label = {row["label"]: row for row in result["summaries"]}
    assert by_label["chunked_blocked_qr_5"]["gates"][
        "median_peak_allocated_ratio_to_exact"
    ]
    assert not by_label["chunked_blocked_qr_7"]["gates"][
        "median_peak_allocated_ratio_to_exact"
    ]
    assert (tmp_path / "output/priority2a_memory_results.json").is_file()
