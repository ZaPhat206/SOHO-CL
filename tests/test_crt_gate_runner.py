import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import crt_gate_runner, experiment_runner


def gate_args(tmp_path):
    feature_cache = tmp_path / "features"
    gate_cache = tmp_path / "crt-gate-cache"
    output = tmp_path / "gates"
    source_args = SimpleNamespace(dataset="tiny", model_name="synthetic", data_augmentation="none")
    generator = torch.Generator().manual_seed(73)
    labels = torch.arange(3).repeat_interleave(30)
    centers = torch.tensor([
        [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [1.0, 0.0, -0.5, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.5, 0.0, 0.0],
    ])
    train_x = centers[labels] + 0.4 * torch.randn(90, 6, generator=generator)
    test_labels = torch.arange(3).repeat_interleave(4)
    test_x = centers[test_labels] + 0.4 * torch.randn(12, 6, generator=generator)
    experiment_runner.save_cache(
        feature_cache, train_x, labels, test_x, test_labels, source_args
    )
    return SimpleNamespace(
        prepare_cache=True,
        run_gates=True,
        feature_cache_dir=str(feature_cache),
        gate_cache_dir=str(gate_cache),
        output_dir=str(output),
        dataset="tiny",
        model_name="synthetic",
        num_classes=3,
        num_tasks=3,
        validation_fraction=0.2,
        seed=19,
        device="cpu",
        anchor_dim=12,
        synaptic_degree=3,
        coding_level=0.25,
        scatter_epsilon=1e-5,
        statistics_dtype="float64",
        anchor_batch_size=8,
        anchor_ridges="0.1,1.0",
        residual_ridges="0.1",
        complement_ridges="0.1",
        ranks="1,2",
        temperatures="0.5,1.0",
        minimum_full_gain=-100.0,
        maximum_low_rank_gap=100.0,
        minimum_confusion_gain=-100.0,
        maximum_relative_solver_residual=1e-8,
    )


def test_gate_cache_and_all_stages_never_require_test_cache(tmp_path):
    args = gate_args(tmp_path)
    test_path = Path(args.feature_cache_dir) / "test.pt"
    test_path.rename(Path(args.feature_cache_dir) / "test.hidden")

    manifest = crt_gate_runner.prepare_cache(args)
    report = crt_gate_runner.run_gates(args)

    assert manifest["experiment_cache_only"] is True
    assert manifest["allowed_in_learner_checkpoint"] is False
    assert manifest["test_cache_opened"] is False
    assert report["test_cache_opened"] is False
    assert report["held_out_test_authorized"] is True
    assert all(candidate["uses_test_set"] is False for candidate in report["candidates"])
    assert report["gates"]["gate0_numerical_stability"]["pass"] is True
    assert (Path(args.output_dir) / "gate_results.json").is_file()
    # 2 anchor + 1 full + 4 proposed + 2 random + 2 Fisher + 4 shuffled
    # + 4 no-residualization candidates.
    assert len(report["candidates"]) == 19


def test_gate1_failure_stops_before_structured_candidates(tmp_path):
    args = gate_args(tmp_path)
    crt_gate_runner.prepare_cache(args)
    args.minimum_full_gain = 101.0
    report = crt_gate_runner.run_gates(args)

    assert report["status"] == "stopped_after_gate1"
    assert report["held_out_test_authorized"] is False
    assert {candidate["method"] for candidate in report["candidates"]} == {
        "anchor_only", "full_raw_residual",
    }


def test_gate_cache_integrity_failure_is_explicit(tmp_path):
    args = gate_args(tmp_path)
    crt_gate_runner.prepare_cache(args)
    target = Path(args.gate_cache_dir) / "validation_task_00.pt"
    target.write_bytes(target.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="integrity failure"):
        crt_gate_runner.validate_gate_cache(args)


def test_gate_result_is_json_serializable_and_records_thresholds(tmp_path):
    args = gate_args(tmp_path)
    crt_gate_runner.prepare_cache(args)
    report = crt_gate_runner.run_gates(args)
    encoded = json.dumps(report)

    assert "minimum_full_gain" in encoded
    assert report["selected_confusion_residual"]["rank"] in (1, 2)
