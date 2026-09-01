"""Correctness gates for the opt-in SRQ-FLY update optimization."""

import json
from pathlib import Path

import pytest
import torch

from methods.srq_fly import CompressedUpper as LockedCompressedUpper
from methods.srq_fly import SquareRootFLYLearner as LockedSquareRootFLYLearner
from methods.srq_fly_optimized import CompressedUpper, SquareRootFLYLearner
from methods.srq_fly_optimized.learner import _blocked_qr_rank_update
from tools import srq_fly_system_benchmark, srq_fly_update_benchmark


ROOT = Path(__file__).resolve().parents[1]


def _kwargs(**overrides):
    arguments = dict(
        feature_dim=7,
        expand_dim=24,
        synaptic_degree=4,
        coding_level=0.25,
        ridge_lambda=100.0,
        block_size=6,
        group_size=5,
        seed=2025,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    arguments.update(overrides)
    return arguments


def _stream():
    generator = torch.Generator().manual_seed(709)
    first = torch.randn(17, 24, generator=generator, dtype=torch.float64)
    second = torch.randn(13, 24, generator=generator, dtype=torch.float64)
    first_labels = torch.tensor(
        [9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2]
    )
    second_labels = torch.tensor([11, 5, 2, 11, 5, 2, 11, 5, 2, 11, 5, 2, 11])
    return first, first_labels, second, second_labels


def _spd(seed=701, dimension=23):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    return values.T @ values + 3.0 * torch.eye(dimension, dtype=torch.float64)


@pytest.mark.parametrize("mode", ["float16", "int8"])
def test_vectorized_storage_is_byte_identical_to_locked_blockwise_storage(mode):
    matrix = torch.linalg.cholesky(_spd()).T
    locked = LockedCompressedUpper.from_upper(
        matrix, block_size=6, group_size=7, mode=mode
    )
    optimized = CompressedUpper.from_upper(
        matrix, block_size=6, group_size=7, mode=mode
    )
    torch.testing.assert_close(
        optimized.reconstruct_upper(dtype=torch.float64),
        locked.reconstruct_upper(dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    for left, right in zip(optimized.blocks, locked.blocks):
        assert torch.equal(left.values, right.values)
        if mode == "int8":
            assert torch.equal(left.scales, right.scales)


@pytest.mark.parametrize("mode", ["float16", "int8"])
def test_fused_inplace_reconstruction_preserves_version1_storage(mode):
    original = torch.linalg.cholesky(_spd()).T
    expected = CompressedUpper.from_upper(
        original, block_size=6, group_size=7, mode=mode
    )
    work = original.clone()
    fused, relative_error = CompressedUpper.from_upper_inplace(
        work, block_size=6, group_size=7, mode=mode
    )
    torch.testing.assert_close(
        work, expected.reconstruct_upper(dtype=torch.float64), rtol=0, atol=0
    )
    for left, right in zip(fused.blocks, expected.blocks):
        assert torch.equal(left.values, right.values)
        if mode == "int8":
            assert torch.equal(left.scales, right.scales)
    reference_error = float(
        torch.linalg.vector_norm(work - original)
        / torch.linalg.vector_norm(original)
    )
    assert relative_error == pytest.approx(reference_error, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("mode", ["float16", "int8"])
def test_stacked_qr_matches_gram_cholesky_stream(mode):
    first, first_labels, second, second_labels = _stream()
    legacy_backend = SquareRootFLYLearner(
        storage_mode=mode, update_backend="gram_cholesky", **_kwargs()
    )
    qr_backend = SquareRootFLYLearner(
        storage_mode=mode, update_backend="stacked_qr", **_kwargs()
    )
    for codes, labels in ((first, first_labels), (second, second_labels)):
        legacy_backend.update_codes(codes, labels)
        qr_backend.update_codes(codes, labels)
    torch.testing.assert_close(
        qr_backend.factor.reconstruct_upper(dtype=torch.float64),
        legacy_backend.factor.reconstruct_upper(dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(qr_backend.weights, legacy_backend.weights, rtol=0, atol=0)
    assert qr_backend.diagnostics["solver_relative_residual"] < 1e-10


def test_blocked_rank_update_matches_dense_stacked_qr():
    upper = torch.linalg.cholesky(_spd(dimension=31)).T
    generator = torch.Generator().manual_seed(991)
    updates = torch.randn(9, 31, generator=generator, dtype=torch.float64)
    _, expected = torch.linalg.qr(torch.cat((upper, updates)), mode="r")
    signs = torch.where(expected.diagonal() < 0, -1.0, 1.0)
    expected = signs[:, None] * expected
    actual = _blocked_qr_rank_update(upper.clone(), updates, panel_size=7)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    assert bool((actual.diagonal() > 0).all())


@pytest.mark.parametrize("mode", ["float16", "int8"])
def test_blocked_qr_backend_matches_full_qr_stream(mode):
    first, first_labels, second, second_labels = _stream()
    full = SquareRootFLYLearner(
        storage_mode=mode, update_backend="stacked_qr", **_kwargs()
    )
    blocked = SquareRootFLYLearner(
        storage_mode=mode,
        update_backend="blocked_qr",
        update_panel_size=7,
        **_kwargs(),
    )
    for codes, labels in ((first, first_labels), (second, second_labels)):
        full.update_codes(codes, labels)
        blocked.update_codes(codes, labels)
    torch.testing.assert_close(
        blocked.factor.reconstruct_upper(dtype=torch.float64),
        full.factor.reconstruct_upper(dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(blocked.weights, full.weights, rtol=2e-12, atol=2e-12)
    assert blocked.persistent_state_bytes() == full.persistent_state_bytes()


@pytest.mark.parametrize("mode", ["float16", "int8"])
def test_direct_cholesky_backend_matches_compatibility_backend(mode):
    first, first_labels, second, second_labels = _stream()
    compatibility = SquareRootFLYLearner(
        storage_mode=mode, update_backend="gram_cholesky", **_kwargs()
    )
    direct = SquareRootFLYLearner(
        storage_mode=mode, update_backend="gram_cholesky_direct", **_kwargs()
    )
    for codes, labels in ((first, first_labels), (second, second_labels)):
        compatibility.update_codes(codes, labels)
        direct.update_codes(codes, labels)
    torch.testing.assert_close(
        direct.factor.reconstruct_upper(dtype=torch.float64),
        compatibility.factor.reconstruct_upper(dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        direct.weights, compatibility.weights, rtol=2e-6, atol=1e-10
    )
    assert direct.persistent_state_bytes() == compatibility.persistent_state_bytes()


def test_profiler_is_opt_in_and_does_not_enter_persistent_state():
    first, labels, _, _ = _stream()
    learner = SquareRootFLYLearner(
        storage_mode="int8", profile_updates=True, **_kwargs()
    )
    learner.update_codes(first, labels)
    timings = learner.diagnostics["last_update_stage_seconds"]
    assert {
        "class_expansion",
        "current_gram",
        "cholesky",
        "factor_quantization",
        "factor_reconstruction",
        "cross_update",
        "triangular_solve",
        "diagnostics",
    } <= set(timings)
    assert all(value >= 0 for value in timings.values())
    assert learner.diagnostics["last_update_total_seconds"] >= sum(timings.values())
    assert not any("timing" in name for name in learner.persistent_tensors())


def test_locked_checkpoint_loads_and_resumes_in_optimized_default_backend():
    first, first_labels, second, second_labels = _stream()
    locked = LockedSquareRootFLYLearner(storage_mode="int8", **_kwargs())
    locked.update_codes(first, first_labels)
    optimized = SquareRootFLYLearner(storage_mode="int8", **_kwargs())
    optimized.load_state_dict(locked.state_dict())
    torch.testing.assert_close(
        optimized.predict_logits_from_codes(first[:5]),
        locked.predict_logits_from_codes(first[:5]),
        rtol=0,
        atol=0,
    )
    locked.update_codes(second, second_labels)
    optimized.update_codes(second, second_labels)
    torch.testing.assert_close(optimized.weights, locked.weights, rtol=0, atol=0)
    assert optimized.persistent_state_bytes() == locked.persistent_state_bytes()


def test_backend_identity_is_checkpoint_locked():
    first, labels, _, _ = _stream()
    learner = SquareRootFLYLearner(
        storage_mode="int8", update_backend="stacked_qr", **_kwargs()
    )
    learner.update_codes(first, labels)
    with pytest.raises(ValueError, match="update backend"):
        SquareRootFLYLearner(storage_mode="int8", **_kwargs()).load_state_dict(
            learner.state_dict()
        )


def test_blocked_backend_panel_size_is_checkpoint_locked():
    first, labels, _, _ = _stream()
    learner = SquareRootFLYLearner(
        storage_mode="int8", update_backend="blocked_qr",
        update_panel_size=7, **_kwargs(),
    )
    learner.update_codes(first, labels)
    with pytest.raises(ValueError, match="panel size"):
        SquareRootFLYLearner(
            storage_mode="int8", update_backend="blocked_qr",
            update_panel_size=8, **_kwargs(),
        ).load_state_dict(learner.state_dict())


def test_synthetic_benchmark_runner_passes_without_test_data(tmp_path):
    config = json.loads(
        (ROOT / "configs/srq_fly_update_optimization_smoke.json").read_text()
    )
    config.update(
        feature_dim=8,
        expand_dim=32,
        synaptic_degree=4,
        block_size=8,
        group_size=4,
        rows_per_task=12,
        num_classes=4,
        probe_rows=4,
        maximum_update_ratio_to_exact_fly=1000.0,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "result.json"
    result = srq_fly_update_benchmark.run(config_path, output, "cpu")
    assert result["status"] == "pass"
    assert result["uses_test_set"] is False and result["synthetic_only"] is True
    assert output.is_file()


def test_isolated_system_benchmark_reports_disjoint_measurements(tmp_path):
    config = json.loads(
        (ROOT / "configs/srq_fly_update_optimization_smoke.json").read_text()
    )
    config.update(
        feature_dim=8,
        expand_dim=24,
        synaptic_degree=4,
        block_size=6,
        group_size=4,
        rows_per_task=10,
        num_classes=4,
        probe_rows=4,
        maximum_update_ratio_to_exact_fly=1000.0,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = srq_fly_system_benchmark.run_isolated(
        config_path=config_path,
        output_dir=tmp_path / "output",
        device_name="cpu",
    )
    assert result["status"] == "pass"
    assert result["uses_test_set"] is False
    assert len(result["results"]) == 6
    assert all(row["peak_cuda_allocated_bytes"] is None for row in result["results"])
    assert result["gates"]["direct_backend_within_tolerance"]
    assert result["gates"]["blocked_qr_backend_within_tolerance"]
    assert result["selected_update_backend"]["name"] == "blocked_qr"
    profiles = {
        row["method"]: row["profiled_task_stage_seconds"]
        for row in result["results"]
    }
    assert profiles["optimized_blocked_qr_srq_int8"] is not None
    assert profiles["exact_fly_dense"] is None
