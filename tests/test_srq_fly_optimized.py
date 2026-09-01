"""Correctness gates for the opt-in SRQ-FLY update optimization."""

import json
from pathlib import Path

import pytest
import torch

from methods.srq_fly import CompressedUpper as LockedCompressedUpper
from methods.srq_fly import SquareRootFLYLearner as LockedSquareRootFLYLearner
from methods.srq_fly_optimized import CompressedUpper, SquareRootFLYLearner
from methods.srq_fly_optimized.learner import _blocked_qr_rank_update
from methods.srq_fly_optimized import storage as optimized_storage
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
@pytest.mark.parametrize("batch_blocks", [1, 3, 16])
def test_streaming_inplace_encoder_is_byte_identical_to_eager(
    mode, batch_blocks
):
    original = torch.linalg.cholesky(_spd(dimension=37)).T
    eager_work = original.clone()
    streaming_work = original.clone()
    eager, eager_error = CompressedUpper.from_upper_inplace(
        eager_work, block_size=8, group_size=7, mode=mode
    )
    streaming, streaming_error = CompressedUpper.from_upper_inplace_streaming(
        streaming_work,
        block_size=8,
        group_size=7,
        mode=mode,
        maximum_batched_blocks=batch_blocks,
    )
    torch.testing.assert_close(streaming_work, eager_work, rtol=0, atol=0)
    assert streaming_error == pytest.approx(eager_error, rel=1e-12, abs=1e-15)
    for left, right in zip(streaming.blocks, eager.blocks):
        assert torch.equal(left.values, right.values)
        if mode == "int8":
            assert torch.equal(left.scales, right.scales)


def test_streaming_encoder_bounds_validation_and_quantization_batches(monkeypatch):
    dimension = 1025
    generator = torch.Generator().manual_seed(702)
    matrix = torch.triu(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float32)
    )
    matrix.diagonal().add_(20.0)
    full_elements = matrix.numel()
    observed_finite_elements = []
    observed_batch_rows = []
    original_isfinite = torch.isfinite
    original_encoder = optimized_storage._groupwise_int8_rows

    def recording_isfinite(values):
        observed_finite_elements.append(values.numel())
        return original_isfinite(values)

    def recording_encoder(values, group_size):
        observed_batch_rows.append(values.shape[0])
        return original_encoder(values, group_size)

    monkeypatch.setattr(torch, "isfinite", recording_isfinite)
    monkeypatch.setattr(
        optimized_storage, "_groupwise_int8_rows", recording_encoder
    )
    CompressedUpper.from_upper_inplace_streaming(
        matrix,
        block_size=128,
        group_size=64,
        mode="int8",
        maximum_batched_blocks=3,
    )
    assert observed_finite_elements
    assert max(observed_finite_elements) < full_elements
    assert observed_batch_rows and max(observed_batch_rows) <= 3


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


def test_chunked_blocked_rank_update_matches_unchunked_and_bounds_rhs(monkeypatch):
    upper = torch.linalg.cholesky(_spd(dimension=31)).T
    generator = torch.Generator().manual_seed(992)
    updates = torch.randn(9, 31, generator=generator, dtype=torch.float64)
    expected = _blocked_qr_rank_update(
        upper.clone(), updates, panel_size=7, trailing_chunk_size=None
    )
    original_updates = updates.clone()
    original_ormqr = torch.ormqr
    observed_widths = []

    def recording_ormqr(reflectors, tau, other, *, left=True, transpose=True):
        observed_widths.append(other.shape[1])
        return original_ormqr(
            reflectors, tau, other, left=left, transpose=transpose
        )

    monkeypatch.setattr(torch, "ormqr", recording_ormqr)
    actual = _blocked_qr_rank_update(
        upper.clone(), updates, panel_size=7, trailing_chunk_size=5
    )
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(updates, original_updates, rtol=0, atol=0)
    assert observed_widths and max(observed_widths) <= 5


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
def test_chunked_blocked_qr_backend_matches_unchunked_stream(mode):
    first, first_labels, second, second_labels = _stream()
    unchunked = SquareRootFLYLearner(
        storage_mode=mode,
        update_backend="blocked_qr",
        update_panel_size=7,
        **_kwargs(),
    )
    chunked = SquareRootFLYLearner(
        storage_mode=mode,
        update_backend="blocked_qr",
        update_panel_size=7,
        update_trailing_chunk_size=5,
        **_kwargs(),
    )
    for codes, labels in ((first, first_labels), (second, second_labels)):
        unchunked.update_codes(codes, labels)
        chunked.update_codes(codes, labels)
    torch.testing.assert_close(
        chunked.factor.reconstruct_upper(dtype=torch.float64),
        unchunked.factor.reconstruct_upper(dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        chunked.weights, unchunked.weights, rtol=2e-12, atol=2e-12
    )
    assert chunked.persistent_state_bytes() == unchunked.persistent_state_bytes()


@pytest.mark.parametrize("mode", ["float16", "int8"])
@pytest.mark.parametrize("batch_blocks", [1, 4, 16])
def test_streaming_quantization_backend_matches_eager_stream(mode, batch_blocks):
    first, first_labels, second, second_labels = _stream()
    eager = SquareRootFLYLearner(
        storage_mode=mode,
        update_backend="blocked_qr",
        update_panel_size=7,
        **_kwargs(),
    )
    streaming = SquareRootFLYLearner(
        storage_mode=mode,
        update_backend="blocked_qr",
        update_panel_size=7,
        quantization_backend="streaming",
        quantization_batch_blocks=batch_blocks,
        **_kwargs(),
    )
    for codes, labels in ((first, first_labels), (second, second_labels)):
        eager.update_codes(codes, labels)
        streaming.update_codes(codes, labels)
    torch.testing.assert_close(
        streaming.factor.reconstruct_upper(dtype=torch.float64),
        eager.factor.reconstruct_upper(dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(streaming.Q, eager.Q, rtol=0, atol=0)
    torch.testing.assert_close(streaming.weights, eager.weights, rtol=0, atol=0)
    assert streaming.persistent_state_bytes() == eager.persistent_state_bytes()


def test_streaming_quantization_is_checkpoint_compatible_with_eager():
    first, labels, _, _ = _stream()
    streaming = SquareRootFLYLearner(
        storage_mode="int8",
        update_backend="blocked_qr",
        quantization_backend="streaming",
        quantization_batch_blocks=3,
        **_kwargs(),
    )
    streaming.update_codes(first, labels)
    eager = SquareRootFLYLearner(
        storage_mode="int8", update_backend="blocked_qr", **_kwargs()
    )
    eager.load_state_dict(streaming.state_dict())
    torch.testing.assert_close(eager.weights, streaming.weights, rtol=0, atol=0)


def test_explicit_consuming_update_reuses_codes_without_changing_predictor():
    first, first_labels, second, second_labels = _stream()
    ordinary = SquareRootFLYLearner(
        storage_mode="int8", update_backend="blocked_qr",
        update_panel_size=7, update_trailing_chunk_size=5, **_kwargs(),
    )
    consuming = SquareRootFLYLearner(
        storage_mode="int8", update_backend="blocked_qr",
        update_panel_size=7, update_trailing_chunk_size=5, **_kwargs(),
    )
    ordinary.update_codes(first, first_labels)
    consuming.update_codes_consuming(first.clone(), first_labels)
    ordinary_second = second.clone()
    consumed_second = second.clone()
    ordinary.update_codes(ordinary_second, second_labels)
    consuming.update_codes_consuming(consumed_second, second_labels)
    torch.testing.assert_close(ordinary_second, second, rtol=0, atol=0)
    assert not torch.equal(consumed_second, second)
    torch.testing.assert_close(consuming.Q, ordinary.Q, rtol=0, atol=0)
    torch.testing.assert_close(
        consuming.factor.reconstruct_upper(dtype=torch.float64),
        ordinary.factor.reconstruct_upper(dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(consuming.weights, ordinary.weights, rtol=0, atol=0)


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
    stage_memory = learner.diagnostics["last_update_stage_cuda_memory"]
    assert set(stage_memory) == set(timings)
    assert all(
        row["peak_allocated_bytes"] is None for row in stage_memory.values()
    )
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


def test_blocked_backend_trailing_chunk_size_is_checkpoint_locked():
    first, labels, _, _ = _stream()
    learner = SquareRootFLYLearner(
        storage_mode="int8",
        update_backend="blocked_qr",
        update_panel_size=7,
        update_trailing_chunk_size=5,
        **_kwargs(),
    )
    learner.update_codes(first, labels)
    state = learner.state_dict()
    resumed = SquareRootFLYLearner(
        storage_mode="int8",
        update_backend="blocked_qr",
        update_panel_size=7,
        update_trailing_chunk_size=5,
        **_kwargs(),
    )
    resumed.load_state_dict(state)
    with pytest.raises(ValueError, match="trailing chunk size"):
        SquareRootFLYLearner(
            storage_mode="int8",
            update_backend="blocked_qr",
            update_panel_size=7,
            update_trailing_chunk_size=6,
            **_kwargs(),
        ).load_state_dict(state)


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
    assert len(result["results"]) == 7
    assert all(row["peak_cuda_allocated_bytes"] is None for row in result["results"])
    assert result["gates"]["direct_backend_within_tolerance"]
    assert result["gates"]["blocked_qr_backend_within_tolerance"]
    assert result["gates"]["chunked_blocked_qr_backend_within_tolerance"]
    assert result["selected_update_backend"]["name"] == "blocked_qr"
    profiles = {
        row["method"]: row["profiled_task_stage_seconds"]
        for row in result["results"]
    }
    assert profiles["optimized_blocked_qr_srq_int8"] is not None
    assert profiles["optimized_chunked_blocked_qr_srq_int8"] is not None
    assert profiles["exact_fly_dense"] is None


def test_priority2b_system_workers_preserve_probe_state_and_profile(tmp_path):
    config = json.loads(
        (ROOT / "configs/srq_fly_update_optimization_smoke.json").read_text()
    )
    config.update(
        feature_dim=8,
        expand_dim=24,
        synaptic_degree=4,
        block_size=6,
        group_size=4,
        update_panel_size=6,
        quantization_batch_blocks=3,
        rows_per_task=10,
        num_classes=4,
        probe_rows=4,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    results = {}
    probes = {}
    for method in (
        "optimized_eager_quant_blocked_qr_srq_int8",
        "optimized_streaming_quant_blocked_qr_srq_int8",
    ):
        output = tmp_path / f"{method}.json"
        probe = tmp_path / f"{method}.probe.pt"
        results[method] = srq_fly_system_benchmark.run_worker(
            config_path=config_path,
            method=method,
            output=output,
            probe_output=probe,
            device_name="cpu",
            profile_stages=True,
        )
        probes[method] = torch.load(probe, weights_only=True)
    eager = results["optimized_eager_quant_blocked_qr_srq_int8"]
    streaming = results["optimized_streaming_quant_blocked_qr_srq_int8"]
    assert eager["persistent_state_bytes"] == streaming["persistent_state_bytes"]
    assert eager["profiled_task_stage_seconds"] is not None
    assert streaming["profiled_task_stage_seconds"] is not None
    torch.testing.assert_close(
        probes["optimized_streaming_quant_blocked_qr_srq_int8"],
        probes["optimized_eager_quant_blocked_qr_srq_int8"],
        rtol=0,
        atol=0,
    )
