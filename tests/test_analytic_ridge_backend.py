"""M1 gates for representation-agnostic analytic Ridge backends."""

from __future__ import annotations

import inspect

import pytest
import torch

from methods.analytic_ridge import ExactGramBackend, SquareRootBackend
from methods.analytic_ridge import backends as generic_backends
from methods.analytic_ridge import qr as generic_qr
from methods.frontends import FLYAnalyticLearner
from methods.srq_fly_optimized import SquareRootFLYLearner


def _backend_kwargs(**overrides):
    values = dict(
        dimension=24,
        ridge_lambda=100.0,
        block_size=6,
        group_size=5,
        storage_mode="int8",
        update_panel_size=7,
        first_update_backend="gram_cholesky",
        quantization_backend="streaming",
        quantization_batch_blocks=3,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    values.update(overrides)
    return values


def _legacy_kwargs(**overrides):
    values = dict(
        feature_dim=7,
        expand_dim=24,
        synaptic_degree=4,
        coding_level=0.25,
        ridge_lambda=100.0,
        block_size=6,
        group_size=5,
        seed=2025,
        storage_mode="int8",
        update_backend="blocked_qr",
        update_panel_size=7,
        first_update_backend="gram_cholesky",
        quantization_backend="streaming",
        quantization_batch_blocks=3,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    values.update(overrides)
    return values


def _stream():
    generator = torch.Generator().manual_seed(901)
    return (
        (
            torch.randn(17, 24, generator=generator, dtype=torch.float64),
            torch.tensor([9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2]),
        ),
        (
            torch.randn(13, 24, generator=generator, dtype=torch.float64),
            torch.tensor([11, 5, 2, 11, 5, 2, 11, 5, 2, 11, 5, 2, 11]),
        ),
    )


def _assert_same_compressed_state(left, right):
    assert left.dimension == right.dimension
    assert left.mode == right.mode
    assert torch.equal(left.diagonal, right.diagonal)
    assert len(left.blocks) == len(right.blocks)
    for left_block, right_block in zip(left.blocks, right.blocks):
        assert left_block.row_block == right_block.row_block
        assert left_block.col_block == right_block.col_block
        assert torch.equal(left_block.values, right_block.values)
        if left.mode == "int8":
            assert torch.equal(left_block.scales, right_block.scales)


def test_generic_backend_has_no_fly_representation_dependency():
    sources = inspect.getsource(generic_backends) + inspect.getsource(generic_qr)
    assert "FlyHash" not in sources
    assert "coding_level" not in sources
    assert "synaptic_degree" not in sources


@pytest.mark.parametrize("storage_mode", ["float16", "int8"])
def test_generic_square_root_is_byte_identical_to_legacy_p2b(storage_mode):
    legacy = SquareRootFLYLearner(
        **_legacy_kwargs(storage_mode=storage_mode)
    )
    generic = SquareRootBackend(
        **_backend_kwargs(storage_mode=storage_mode)
    )
    for features, labels in _stream():
        legacy.update_codes(features, labels)
        generic.update(features, labels)

    _assert_same_compressed_state(generic.factor, legacy.factor)
    torch.testing.assert_close(generic.Q, legacy.Q, rtol=0, atol=0)
    torch.testing.assert_close(generic.counts, legacy.counts, rtol=0, atol=0)
    torch.testing.assert_close(generic.weights, legacy.weights, rtol=0, atol=0)
    assert generic.class_ids == legacy.class_ids
    projection_bytes = legacy.persistent_state_bytes() - sum(
        tensor.numel() * tensor.element_size()
        if tensor.layout == torch.strided
        else (
            tensor.values().numel() * tensor.values().element_size()
            + tensor.ccol_indices().numel() * tensor.ccol_indices().element_size()
            + tensor.row_indices().numel() * tensor.row_indices().element_size()
        )
        for name, tensor in legacy.persistent_tensors().items()
        if name == "projection"
    )
    assert generic.persistent_state_bytes() == projection_bytes


def test_generic_checkpoint_round_trip_and_legacy_import():
    original = SquareRootBackend(**_backend_kwargs())
    for features, labels in _stream():
        original.update(features, labels)
    restored = SquareRootBackend(**_backend_kwargs())
    restored.load_state_dict(original.state_dict())
    _assert_same_compressed_state(restored.factor, original.factor)
    torch.testing.assert_close(restored.weights, original.weights, rtol=0, atol=0)

    legacy = SquareRootFLYLearner(**_legacy_kwargs())
    for features, labels in _stream():
        legacy.update_codes(features, labels)
    imported = SquareRootBackend(**_backend_kwargs())
    imported.load_legacy_srq_fly_state_dict(legacy.state_dict())
    _assert_same_compressed_state(imported.factor, legacy.factor)
    torch.testing.assert_close(imported.weights, legacy.weights, rtol=0, atol=0)


def test_fly_frontend_reproduces_legacy_representation_and_state_bytes():
    legacy = SquareRootFLYLearner(**_legacy_kwargs())
    backend = SquareRootBackend(**_backend_kwargs())
    generic = FLYAnalyticLearner(
        feature_dim=7,
        synaptic_degree=4,
        coding_level=0.25,
        backend=backend,
        seed=2025,
        projection=legacy.flyhash.projection_matrix,
    )
    generator = torch.Generator().manual_seed(113)
    features = torch.randn(19, 7, generator=generator)
    labels = torch.tensor([2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2])
    torch.testing.assert_close(generic.encode(features), legacy.encode(features), rtol=0, atol=0)
    legacy.update(features, labels)
    generic.update(features, labels)
    _assert_same_compressed_state(generic.backend.factor, legacy.factor)
    torch.testing.assert_close(generic.weights, legacy.weights, rtol=0, atol=0)
    assert generic.persistent_state_bytes() == legacy.persistent_state_bytes()
    generic.assert_exemplar_free_state()


def test_fly_frontend_imports_legacy_checkpoint_without_prediction_drift():
    legacy = SquareRootFLYLearner(**_legacy_kwargs())
    generator = torch.Generator().manual_seed(114)
    features = torch.randn(19, 7, generator=generator)
    labels = torch.tensor([2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2])
    legacy.update(features, labels)
    generic = FLYAnalyticLearner(
        feature_dim=7,
        synaptic_degree=4,
        coding_level=0.25,
        backend=SquareRootBackend(**_backend_kwargs()),
        seed=2025,
        projection=legacy.flyhash.projection_matrix,
    )
    generic.load_legacy_srq_fly_state_dict(legacy.state_dict())
    probe = torch.randn(11, 7, generator=generator)
    torch.testing.assert_close(
        generic.predict_logits(probe), legacy.predict_logits(probe), rtol=0, atol=0
    )
    assert torch.equal(generic.predict(probe), legacy.predict(probe))


def test_fly_frontend_checkpoint_round_trip():
    legacy = SquareRootFLYLearner(**_legacy_kwargs())
    first = FLYAnalyticLearner(
        feature_dim=7,
        synaptic_degree=4,
        coding_level=0.25,
        backend=SquareRootBackend(**_backend_kwargs()),
        seed=2025,
        projection=legacy.flyhash.projection_matrix,
    )
    generator = torch.Generator().manual_seed(115)
    features = torch.randn(19, 7, generator=generator)
    labels = torch.tensor([2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2])
    first.update(features, labels)
    restored = FLYAnalyticLearner(
        feature_dim=7,
        synaptic_degree=4,
        coding_level=0.25,
        backend=SquareRootBackend(**_backend_kwargs()),
        seed=2025,
        projection=legacy.flyhash.projection_matrix,
    )
    restored.load_state_dict(first.state_dict())
    probe = torch.randn(11, 7, generator=generator)
    torch.testing.assert_close(
        restored.predict_logits(probe), first.predict_logits(probe), rtol=0, atol=0
    )
    assert restored.persistent_state_bytes() == first.persistent_state_bytes()


def test_checkpoint_rejects_quantization_policy_mismatch():
    backend = SquareRootBackend(**_backend_kwargs())
    features, labels = _stream()[0]
    backend.update(features, labels)
    state = backend.state_dict()
    state["quantization_batch_blocks"] = 4
    with pytest.raises(ValueError, match="quantization batch size"):
        SquareRootBackend(**_backend_kwargs()).load_state_dict(state)


def test_exact_backend_matches_joint_ridge_reference():
    backend = ExactGramBackend(
        dimension=24,
        ridge_lambda=100.0,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    feature_parts = []
    label_parts = []
    for features, labels in _stream():
        backend.update(features, labels)
        feature_parts.append(features)
        label_parts.append(labels)
    features = torch.cat(feature_parts)
    labels = torch.cat(label_parts)
    class_ids = sorted(set(labels.tolist()))
    columns = torch.tensor([class_ids.index(int(value)) for value in labels])
    targets = torch.nn.functional.one_hot(columns, num_classes=len(class_ids)).to(
        torch.float64
    )
    system = features.T @ features
    system.diagonal().add_(100.0)
    expected = torch.linalg.solve(system, features.T @ targets)
    torch.testing.assert_close(backend.weights, expected, rtol=2e-13, atol=2e-13)
    assert backend.class_ids == class_ids
    restored = ExactGramBackend(
        dimension=24,
        ridge_lambda=100.0,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    restored.load_state_dict(backend.state_dict())
    torch.testing.assert_close(restored.weights, backend.weights, rtol=0, atol=0)
    assert restored.persistent_state_bytes() == backend.persistent_state_bytes()


def test_exact_checkpoint_accepts_roundoff_skew_but_rejects_asymmetry():
    backend = ExactGramBackend(
        dimension=24,
        ridge_lambda=100.0,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    features, labels = _stream()[0]
    backend.update(features, labels)

    roundoff_state = backend.state_dict()
    roundoff_state["gram"][0, 1] += torch.finfo(torch.float64).eps
    restored = ExactGramBackend(
        dimension=24,
        ridge_lambda=100.0,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    restored.load_state_dict(roundoff_state)
    assert restored.weights is not None

    asymmetric_state = backend.state_dict()
    asymmetric_state["gram"][0, 1] += 1.0
    with pytest.raises(ValueError, match="numerically symmetric"):
        ExactGramBackend(
            dimension=24,
            ridge_lambda=100.0,
            statistics_dtype=torch.float64,
            solver_dtype=torch.float64,
        ).load_state_dict(asymmetric_state)
