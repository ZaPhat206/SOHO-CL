from __future__ import annotations

import torch

from methods.analytic_ridge.backends import ExactGramBackend
from methods.analytic_ridge.equal_memory_controls import (
    FrequentDirectionsRidgeBackend,
    PackedExactGramBackend,
    frequent_directions_backend_bytes,
    largest_frequent_directions_rank,
    largest_packed_exact_dimension,
    packed_exact_backend_bytes,
    packed_upper_element_count,
)


def _update_two_tasks(backend) -> None:
    generator = torch.Generator().manual_seed(2025)
    first = torch.randn((9, backend.dimension), generator=generator)
    second = torch.randn((7, backend.dimension), generator=generator)
    backend.update(first, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]))
    backend.update(second, torch.tensor([1, 2, 3, 1, 2, 3, 1]))


def test_packed_upper_accounting_and_budget_are_exact():
    assert packed_upper_element_count(5) == 15
    assert packed_exact_backend_bytes(5, 3) == 4 * 15 + 8 * 5 * 3 + 4 * 3
    dimension, used = largest_packed_exact_dimension(
        target_total_bytes=10_000,
        feature_dimension=7,
        classes=3,
        maximum_dimension=100,
    )
    assert 4 * 7 * dimension + packed_exact_backend_bytes(dimension, 3) == used
    assert used <= 10_000
    assert 4 * 7 * (dimension + 1) + packed_exact_backend_bytes(
        dimension + 1, 3
    ) > 10_000


def test_packed_exact_matches_dense_exact_and_roundtrips():
    common = dict(
        dimension=11,
        ridge_lambda=3.0,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        device="cpu",
    )
    dense = ExactGramBackend(**common)
    packed = PackedExactGramBackend(block_size=4, **common)
    _update_two_tasks(dense)
    _update_two_tasks(packed)
    probe = torch.randn((13, 11), generator=torch.Generator().manual_seed(99))
    torch.testing.assert_close(
        packed.predict_logits(probe), dense.predict_logits(probe), rtol=2e-5, atol=2e-5
    )
    expected = packed_exact_backend_bytes(11, 4)
    assert packed.persistent_state_bytes() == expected
    restored = PackedExactGramBackend(block_size=4, **common)
    restored.load_state_dict(packed.state_dict())
    torch.testing.assert_close(
        restored.predict_logits(probe), packed.predict_logits(probe), rtol=0, atol=0
    )
    packed.assert_exemplar_free_state()


def test_frequent_directions_matches_exact_when_rank_covers_rows():
    common = dict(
        dimension=8,
        ridge_lambda=2.0,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
        device="cpu",
    )
    exact = ExactGramBackend(**common)
    fd = FrequentDirectionsRidgeBackend(sketch_rank=8, **common)
    generator = torch.Generator().manual_seed(7)
    values = torch.randn((6, 8), generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    exact.update(values, labels)
    fd.update(values, labels)
    probe = torch.randn((5, 8), generator=generator, dtype=torch.float64)
    torch.testing.assert_close(
        fd.predict_logits(probe), exact.predict_logits(probe), rtol=1e-10, atol=1e-10
    )
    assert fd.diagnostics["last_shrinkage_delta"] == 0.0


def test_frequent_directions_shrink_has_standard_psd_error():
    common = dict(
        dimension=7,
        ridge_lambda=1.0,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
        device="cpu",
    )
    fd = FrequentDirectionsRidgeBackend(sketch_rank=3, **common)
    generator = torch.Generator().manual_seed(11)
    values = torch.randn((12, 7), generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2] * 4)
    fd.update(values, labels)
    covariance_error = values.T @ values - fd.sketch.T @ fd.sketch
    eigenvalues = torch.linalg.eigvalsh((covariance_error + covariance_error.T) * 0.5)
    assert float(eigenvalues.min()) >= -1e-9
    assert float(eigenvalues.max()) <= fd.covariance_error_bound + 1e-9
    assert fd.sketch.shape == (3, 7)
    assert fd.diagnostics["solver_relative_residual"] < 1e-10
    fd.assert_exemplar_free_state()


def test_frequent_directions_compact_fp64_correction_avoids_wide_cancellation():
    fd = FrequentDirectionsRidgeBackend(
        dimension=31,
        sketch_rank=7,
        ridge_lambda=1_000_000.0,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        device="cpu",
    )
    generator = torch.Generator().manual_seed(17)
    values = torch.randn((64, 31), generator=generator) * 10_000.0
    labels = torch.tensor([0, 1, 2, 3] * 16)
    fd.update(values, labels)
    assert fd.weights is None
    assert fd.correction.dtype == torch.float64
    assert fd.diagnostics["solver_relative_residual"] < 1e-10
    explicit = (
        fd.Q.to(torch.float64) / fd.ridge_lambda
        - fd.sketch.to(torch.float64).T @ fd.correction
    )
    probe = torch.randn((5, 31), generator=generator)
    torch.testing.assert_close(fd.predict_logits(probe), probe.to(torch.float64) @ explicit)


def test_frequent_directions_accounting_rank_lock_and_roundtrip():
    target = 50_000
    rank, used = largest_frequent_directions_rank(
        target_total_bytes=target,
        feature_dimension=7,
        expanded_dimension=31,
        classes=5,
    )
    expected = 4 * 7 * 31 + frequent_directions_backend_bytes(31, 5, rank)
    assert used == expected <= target
    if rank < 31:
        assert 4 * 7 * 31 + frequent_directions_backend_bytes(31, 5, rank + 1) > target

    common = dict(
        dimension=7,
        ridge_lambda=1.5,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        device="cpu",
    )
    fd = FrequentDirectionsRidgeBackend(sketch_rank=3, **common)
    _update_two_tasks(fd)
    assert fd.persistent_state_bytes() == frequent_directions_backend_bytes(7, 4, 3)
    restored = FrequentDirectionsRidgeBackend(sketch_rank=3, **common)
    restored.load_state_dict(fd.state_dict())
    probe = torch.randn((4, 7), generator=torch.Generator().manual_seed(51))
    torch.testing.assert_close(
        restored.predict_logits(probe), fd.predict_logits(probe), rtol=0, atol=0
    )
