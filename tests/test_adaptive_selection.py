import math

import pytest
import torch

from methods.analytic_ridge import AdaptiveCompressedUpper, SquareRootBackend
from methods.analytic_ridge.adaptive_selection import (
    CriterionAdaptiveSquareRootBackend,
    batched_factor_mse_benefits,
    budget_bytes,
    compress_upper_with_criterion,
    factor_row_weights,
    greedy_select,
    score_blocks,
    upper_block_descriptors,
)


def _random_upper(dimension: int, seed: int, dtype=torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(dimension, dimension, generator=generator, dtype=dtype)
    # Heterogeneous row scales, as in a Cholesky factor of a sum of Gram terms.
    scales = torch.logspace(2, -1, dimension, dtype=dtype)
    matrix = torch.triu(matrix * scales[:, None])
    matrix.diagonal().copy_(matrix.diagonal().abs() + 1.0)
    return matrix


CASES = [
    (1, 4, 3),
    (5, 4, 3),
    (64, 16, 8),
    (130, 32, 16),
    (257, 64, 16),
]


@pytest.mark.parametrize("dimension,block_size,group_size", CASES)
@pytest.mark.parametrize("budget", [0.0, 0.1, 0.25, 1.0])
def test_factor_mse_twin_matches_locked_compressor_exactly(
    dimension, block_size, group_size, budget
):
    source = _random_upper(dimension, seed=dimension + int(budget * 100))
    locked_matrix = source.clone()
    twin_matrix = source.clone()
    locked, locked_error, locked_diag = AdaptiveCompressedUpper.from_upper_inplace(
        locked_matrix, block_size=block_size, group_size=group_size, budget_fraction=budget
    )
    twin, twin_error, twin_diag = compress_upper_with_criterion(
        twin_matrix,
        block_size=block_size,
        group_size=group_size,
        budget_fraction=budget,
        criterion="factor_mse",
    )
    assert torch.equal(locked_matrix, twin_matrix)
    assert torch.equal(locked.precision_mask, twin.precision_mask)
    assert torch.equal(locked.diagonal, twin.diagonal)
    assert len(locked.blocks) == len(twin.blocks)
    for left, right in zip(locked.blocks, twin.blocks):
        assert (left.row_block, left.col_block) == (right.row_block, right.col_block)
        assert torch.equal(left.values, right.values)
        assert (left.scales is None) == (right.scales is None)
        if left.scales is not None:
            assert torch.equal(left.scales, right.scales)
    assert locked_error == twin_error
    assert twin_diag.pop("selection_criterion") == "factor_mse"
    assert locked_diag == twin_diag


def test_greedy_skips_unaffordable_block_and_continues():
    selected, used = greedy_select([9.0, 10.0, 2.0], [3, 5, 2], extra_budget=6)
    assert selected == [True, False, True]
    assert used == 5


def test_greedy_ignores_non_positive_benefit_and_cost():
    selected, used = greedy_select([0.0, -1.0, 4.0, 5.0], [2, 2, 0, 2], extra_budget=100)
    assert selected == [False, False, False, True]
    assert used == 2


def test_budget_bytes_follow_locked_rule():
    base, full, extra = budget_bytes([65536, 32640], group_size=64, budget_fraction=0.25)
    assert base == 65536 + 4 * 1024 + 32640 + 4 * math.ceil(32640 / 64)
    assert full == 2 * (65536 + 32640)
    assert extra == math.floor(0.25 * (full - base))


def test_row_weighted_scores_equal_bruteforce_expectation():
    dimension, block_size, group_size = 37, 8, 5
    matrix = _random_upper(dimension, seed=11, dtype=torch.float64)
    weights = factor_row_weights(matrix)
    scores = score_blocks(
        matrix,
        block_size=block_size,
        group_size=group_size,
        criterion="row_weighted_system",
        row_weights=weights,
    )
    for index, (row_block, col_block) in enumerate(scores["descriptors"]):
        rs, re = row_block * block_size, min((row_block + 1) * block_size, dimension)
        cs, ce = col_block * block_size, min((col_block + 1) * block_size, dimension)
        local = matrix[rs:re, cs:ce]
        if row_block == col_block:
            rows, columns = torch.triu_indices(re - rs, ce - cs, offset=1)
        else:
            rows, columns = torch.meshgrid(
                torch.arange(re - rs), torch.arange(ce - cs), indexing="ij"
            )
            rows, columns = rows.reshape(-1), columns.reshape(-1)
        values = local[rows, columns]
        _, _, decoded_int8 = __import__(
            "methods.analytic_ridge.adaptive_upper", fromlist=["_groupwise_int8"]
        )._groupwise_int8(values, group_size)
        decoded_fp16 = values.to(torch.float16).to(values.dtype)
        expected = (
            weights[rows + rs]
            * ((decoded_int8 - values).square() - (decoded_fp16 - values).square())
        ).sum()
        assert scores["benefits"][index] == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


def test_row_weights_are_chunk_invariant_squared_row_norms():
    matrix = _random_upper(50, seed=5)
    expected = torch.linalg.vector_norm(matrix.to(torch.float64), dim=1).square()
    for chunk in (1, 7, 64):
        assert torch.allclose(factor_row_weights(matrix, chunk_rows=chunk), expected, rtol=1e-12)


@pytest.mark.parametrize("criterion", ["factor_mse", "row_weighted_system"])
def test_selection_respects_byte_ceiling_and_is_monotone_in_budget(criterion):
    dimension, block_size, group_size = 130, 32, 16
    source = _random_upper(dimension, seed=3)
    previous_blocks, previous_bytes = -1, -1
    for budget in (0.0, 0.05, 0.25, 0.5, 1.0):
        matrix = source.clone()
        state, _, diagnostics = compress_upper_with_criterion(
            matrix,
            block_size=block_size,
            group_size=group_size,
            budget_fraction=budget,
            criterion=criterion,
            row_weights=factor_row_weights(source) if criterion == "row_weighted_system" else None,
        )
        assert diagnostics["factor_persistent_bytes"] <= diagnostics["factor_budget_ceiling_bytes"]
        assert diagnostics["used_extra_bytes"] <= diagnostics["extra_budget_bytes"]
        if budget == 0.0:
            assert diagnostics["selected_fp16_blocks"] == 0
        assert diagnostics["selected_fp16_blocks"] >= previous_blocks
        assert diagnostics["factor_persistent_bytes"] >= previous_bytes
        previous_blocks = diagnostics["selected_fp16_blocks"]
        previous_bytes = diagnostics["factor_persistent_bytes"]
        assert bool((state.diagonal > 0).all())


def test_row_weighted_criterion_can_differ_from_factor_mse():
    # Rows with small entries but huge row norm: the weighted criterion must
    # prefer them over rows with larger entries but tiny row norm.
    dimension, block_size, group_size = 16, 4, 4
    matrix = torch.zeros(dimension, dimension)
    matrix.diagonal().fill_(1.0)
    matrix[0, 1:] = torch.linspace(0.01, 0.02, dimension - 1)
    matrix[0, 0] = 1.0e4
    matrix[12, 13:] = torch.tensor([5.0, -3.0, 4.0])
    weights = factor_row_weights(matrix)
    common = dict(block_size=block_size, group_size=group_size)
    mse = score_blocks(matrix, criterion="factor_mse", **common)
    weighted = score_blocks(
        matrix, criterion="row_weighted_system", row_weights=weights, **common
    )
    top_mse = max(range(len(mse["benefits"])), key=mse["benefits"].__getitem__)
    top_weighted = max(range(len(weighted["benefits"])), key=weighted["benefits"].__getitem__)
    assert mse["descriptors"][top_mse][0] == 3
    assert weighted["descriptors"][top_weighted][0] == 0


@pytest.mark.parametrize("dimension,block_size,group_size", CASES)
def test_batched_benefits_match_loop_and_give_the_same_decisions(
    dimension, block_size, group_size
):
    matrix = _random_upper(dimension, seed=dimension + 17)
    loop = score_blocks(
        matrix, block_size=block_size, group_size=group_size, criterion="factor_mse"
    )
    batched, costs, counts = batched_factor_mse_benefits(
        matrix, block_size=block_size, group_size=group_size, batch_blocks=3
    )
    assert costs == loop["extra_costs"]
    assert counts == loop["value_counts"]
    assert len(batched) == len(loop["benefits"])
    for left, right in zip(batched, loop["benefits"]):
        assert left == pytest.approx(right, rel=1e-5, abs=1e-6)
    for budget in (0.05, 0.25, 0.5):
        _, _, extra = budget_bytes(counts, group_size=group_size, budget_fraction=budget)
        assert greedy_select(batched, costs, extra) == greedy_select(
            loop["benefits"], loop["extra_costs"], extra
        )


def test_descriptors_follow_locked_storage_order():
    assert upper_block_descriptors(10, 4) == [
        (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)
    ]


def _stream(dimension: int, tasks: int, rows: int, classes_per_task: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    for task in range(tasks):
        features = torch.relu(torch.randn(rows, dimension, generator=generator))
        labels = torch.randint(
            task * classes_per_task,
            (task + 1) * classes_per_task,
            (rows,),
            generator=generator,
        )
        yield features, labels


def _common(dimension: int) -> dict:
    return dict(
        dimension=dimension,
        ridge_lambda=10.0,
        device=torch.device("cpu"),
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        block_size=32,
        group_size=16,
        update_panel_size=16,
        update_trailing_chunk_size=None,
        first_update_backend="gram_cholesky",
        quantization_backend="streaming",
        quantization_batch_blocks=4,
    )


def test_factor_mse_backend_reproduces_locked_backend_bitwise():
    dimension = 96
    locked = SquareRootBackend(
        storage_mode="adaptive_int8_fp16", adaptive_budget_fraction=0.25, **_common(dimension)
    )
    twin = CriterionAdaptiveSquareRootBackend(
        selection_criterion="factor_mse", adaptive_budget_fraction=0.25, **_common(dimension)
    )
    seen = []
    twin._pre_compression_hook = lambda factor: seen.append(factor.clone())
    for features, labels in _stream(dimension, tasks=3, rows=120, classes_per_task=3, seed=7):
        locked.update(features, labels)
        twin.update(features, labels)
        assert torch.equal(locked.weights, twin.weights)
        assert torch.equal(locked.factor.precision_mask, twin.factor.precision_mask)
        locked_tensors = locked.persistent_tensors()
        twin_tensors = twin.persistent_tensors()
        assert locked_tensors.keys() == twin_tensors.keys()
        for name in locked_tensors:
            assert torch.equal(locked_tensors[name], twin_tensors[name]), name
    assert len(seen) == 3


def test_row_weighted_backend_is_structurally_valid():
    dimension = 96
    backend = CriterionAdaptiveSquareRootBackend(
        selection_criterion="row_weighted_system",
        adaptive_budget_fraction=0.1,
        **_common(dimension),
    )
    for features, labels in _stream(dimension, tasks=3, rows=120, classes_per_task=3, seed=9):
        backend.update(features, labels)
        diagnostics = backend.diagnostics
        assert diagnostics["selection_criterion"] == "row_weighted_system"
        assert diagnostics["factor_persistent_bytes"] <= diagnostics["factor_budget_ceiling_bytes"]
        assert diagnostics["solver_relative_residual"] < 1e-4
    reconstructed = backend.factor.reconstruct_upper()
    assert bool((reconstructed.diagonal() > 0).all())


def test_criterion_backend_rejects_unknown_criterion_and_storage_override():
    with pytest.raises(ValueError):
        CriterionAdaptiveSquareRootBackend(
            selection_criterion="accuracy", adaptive_budget_fraction=0.25, **_common(32)
        )
    with pytest.raises(ValueError):
        CriterionAdaptiveSquareRootBackend(
            selection_criterion="factor_mse",
            adaptive_budget_fraction=0.25,
            storage_mode="int8",
            **_common(32),
        )


def test_row_weighted_scoring_requires_valid_weights():
    matrix = _random_upper(20, seed=1)
    with pytest.raises(ValueError):
        score_blocks(matrix, block_size=8, group_size=4, criterion="row_weighted_system")
    with pytest.raises(ValueError):
        score_blocks(
            matrix,
            block_size=8,
            group_size=4,
            criterion="factor_mse",
            row_weights=factor_row_weights(matrix),
        )
