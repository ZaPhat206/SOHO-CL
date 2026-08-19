import torch

from methods.tail_fly import (
    StreamingTruncatedSVD,
    approximate_gram,
    diagonal_tail,
    solve_diagonal_ridge,
    solve_tail_ridge,
    solve_truncated_svd_ridge,
)


DTYPE = torch.float64


def _wta_like(rows: int, dimension: int, active: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    dense = torch.randn(rows, dimension, generator=generator, dtype=DTYPE)
    values, indices = dense.topk(active, dim=1)
    result = torch.zeros_like(dense)
    result.scatter_(1, indices, values)
    return result


def test_full_rank_streaming_svd_reconstructs_batch_gram():
    rows = _wta_like(17, 23, 6, 7)
    sketch = StreamingTruncatedSVD(23, 23, dtype=DTYPE)
    for part in rows.split([3, 5, 9]):
        sketch.update(part)
    reconstructed = (sketch.U * sketch.s) @ (sketch.U * sketch.s).T
    torch.testing.assert_close(reconstructed, rows.T @ rows, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        sketch.U.T @ sketch.U,
        torch.eye(sketch.effective_rank, dtype=DTYPE),
        atol=1e-10,
        rtol=1e-10,
    )


def test_truncated_stream_has_exact_nonnegative_diagonal_tail():
    rows = _wta_like(31, 19, 5, 11)
    sketch = StreamingTruncatedSVD(19, 6, dtype=DTYPE)
    for part in rows.split([7, 8, 16]):
        sketch.update(part)
    exact_diagonal = rows.square().sum(0)
    tail = diagonal_tail(exact_diagonal, sketch.U, sketch.s)
    approximation = approximate_gram(sketch.U, sketch.s, tail)
    assert float(tail.min()) >= 0
    torch.testing.assert_close(
        torch.diagonal(approximation), exact_diagonal, atol=1e-10, rtol=1e-10
    )


def test_woodbury_solution_matches_direct_solve():
    rows = _wta_like(29, 17, 4, 13)
    Q = torch.randn(17, 5, generator=torch.Generator().manual_seed(14), dtype=DTYPE)
    sketch = StreamingTruncatedSVD(17, 7, dtype=DTYPE)
    sketch.update(rows[:12])
    sketch.update(rows[12:])
    tail = diagonal_tail(rows.square().sum(0), sketch.U, sketch.s)
    ridge = 0.37
    solution = solve_tail_ridge(sketch.U, sketch.s, tail, Q, ridge)
    system = approximate_gram(sketch.U, sketch.s, tail) + ridge * torch.eye(17, dtype=DTYPE)
    oracle = torch.linalg.solve(system, Q)
    torch.testing.assert_close(solution.weights, oracle, atol=1e-10, rtol=1e-10)
    assert solution.relative_residual < 1e-10


def test_float32_statistics_can_be_solved_consistently_in_float64():
    rows = _wta_like(37, 29, 7, 101).to(torch.float32)
    Q = torch.randn(29, 6, generator=torch.Generator().manual_seed(102))
    sketch = StreamingTruncatedSVD(29, 9, dtype=torch.float32)
    sketch.update(rows[:13])
    sketch.update(rows[13:])
    tail = diagonal_tail(rows.square().sum(0), sketch.U, sketch.s)
    ridge = 0.13
    solution = solve_tail_ridge(
        sketch.U, sketch.s, tail, Q, ridge, solve_dtype=torch.float64
    )
    system = approximate_gram(
        sketch.U.double(), sketch.s.double(), tail.double()
    ) + ridge * torch.eye(29, dtype=torch.float64)
    oracle = torch.linalg.solve(system, Q.double())
    assert solution.weights.dtype == torch.float64
    torch.testing.assert_close(solution.weights, oracle, atol=1e-10, rtol=1e-10)
    assert solution.relative_residual < 1e-10


def test_full_rank_tail_logits_equal_exact_fly_ridge():
    train = _wta_like(14, 21, 5, 17)
    test = _wta_like(8, 21, 5, 18)
    targets = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]),
        num_classes=4,
    ).to(DTYPE)
    Q = train.T @ targets
    sketch = StreamingTruncatedSVD(21, 21, dtype=DTYPE)
    sketch.update(train[:6])
    sketch.update(train[6:])
    tail = diagonal_tail(train.square().sum(0), sketch.U, sketch.s)
    tail_weights = solve_tail_ridge(sketch.U, sketch.s, tail, Q, 0.2).weights
    exact_weights = torch.linalg.solve(
        train.T @ train + 0.2 * torch.eye(21, dtype=DTYPE), Q
    )
    torch.testing.assert_close(
        test @ tail_weights, test @ exact_weights, atol=1e-8, rtol=1e-8
    )


def test_rank_zero_is_diagonal_only_ridge():
    exact_diagonal = torch.tensor([2.0, 5.0, 0.0, 3.0], dtype=DTYPE)
    Q = torch.arange(12, dtype=DTYPE).reshape(4, 3)
    empty_U, empty_s = torch.empty(4, 0, dtype=DTYPE), torch.empty(0, dtype=DTYPE)
    tail = solve_tail_ridge(empty_U, empty_s, exact_diagonal, Q, 0.5)
    diagonal = solve_diagonal_ridge(exact_diagonal, Q, 0.5)
    torch.testing.assert_close(tail.weights, diagonal.weights)
    torch.testing.assert_close(
        tail.weights, Q / (exact_diagonal + 0.5).unsqueeze(1)
    )


def test_plain_tsvd_control_drops_orthogonal_label_signal():
    U = torch.eye(5, dtype=DTYPE)[:, :2]
    s = torch.tensor([3.0, 2.0], dtype=DTYPE)
    Q = torch.eye(5, dtype=DTYPE)
    solution = solve_truncated_svd_ridge(U, s, Q, 0.1)
    assert solution.active_rank == 2
    torch.testing.assert_close(solution.weights[2:], torch.zeros(3, 5, dtype=DTYPE))


def test_ridge_perturbation_bound_holds_on_synthetic_psd_system():
    rows = _wta_like(24, 13, 4, 23)
    Q = torch.randn(13, 3, generator=torch.Generator().manual_seed(24), dtype=DTYPE)
    sketch = StreamingTruncatedSVD(13, 5, dtype=DTYPE)
    sketch.update(rows)
    exact = rows.T @ rows
    tail = diagonal_tail(torch.diagonal(exact), sketch.U, sketch.s)
    approximate = approximate_gram(sketch.U, sketch.s, tail)
    ridge = 0.7
    exact_weights = torch.linalg.solve(exact + ridge * torch.eye(13, dtype=DTYPE), Q)
    approximate_weights = torch.linalg.solve(
        approximate + ridge * torch.eye(13, dtype=DTYPE), Q
    )
    bound = (
        torch.linalg.matrix_norm(exact - approximate, ord=2)
        * torch.linalg.matrix_norm(Q)
        / ridge**2
    )
    assert torch.linalg.matrix_norm(exact_weights - approximate_weights) <= bound + 1e-12
