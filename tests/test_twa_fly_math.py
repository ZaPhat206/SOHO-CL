import torch

from methods.twa_fly.solver import solve_one_way, solve_symmetric
from methods.twa_fly.statistics import TWAStatistics


DTYPE = torch.float64


def fixture(seed=19):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(36, 5, generator=generator, dtype=DTYPE)
    z = torch.randn(36, 7, generator=generator, dtype=DTYPE)
    labels = torch.tensor([0, 1, 2] * 12)
    return x, z, labels


def batch_statistics(x, z, labels):
    statistics = TWAStatistics(5, 7, 3, dtype=DTYPE)
    statistics.update(x, z, labels)
    return statistics


def test_streaming_paired_statistics_equal_batch_oracle():
    x, z, labels = fixture()
    streaming = TWAStatistics(5, 7, 3, dtype=DTYPE)
    streaming.update(x[:8], z[:8], labels[:8])
    streaming.update(x[8:21], z[8:21], labels[8:21])
    streaming.update(x[21:], z[21:], labels[21:])
    batch = batch_statistics(x, z, labels)
    for name in ("G_xx", "G_zz", "R_xz", "Q_x", "Q_z", "counts"):
        torch.testing.assert_close(getattr(streaming, name), getattr(batch, name), atol=1e-12, rtol=1e-12)


def test_symmetric_alternating_solver_equals_direct_block_oracle():
    x, z, labels = fixture()
    statistics = batch_statistics(x, z, labels)
    rho, lambda_x, lambda_z = 0.35, 0.4, 0.7
    solution = solve_symmetric(
        statistics, rho, lambda_x, lambda_z, tolerance=1e-12, max_iterations=500
    )
    block = torch.cat((
        torch.cat(((1 + rho) * statistics.G_xx + lambda_x * torch.eye(5, dtype=DTYPE), -rho * statistics.R_xz), dim=1),
        torch.cat((-rho * statistics.R_xz.T, (1 + rho) * statistics.G_zz + lambda_z * torch.eye(7, dtype=DTYPE)), dim=1),
    ), dim=0)
    oracle = torch.linalg.solve(block, torch.cat((statistics.Q_x, statistics.Q_z), dim=0))
    torch.testing.assert_close(
        torch.cat((solution.raw_weights, solution.fly_weights), dim=0), oracle,
        atol=1e-10, rtol=1e-10,
    )
    assert solution.relative_residual <= 1e-12
    assert all(
        later <= earlier + 1e-11
        for earlier, later in zip(solution.objective_history, solution.objective_history[1:])
    )


def test_rho_zero_exactly_recovers_independent_raw_and_fly_ridge():
    x, z, labels = fixture()
    statistics = batch_statistics(x, z, labels)
    solution = solve_symmetric(statistics, 0.0, 0.4, 0.7, tolerance=1e-12)
    expected_x = torch.linalg.solve(statistics.G_xx + 0.4 * torch.eye(5, dtype=DTYPE), statistics.Q_x)
    expected_z = torch.linalg.solve(statistics.G_zz + 0.7 * torch.eye(7, dtype=DTYPE), statistics.Q_z)
    torch.testing.assert_close(solution.raw_weights, expected_x, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(solution.fly_weights, expected_z, atol=1e-12, rtol=1e-12)
    assert solution.relative_residual < 1e-12
    assert solution.iterations == 0


def test_one_way_solver_matches_closed_form_control():
    x, z, labels = fixture()
    statistics = batch_statistics(x, z, labels)
    rho, lambda_x, lambda_z = 0.2, 0.3, 0.6
    solution = solve_one_way(statistics, rho, lambda_x, lambda_z)
    expected_x = torch.linalg.solve(statistics.G_xx + lambda_x * torch.eye(5, dtype=DTYPE), statistics.Q_x)
    expected_z = torch.linalg.solve(
        (1 + rho) * statistics.G_zz + lambda_z * torch.eye(7, dtype=DTYPE),
        statistics.Q_z + rho * statistics.R_xz.T @ expected_x,
    )
    torch.testing.assert_close(solution.raw_weights, expected_x, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(solution.fly_weights, expected_z, atol=1e-12, rtol=1e-12)


def test_destroying_cross_correspondence_changes_symmetric_solution():
    x, z, labels = fixture()
    statistics = batch_statistics(x, z, labels)
    matched = solve_symmetric(statistics, 0.5, 0.4, 0.7, tolerance=1e-12, max_iterations=500)
    permutation = torch.randperm(len(z), generator=torch.Generator().manual_seed(101))
    destroyed_cross = x.T @ z[permutation]
    shuffled = solve_symmetric(
        statistics, 0.5, 0.4, 0.7, tolerance=1e-12, max_iterations=500,
        cross=destroyed_cross,
    )
    assert not torch.allclose(matched.fly_weights, shuffled.fly_weights, atol=1e-8, rtol=1e-8)
