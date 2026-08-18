import pytest
import torch

from methods.cars_fly import adaptive_conditional_directions
from methods.crt_soho.geometry import solve_spd
from methods.crt_soho.solver import (
    reconstruct_residual_statistics,
    solve_block_ridge,
)
from methods.crt_soho.statistics import DualViewStatistics


DTYPE = torch.float64


def synthetic_statistics(seed=73):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(41, 6, generator=generator, dtype=DTYPE)
    phi = torch.randn(41, 9, generator=generator, dtype=DTYPE)
    labels = torch.tensor([2, 7, 11, 13] * 10 + [2])
    statistics = DualViewStatistics(6, 9, dtype=DTYPE)
    statistics.update(x[:7], phi[:7], labels[:7])
    statistics.update(x[7:29], phi[7:29], labels[7:29])
    statistics.update(x[29:], phi[29:], labels[29:])
    return x, phi, labels, statistics


def correction(**overrides):
    _, _, _, statistics = synthetic_statistics()
    arguments = {
        "complement_ridge": 0.3,
        "anchor_ridge": 0.2,
        "residual_ridge": 0.5,
        "energy_threshold": 0.8,
        "max_rank": 4,
        "min_rank": 1,
        "minimum_objective_gain": 0.0,
    }
    arguments.update(overrides)
    return adaptive_conditional_directions(statistics, **arguments)


def test_adaptive_rank_is_smallest_rank_reaching_energy_threshold():
    result = correction(energy_threshold=0.8, max_rank=4)
    energies = result.singular_values.square()
    cumulative = energies.cumsum(0) / energies.sum()
    expected = int(torch.searchsorted(cumulative, torch.tensor(0.8, dtype=DTYPE))) + 1

    assert result.effective_rank == expected
    assert result.threshold_reached is True
    assert result.retained_fraction >= 0.8
    if result.effective_rank > 1:
        assert float(cumulative[result.effective_rank - 2]) < 0.8
    torch.testing.assert_close(
        result.directions.T @ result.directions,
        torch.eye(result.effective_rank, dtype=DTYPE),
        atol=1e-12,
        rtol=1e-12,
    )


def test_rank_budget_reports_when_threshold_cannot_be_reached():
    result = correction(energy_threshold=1.0, max_rank=1)
    assert result.effective_rank == 1
    assert result.threshold_reached is False
    assert result.retained_fraction < 1.0


def test_minimum_gain_can_leave_anchor_unaugmented():
    reference = correction()
    result = correction(minimum_objective_gain=reference.total_energy + 1.0)
    assert result.effective_rank == 0
    assert result.directions.shape == (6, 0)
    assert result.captured_energy == 0.0
    assert result.tail_energy == pytest.approx(result.total_energy)
    assert result.threshold_reached is True


def test_energy_certificate_equals_regularized_objective_reduction():
    _, _, _, statistics = synthetic_statistics()
    result = correction(energy_threshold=0.7, max_rank=4)
    residual = reconstruct_residual_statistics(
        statistics, result.directions, complement_ridge=0.3
    )
    anchor_weights, residual_weights, _ = solve_block_ridge(
        statistics,
        residual,
        anchor_ridge=0.2,
        residual_ridge=0.5,
    )
    baseline = solve_spd(
        statistics.G_pp + 0.2 * torch.eye(9, dtype=DTYPE),
        statistics.Q_p,
    )
    fitted_gain = float(
        (
            statistics.Q_p.T @ anchor_weights
            + residual.Q_r.T @ residual_weights
            - statistics.Q_p.T @ baseline
        )
        .trace()
        .item()
    )

    assert fitted_gain == pytest.approx(result.captured_energy, rel=1e-10, abs=1e-10)
    assert result.total_energy == pytest.approx(
        result.captured_energy + result.tail_energy, rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize(
    "argument,value,message",
    [
        ("energy_threshold", 0.0, "energy_threshold"),
        ("max_rank", 0, "max_rank"),
        ("min_rank", 5, "min_rank"),
        ("minimum_objective_gain", -1.0, "minimum_objective_gain"),
    ],
)
def test_invalid_adaptive_rank_configuration_fails_closed(argument, value, message):
    with pytest.raises(ValueError, match=message):
        correction(**{argument: value})
