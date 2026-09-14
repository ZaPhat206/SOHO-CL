"""Validate the M17 conditioning estimators against dense ground truth.

The point of these tests is to make a GPU run worth starting: if the power
iteration does not reproduce a densely computed operator norm on a small
problem, the large run is wasted. Every estimator M17 reports is checked here
against an exact dense computation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tools import srq_generalization_m17 as m17


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / (
    "srq_generalization_m17_conditioning_audit.json"
)


def _spd(dimension: int, seed: int, ridge: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(dimension * 2, dimension, generator=generator, dtype=torch.float64)
    system = basis.T @ basis
    system.diagonal().add_(ridge)
    return (system + system.T) * 0.5


# ---------------------------------------------------------------------------
# configuration contract
# ---------------------------------------------------------------------------
def test_config_parses_and_is_train_only():
    config = m17._read_config(CONFIG_PATH)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["prediction_free"] is True
    assert config["seed"] == 2025
    assert config["diagnostic_widths"] == [10000, 20000]


def test_config_rejects_unexpected_key():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected keys"):
        m17._validate_config(config)


def test_config_rejects_missing_key():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    del config["conditioning"]
    with pytest.raises(ValueError, match="missing keys"):
        m17._validate_config(config)


def test_config_rejects_test_use():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["uses_test_set"] = True
    with pytest.raises(ValueError, match="train-only"):
        m17._validate_config(config)


def test_config_rejects_non_protocol_seed():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["seed"] = 1234
    with pytest.raises(ValueError, match="seed 2025"):
        m17._validate_config(config)


# ---------------------------------------------------------------------------
# estimator correctness against dense ground truth
# ---------------------------------------------------------------------------
def test_power_iteration_matches_dense_largest_eigenvalue():
    system = _spd(48, seed=11, ridge=0.5)
    exact = float(torch.linalg.eigvalsh(system).max())
    value, converged, _, _ = m17._power_iteration(
        lambda v: system @ v, 48, torch.device("cpu"), torch.float64,
        seed=3, max_steps=5000, tolerance=1e-13, restarts=2,
    )
    assert converged
    assert value == pytest.approx(exact, rel=1e-7)


def test_inverse_power_iteration_matches_dense_smallest_eigenvalue():
    system = _spd(48, seed=12, ridge=0.75)
    factor = m17._cholesky(system)
    exact = float(torch.linalg.eigvalsh(system).min())
    inverse_max, converged, _, _ = m17._power_iteration(
        lambda v: m17._solve(factor, v), 48, torch.device("cpu"), torch.float64,
        seed=5, max_steps=5000, tolerance=1e-13, restarts=2,
    )
    assert converged
    assert 1.0 / inverse_max == pytest.approx(exact, rel=1e-7)


def test_epsilon_matches_dense_operator_norm():
    """epsilon = ||A^{-1} Delta||_2 must match a dense spectral-norm computation."""
    dimension = 40
    system = _spd(dimension, seed=21, ridge=1.0)
    factor = m17._cholesky(system)

    # A perturbed upper factor standing in for the decoded quantized factor.
    exact_upper = torch.linalg.cholesky(system).T
    generator = torch.Generator().manual_seed(99)
    noise = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    upper = exact_upper + 1e-3 * torch.triu(noise)

    delta_dense = upper.T @ upper - system
    dense_epsilon = float(
        torch.linalg.matrix_norm(torch.linalg.solve(system, delta_dense), ord=2)
    )

    def delta(vector):
        return upper.T @ (upper @ vector) - system @ vector

    def normal_operator(vector):
        return delta(m17._solve(factor, m17._solve(factor, delta(vector))))

    squared, converged, _, _ = m17._power_iteration(
        normal_operator, dimension, torch.device("cpu"), torch.float64,
        seed=7, max_steps=5000, tolerance=1e-13, restarts=2,
    )
    assert converged
    assert math.sqrt(squared) == pytest.approx(dense_epsilon, rel=1e-6)


def test_matrix_free_delta_equals_dense_delta():
    """The matrix-free Delta product must equal the materialised one."""
    dimension = 32
    system = _spd(dimension, seed=31, ridge=0.25)
    upper = torch.linalg.cholesky(system).T * 1.01
    dense = upper.T @ upper - system
    generator = torch.Generator().manual_seed(4)
    vector = torch.randn(dimension, generator=generator, dtype=torch.float64)
    free = upper.T @ (upper @ vector) - system @ vector
    assert torch.allclose(free, dense @ vector, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# the scientific claim: the bound must dominate the realised weight error
# ---------------------------------------------------------------------------
def test_ridge_bound_dominates_realised_weight_error():
    """eps/(1-eps) must upper-bound the relative classifier perturbation."""
    dimension = 36
    classes = 5
    system = _spd(dimension, seed=41, ridge=2.0)
    factor = m17._cholesky(system)
    exact_upper = torch.linalg.cholesky(system).T
    generator = torch.Generator().manual_seed(77)
    upper = exact_upper + 1e-3 * torch.triu(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    perturbed = upper.T @ upper

    targets = torch.randn(dimension, classes, generator=generator, dtype=torch.float64)
    weights = torch.linalg.solve(system, targets)
    perturbed_weights = torch.linalg.solve(perturbed, targets)
    realised = float(
        torch.linalg.matrix_norm(perturbed_weights - weights, ord="fro")
        / torch.linalg.matrix_norm(weights, ord="fro")
    )

    def delta(vector):
        return upper.T @ (upper @ vector) - system @ vector

    squared, _, _, _ = m17._power_iteration(
        lambda v: delta(m17._solve(factor, m17._solve(factor, delta(v)))),
        dimension, torch.device("cpu"), torch.float64,
        seed=9, max_steps=5000, tolerance=1e-13, restarts=2,
    )
    epsilon = math.sqrt(squared)
    assert epsilon < 1.0, "test fixture must stay inside the bound's regime"
    assert epsilon / (1.0 - epsilon) >= realised


def test_audit_task_reports_consistent_condition_number():
    dimension = 30
    ridge = 1.5
    gram = _spd(dimension, seed=51, ridge=0.0)
    system = gram.clone()
    system.diagonal().add_(ridge)
    eigenvalues = torch.linalg.eigvalsh(system)
    upper = torch.linalg.cholesky(system).T

    audit = m17._audit_task(
        gram=gram,
        ridge=ridge,
        compressed_factor=upper,
        dimension=dimension,
        device=torch.device("cpu"),
        dtype=torch.float64,
        settings={
            "power_iteration_seed": 8191,
            "power_iteration_max_steps": 5000,
            "power_iteration_tolerance": 1e-13,
            "power_iteration_restarts": 2,
        },
    )
    assert audit["lambda_max"] == pytest.approx(float(eigenvalues.max()), rel=1e-7)
    assert audit["lambda_min"] == pytest.approx(float(eigenvalues.min()), rel=1e-7)
    assert audit["condition_number"] == pytest.approx(
        float(eigenvalues.max() / eigenvalues.min()), rel=1e-6
    )
    # An unperturbed factor leaves only rounding error, which must be reported
    # as sitting at the noise floor rather than as a convergence failure.
    assert audit["epsilon"] < 1e-8
    assert audit["epsilon_at_noise_floor"] is True
    assert audit["epsilon"] <= audit["epsilon_noise_floor"]
    assert audit["cholesky_relative_residual"] < 1e-10
    assert audit["power_iteration_converged"]


def test_audit_resolves_a_real_perturbation_above_the_noise_floor():
    """A genuine INT8-scale perturbation must land well clear of the floor."""
    dimension = 30
    ridge = 1.5
    gram = _spd(dimension, seed=53, ridge=0.0)
    system = gram.clone()
    system.diagonal().add_(ridge)
    generator = torch.Generator().manual_seed(123)
    upper = torch.linalg.cholesky(system).T + 1e-3 * torch.triu(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    audit = m17._audit_task(
        gram=gram, ridge=ridge, compressed_factor=upper, dimension=dimension,
        device=torch.device("cpu"), dtype=torch.float64,
        settings={
            "power_iteration_seed": 8191,
            "power_iteration_max_steps": 5000,
            "power_iteration_tolerance": 1e-13,
            "power_iteration_restarts": 2,
        },
    )
    assert audit["epsilon_at_noise_floor"] is False
    assert audit["epsilon_over_noise_floor"] > 1e3
    assert audit["power_iteration_converged"]
    assert 0.0 < audit["epsilon"] < 1.0


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _record(**overrides):
    record = {
        "task": 1,
        "width": 10000,
        "lambda_max": 10.0,
        "lambda_min": 1.0,
        "condition_number": 10.0,
        "epsilon": 0.05,
        "ridge_perturbation_bound": 0.05 / 0.95,
        "cholesky_relative_residual": 1e-9,
        "power_iteration_converged": True,
        "m7_relative_weight_error": 0.02,
        "bound_over_measured": (0.05 / 0.95) / 0.02,
        "power_iteration_steps": {},
        "epsilon_at_noise_floor": False,
        "epsilon_noise_floor": 1e-12,
        "epsilon_over_noise_floor": 5e10,
        "replay_local_factor_error": 0.006,
        "m7_local_factor_error": 0.006,
        "replay_local_factor_relative_drift": 0.0,
        "power_iteration_relative_change": {},
    }
    record.update(overrides)
    return record


def _blocks(records_by_width):
    return [
        {"width": width, "ridge_lambda": 1e6, "records": records}
        for width, records in records_by_width.items()
    ]


def _config_for_gates():
    config = m17._read_config(CONFIG_PATH)
    config["num_tasks"] = 2
    return config


def test_gates_pass_on_consistent_records():
    config = _config_for_gates()
    blocks = _blocks({
        10000: [_record(task=1), _record(task=2)],
        20000: [_record(task=1, width=20000), _record(task=2, width=20000)],
    })
    verdict = m17._summarize(blocks, config, {"source_m6": {"artifact_sha256": "a"}, "source_m7": {"artifact_sha256": "b"}})
    assert verdict["status"] == m17.STUDY_STATUS_PASS
    assert all(verdict["gates"].values())


def test_gate_fails_when_bound_is_violated():
    """A bound that does not dominate the measurement must fail, not be excused."""
    config = _config_for_gates()
    bad = _record(task=2, epsilon=0.001, ridge_perturbation_bound=0.001001,
                  m7_relative_weight_error=0.5)
    blocks = _blocks({
        10000: [_record(task=1), bad],
        20000: [_record(task=1, width=20000), _record(task=2, width=20000)],
    })
    verdict = m17._summarize(blocks, config, {"source_m6": {"artifact_sha256": "a"}, "source_m7": {"artifact_sha256": "b"}})
    assert verdict["gates"]["bound_dominates_measured_weight_error"] is False
    assert verdict["status"] == m17.STUDY_STATUS_FAIL


def test_gate_fails_when_epsilon_reaches_one():
    config = _config_for_gates()
    blocks = _blocks({
        10000: [_record(task=1), _record(task=2, epsilon=1.4,
                                         ridge_perturbation_bound=float("inf"))],
        20000: [_record(task=1, width=20000), _record(task=2, width=20000)],
    })
    verdict = m17._summarize(blocks, config, {"source_m6": {"artifact_sha256": "a"}, "source_m7": {"artifact_sha256": "b"}})
    assert verdict["gates"]["epsilon_below_one"] is False
    assert verdict["status"] == m17.STUDY_STATUS_FAIL


def test_gate_fails_on_incomplete_sweep():
    config = _config_for_gates()
    blocks = _blocks({10000: [_record(task=1), _record(task=2)]})
    verdict = m17._summarize(blocks, config, {"source_m6": {"artifact_sha256": "a"}, "source_m7": {"artifact_sha256": "b"}})
    assert verdict["gates"]["all_widths_and_tasks_complete"] is False


def test_gate_fails_without_power_iteration_convergence():
    config = _config_for_gates()
    blocks = _blocks({
        10000: [_record(task=1), _record(task=2, power_iteration_converged=False)],
        20000: [_record(task=1, width=20000), _record(task=2, width=20000)],
    })
    verdict = m17._summarize(blocks, config, {"source_m6": {"artifact_sha256": "a"}, "source_m7": {"artifact_sha256": "b"}})
    assert verdict["gates"]["power_iteration_convergence"] is False


def test_gate_fails_when_replay_diverges_from_m7():
    """If the replay did not regenerate M7's stream, the comparison is void."""
    config = _config_for_gates()
    drifted = _record(task=2, replay_local_factor_relative_drift=1e-2)
    blocks = _blocks({
        10000: [_record(task=1), drifted],
        20000: [_record(task=1, width=20000), _record(task=2, width=20000)],
    })
    verdict = m17._summarize(
        blocks, config,
        {"source_m6": {"artifact_sha256": "a"}, "source_m7": {"artifact_sha256": "b"}},
    )
    assert verdict["gates"]["replay_matches_m7"] is False
    assert verdict["status"] == m17.STUDY_STATUS_FAIL


def test_precision_lock_disables_tf32_and_records_versions():
    locked = m17._lock_precision()
    assert locked.get("cuda_matmul_allow_tf32") is False
    assert locked.get("cudnn_allow_tf32") is False
    assert "torch_version" in locked
