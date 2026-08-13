import inspect

import pytest
import torch

from methods.crt_soho.geometry import (
    anchor_weights,
    relative_margin_affinity,
    shuffled_affinity,
)
from methods.crt_soho.learner import METHODS, CRTSOHOLearner
from methods.crt_soho.solver import (
    reconstruct_residual_statistics,
    schur_residual_directions,
    solve_block_ridge,
)
from methods.crt_soho.statistics import DualViewStatistics


DTYPE = torch.float64


def synthetic_views(seed=17):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(24, 5, generator=generator, dtype=DTYPE)
    phi = torch.randn(24, 7, generator=generator, dtype=DTYPE)
    labels = torch.tensor([7, 3, 11, 3, 7, 11] * 4)
    return x, phi, labels


def accumulated_statistics(x, phi, labels):
    statistics = DualViewStatistics(x.shape[1], phi.shape[1], dtype=DTYPE)
    statistics.update(x[:9], phi[:9], labels[:9])
    statistics.update(x[9:17], phi[9:17], labels[9:17])
    statistics.update(x[17:], phi[17:], labels[17:])
    return statistics


def test_dual_view_streaming_statistics_equal_batch_oracle():
    x, phi, labels = synthetic_views()
    streaming = accumulated_statistics(x, phi, labels)
    batch = DualViewStatistics(5, 7, dtype=DTYPE)
    batch.update(x, phi, labels)

    assert streaming.class_ids == batch.class_ids == [3, 7, 11]
    for name in ("G_pp", "G_xx", "H_px", "Q_p", "Q_x", "counts"):
        torch.testing.assert_close(getattr(streaming, name), getattr(batch, name), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("residualize", [True, False])
def test_reconstructed_residual_statistics_equal_explicit_sample_oracle(residualize):
    x, phi, labels = synthetic_views()
    statistics = accumulated_statistics(x, phi, labels)
    generator = torch.Generator().manual_seed(4)
    directions, _ = torch.linalg.qr(torch.randn(5, 3, generator=generator, dtype=DTYPE))
    reconstructed = reconstruct_residual_statistics(
        statistics, directions, complement_ridge=0.31, residualize=residualize
    )
    if residualize:
        eye = torch.eye(7, dtype=DTYPE)
        expected_c = torch.linalg.solve(phi.T @ phi + 0.31 * eye, phi.T @ x @ directions)
    else:
        expected_c = torch.zeros(7, 3, dtype=DTYPE)
    residual = x @ directions - phi @ expected_c
    targets = torch.nn.functional.one_hot(
        torch.tensor([statistics.class_ids.index(int(label)) for label in labels]), 3
    ).to(DTYPE)

    torch.testing.assert_close(reconstructed.C, expected_c, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(reconstructed.G_pr, phi.T @ residual, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(reconstructed.G_rr, residual.T @ residual, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(reconstructed.Q_r, residual.T @ targets, atol=1e-11, rtol=1e-11)


def test_block_solver_equals_direct_augmented_ridge_oracle():
    x, phi, labels = synthetic_views()
    statistics = accumulated_statistics(x, phi, labels)
    directions, _ = torch.linalg.qr(torch.randn(5, 3, generator=torch.Generator().manual_seed(8), dtype=DTYPE))
    residual = reconstruct_residual_statistics(statistics, directions, complement_ridge=0.4)
    anchor_w, residual_w, equation_error = solve_block_ridge(
        statistics, residual, anchor_ridge=0.2, residual_ridge=0.7
    )

    explicit_r = x @ directions - phi @ residual.C
    design = torch.cat((phi, explicit_r), dim=1)
    targets = torch.nn.functional.one_hot(
        torch.tensor([statistics.class_ids.index(int(label)) for label in labels]), 3
    ).to(DTYPE)
    penalty = torch.diag(torch.tensor([0.2] * 7 + [0.7] * 3, dtype=DTYPE))
    oracle = torch.linalg.solve(design.T @ design + penalty, design.T @ targets)

    torch.testing.assert_close(torch.cat((anchor_w, residual_w)), oracle, atol=1e-10, rtol=1e-10)
    assert equation_error < 1e-10


def test_schur_directions_are_deterministic_orthonormal_and_energy_monotone():
    x, phi, labels = synthetic_views()
    statistics = accumulated_statistics(x, phi, labels)
    rank_one = schur_residual_directions(statistics, 0.4, 0.2, 0.7, 1)
    rank_three = schur_residual_directions(statistics, 0.4, 0.2, 0.7, 3)
    repeated = schur_residual_directions(statistics, 0.4, 0.2, 0.7, 3)

    torch.testing.assert_close(rank_three.directions.T @ rank_three.directions, torch.eye(3, dtype=DTYPE))
    torch.testing.assert_close(rank_three.directions, repeated.directions)
    assert rank_one.retained_correction_energy <= rank_three.retained_correction_energy
    assert rank_three.retained_correction_energy == pytest.approx(1.0, abs=1e-12)
    assert rank_three.effective_rank == 3


def test_rank_one_schur_subspace_maximizes_block_ridge_correction_energy():
    x, phi, labels = synthetic_views()
    statistics = accumulated_statistics(x, phi, labels)
    full = reconstruct_residual_statistics(
        statistics, torch.eye(5, dtype=DTYPE), complement_ridge=0.4
    )
    anchor_system = statistics.G_pp + 0.2 * torch.eye(7, dtype=DTYPE)
    solved_cross = torch.linalg.solve(anchor_system, full.G_pr)
    solved_targets = torch.linalg.solve(anchor_system, statistics.Q_p)
    schur = full.G_rr + 0.7 * torch.eye(5, dtype=DTYPE) - full.G_pr.T @ solved_cross
    target = full.Q_r - full.G_pr.T @ solved_targets

    def correction_energy(direction):
        reduced_system = direction.T @ schur @ direction
        reduced_target = direction.T @ target
        return float((reduced_target.T @ torch.linalg.solve(reduced_system, reduced_target)).trace())

    selected = schur_residual_directions(statistics, 0.4, 0.2, 0.7, 1).directions
    selected_energy = correction_energy(selected)
    generator = torch.Generator().manual_seed(103)
    random_energies = []
    for _ in range(50):
        direction = torch.randn(5, 1, generator=generator, dtype=DTYPE)
        direction /= torch.linalg.vector_norm(direction)
        random_energies.append(correction_energy(direction))

    assert selected_energy >= max(random_energies) - 1e-12


def test_relative_confusion_is_symmetric_nonnegative_and_shuffle_preserves_edges():
    x, phi, labels = synthetic_views()
    statistics = accumulated_statistics(x, phi, labels)
    affinity = relative_margin_affinity(statistics, anchor_weights(statistics, 0.2), temperature=0.8)
    shuffled = shuffled_affinity(affinity, seed=9)
    rows, columns = torch.triu_indices(3, 3, 1)

    torch.testing.assert_close(affinity, affinity.T)
    assert bool((affinity >= 0).all())
    assert bool((torch.diag(affinity) == 0).all())
    torch.testing.assert_close(
        torch.sort(affinity[rows, columns]).values,
        torch.sort(shuffled[rows, columns]).values,
    )
    assert not torch.equal(affinity, shuffled)


def learner(method, seed=23):
    return CRTSOHOLearner(
        method=method,
        raw_dim=6,
        anchor_dim=13,
        synaptic_degree=3,
        coding_level=0.25,
        anchor_ridge=0.2,
        residual_ridge=0.4,
        complement_ridge=0.3,
        requested_rank=2,
        confusion_temperature=0.7,
        scatter_epsilon=1e-5,
        seed=seed,
        dtype=DTYPE,
    )


@pytest.mark.parametrize("method", sorted(METHODS))
def test_all_crt_methods_are_global_finite_and_task_id_free(method):
    model = learner(method)
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(30, 6, generator=generator)
    labels = torch.tensor([20, 4, 12] * 10)
    model.update(x[:15], labels[:15])
    model.update(x[15:], labels[15:])
    logits = model.predict_logits(x[:5])

    assert logits.shape == (5, 3)
    assert bool(torch.isfinite(logits).all())
    assert model.class_ids == [4, 12, 20]
    assert "task_id" not in inspect.signature(model.predict_logits).parameters
    model.assert_exemplar_free_state()
    assert model.persistent_state_bytes() > 0


@pytest.mark.parametrize("method", ["confusion_residual", "schur_residual"])
def test_checkpoint_contains_no_derived_or_sample_level_state_and_roundtrips_logits(method):
    original = learner(method)
    x = torch.randn(36, 6, generator=torch.Generator().manual_seed(41))
    labels = torch.tensor([9, 2, 5] * 12)
    original.update(x, labels)
    expected = original.predict_logits(x[:7])
    state = original.state_dict()

    assert set(state) == {
        "version", "method", "raw_dim", "anchor_dim", "synaptic_degree", "coding_level",
        "anchor_ridge", "residual_ridge", "complement_ridge", "requested_rank",
        "confusion_temperature", "scatter_epsilon", "seed", "anchor_projection", "statistics",
    }
    assert not any(token in str(state).lower() for token in ("sample", "replay", "historical"))
    restored = learner(method)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.predict_logits(x[:7]), expected, atol=1e-10, rtol=1e-10)


def test_checkpoint_configuration_mismatch_fails_closed():
    original = learner("anchor_only")
    x = torch.randn(12, 6, generator=torch.Generator().manual_seed(2))
    original.update(x, torch.tensor([0, 1, 2] * 4))
    incompatible = learner("anchor_only", seed=24)
    with pytest.raises(ValueError, match="seed"):
        incompatible.load_state_dict(original.state_dict())


def test_full_raw_residual_can_recover_information_absent_from_anchor():
    common = dict(
        raw_dim=4, anchor_dim=1, synaptic_degree=1, coding_level=1.0,
        anchor_ridge=0.1, residual_ridge=0.1, complement_ridge=0.1,
        requested_rank=2, seed=6, dtype=DTYPE,
    )
    anchor = CRTSOHOLearner(method="anchor_only", **common)
    augmented = CRTSOHOLearner(method="full_raw_residual", **common)
    used_coordinate = int(anchor.anchor.projection_matrix.to_dense().abs().argmax())
    unused = [coordinate for coordinate in range(4) if coordinate != used_coordinate]
    x = torch.zeros(20, 4)
    x[:, unused[0]] = 1.0
    x[:10, unused[1]], x[10:, unused[1]] = -1.0, 1.0
    labels = torch.tensor([0] * 10 + [1] * 10)

    anchor.update(x, labels)
    augmented.update(x, labels)
    anchor_accuracy = float((anchor.predict(x) == labels).float().mean())
    augmented_accuracy = float((augmented.predict(x) == labels).float().mean())

    assert anchor_accuracy == 0.5
    assert augmented_accuracy == 1.0


def test_full_rank_schur_reproduces_full_raw_residual_logits():
    common = dict(
        raw_dim=6, anchor_dim=13, synaptic_degree=3, coding_level=0.25,
        anchor_ridge=0.2, residual_ridge=0.4, complement_ridge=0.3,
        requested_rank=6, confusion_temperature=0.7, scatter_epsilon=1e-5,
        seed=23, dtype=DTYPE,
    )
    full = CRTSOHOLearner(method="full_raw_residual", **common)
    schur = CRTSOHOLearner(method="schur_residual", **common)
    x = torch.randn(42, 6, generator=torch.Generator().manual_seed(91))
    labels = torch.tensor([0, 1, 2] * 14)
    full.update(x, labels)
    schur.update(x, labels)

    torch.testing.assert_close(schur.predict_logits(x), full.predict_logits(x), atol=1e-9, rtol=1e-9)
    assert schur.diagnostics["effective_rank"] == 3
    assert schur.diagnostics["retained_correction_energy"] == pytest.approx(1.0, abs=1e-12)
