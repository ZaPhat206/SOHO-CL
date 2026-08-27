import torch

from methods.mars_soho.learner import DynamicSOHOMap
from methods.mars_soho.statistics import SphericalClassMoments
from methods.wbt_soho.transport import (
    BoundaryTransportMemory,
    certified_topk_support_stable,
    topk_gap,
    whiten_color_tangent_residuals,
)


def test_full_rank_whiten_color_matches_target_covariance():
    torch.manual_seed(17)
    samples = torch.randn(400, 3, dtype=torch.float64)
    samples -= samples.mean(dim=0)
    source_covariance = samples.T @ samples / (len(samples) - 1)
    source_values, source_vectors = torch.linalg.eigh(source_covariance)
    source_order = torch.argsort(source_values, descending=True)
    source_values = source_values[source_order]
    source_vectors = source_vectors[:, source_order]
    target_values = torch.tensor([2.0, 0.7, 0.2], dtype=torch.float64)
    target_vectors, _ = torch.linalg.qr(
        torch.randn(3, 3, dtype=torch.float64)
    )
    source_basis = torch.zeros(4, 3, dtype=torch.float64)
    target_basis = torch.zeros(4, 3, dtype=torch.float64)
    source_basis[:3] = source_vectors
    target_basis[:3] = target_vectors
    residuals = torch.zeros(400, 4, dtype=torch.float64)
    residuals[:, :3] = samples
    transported = whiten_color_tangent_residuals(
        residuals,
        source_basis=source_basis,
        source_eigenvalues=source_values,
        source_diagonal_residual=torch.zeros(4, dtype=torch.float64),
        target_basis=target_basis,
        target_eigenvalues=target_values,
        target_diagonal_residual=torch.zeros(4, dtype=torch.float64),
        target_origin=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
    )
    covariance = transported.T @ transported / (len(transported) - 1)
    expected = (target_basis * target_values.sqrt().unsqueeze(0)) @ (
        target_basis * target_values.sqrt().unsqueeze(0)
    ).T
    assert torch.allclose(transported.mean(0), torch.zeros(4, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(covariance, expected, atol=1e-10, rtol=1e-10)


def test_topk_margin_certificate_implies_unchanged_support():
    expanded = torch.tensor(
        [[4.0, 3.0, 1.0, 0.0], [2.0, 1.9, 1.8, 0.0]], dtype=torch.float64
    )
    perturbation = torch.tensor(
        [[0.1, -0.1, 0.1, 0.0], [0.2, -0.2, 0.2, 0.0]], dtype=torch.float64
    )
    certified = certified_topk_support_stable(expanded, perturbation, 2)
    assert certified.tolist() == [True, False]
    before = torch.topk(expanded[certified], 2, dim=1).indices.sort(1).values
    after = torch.topk(
        (expanded + perturbation)[certified], 2, dim=1
    ).indices.sort(1).values
    assert torch.equal(before, after)
    assert torch.allclose(
        topk_gap(expanded, 2),
        torch.tensor([2.0, 0.1], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def _classes(class_ids, *, samples=24, dimension=6):
    labels = torch.tensor(class_ids).repeat_interleave(samples)
    centers = torch.eye(dimension, dtype=torch.float64)[labels]
    generator = torch.Generator().manual_seed(91 + sum(class_ids))
    values = centers + 0.13 * torch.randn(
        len(labels), dimension, generator=generator, dtype=torch.float64
    )
    return torch.nn.functional.normalize(values, p=2, dim=1), labels


def test_boundary_transport_is_deterministic_reduces_gap_and_keeps_old_dominance():
    old_features, old_labels = _classes([0, 1])
    current_features, current_labels = _classes([2, 3])
    memory = BoundaryTransportMemory(
        feature_dim=6, rank=3, seed=2025, dtype=torch.float64
    )
    memory.update(old_features, old_labels)
    moments = SphericalClassMoments(6, dtype=torch.float64)
    moments.update(torch.cat((old_features, current_features)), torch.cat((old_labels, current_labels)))
    encoder = DynamicSOHOMap(
        feature_dim=6,
        expand_dim=24,
        density=0.5,
        olda_dim=6,
        coding_level=0.25,
        use_etf=True,
        seed=31,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    encoder.update_rotation(moments.snapshot())
    first = memory.transport(
        current_features=current_features,
        current_labels=current_labels,
        target_class_ids=[0, 1],
        count=16,
        encoder=encoder,
        mode="wta_boundary_transport",
        boundary_fraction=0.5,
        boundary_strength=0.5,
        stream_offset=7,
    )
    second = memory.transport(
        current_features=current_features,
        current_labels=current_labels,
        target_class_ids=[0, 1],
        count=16,
        encoder=encoder,
        mode="wta_boundary_transport",
        boundary_fraction=0.5,
        boundary_strength=0.5,
        stream_offset=7,
    )
    for class_id in [0, 1]:
        assert torch.equal(first.features[class_id], second.features[class_id])
        audit = first.diagnostics[class_id]
        assert audit["boundary_rows"] == 8
        assert audit["mean_topk_gap_after"] <= audit["mean_topk_gap_before"] + 1e-12
        assert audit["old_dominance_fraction"] == 1.0
    memory.assert_exemplar_free_state()
    assert not any(
        old_features.shape[0] in tensor.shape
        for tensor in memory.persistent_tensors().values()
    )
