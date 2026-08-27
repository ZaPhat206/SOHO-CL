import torch

from methods.mars_soho.geometry import (
    align_projection_gauge,
    certified_stable_support,
    compute_soho_rotation,
    topk_support_turnover,
)
from methods.mars_soho.reconstruction import (
    SphericalReconstructor,
    allocate_pseudo_budget,
    shuffled_risks,
    wta_statistic_variance,
)
from methods.mars_soho.statistics import SphericalClassMoments


def test_streaming_spherical_moments_equal_batch_reference():
    torch.manual_seed(3)
    features = torch.randn(31, 6, dtype=torch.float64)
    labels = torch.tensor([index % 4 for index in range(31)])
    streaming = SphericalClassMoments(6)
    streaming.update(features[:11], labels[:11])
    streaming.update(features[11:23], labels[11:23])
    streaming.update(features[23:], labels[23:])
    batch = SphericalClassMoments(6)
    batch.update(features, labels)
    left, right = streaming.snapshot(), batch.snapshot()
    assert left.class_ids == right.class_ids
    assert left.total_count == right.total_count == 31
    for name in (
        "counts", "sums", "squared_sums", "within_scatter", "global_sum"
    ):
        assert torch.allclose(getattr(left, name), getattr(right, name), atol=1e-12, rtol=1e-12)


def test_projection_is_orthonormal_and_deterministic():
    torch.manual_seed(5)
    labels = torch.arange(4).repeat_interleave(8)
    features = torch.randn(32, 6, dtype=torch.float64)
    features += torch.nn.functional.one_hot(labels, 6).double()
    moments = SphericalClassMoments(6)
    moments.update(features, labels)
    first = compute_soho_rotation(moments.snapshot(), output_dim=6, use_etf=True)
    second = compute_soho_rotation(moments.snapshot(), output_dim=6, use_etf=True)
    identity = torch.eye(6, dtype=torch.float64)
    assert first.discriminative_rank == 3
    assert torch.allclose(first.rotation @ first.rotation.T, identity, atol=1e-10, rtol=1e-10)
    assert torch.allclose(first.rotation, second.rotation, atol=1e-12, rtol=1e-12)


def test_fixed_support_certificate_never_certifies_a_changed_support():
    torch.manual_seed(7)
    old = torch.randn(200, 17, dtype=torch.float64)
    new = old + 0.01 * torch.randn_like(old)
    k = 4
    certified = certified_stable_support(old, new, k)
    old_support = torch.topk(old, k, dim=1).indices.sort(dim=1).values
    new_support = torch.topk(new, k, dim=1).indices.sort(dim=1).values
    unchanged = (old_support == new_support).all(dim=1)
    assert bool(unchanged[certified].all())


def test_topk_turnover_is_exact_continuous_and_bounded():
    old = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    unchanged = torch.tensor([[5.1, 4.2, 3.3, 0.0, -1.0]])
    one_replaced = torch.tensor([[5.1, 4.2, 0.0, 3.3, -1.0]])
    assert torch.equal(topk_support_turnover(old, unchanged, 3), torch.tensor([0.0]))
    assert torch.allclose(
        topk_support_turnover(old, one_replaced, 3), torch.tensor([1 / 3])
    )


def test_wta_statistic_variance_detects_nonconstant_sufficient_statistics():
    constant = torch.tensor([[2.0, 0.0, 1.0]]).repeat(6, 1)
    variable = constant.clone()
    variable[::2] = torch.tensor([0.0, 3.0, 0.0])
    assert wta_statistic_variance(constant) == 0
    risk = wta_statistic_variance(variable)
    assert 0 < risk <= 1
    assert bool(torch.isfinite(risk))


def test_gauge_alignment_preserves_basis_and_reduces_arbitrary_null_rotation():
    torch.manual_seed(9)
    basis, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
    previous = basis.T
    null_mixing, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    changed = previous.clone()
    changed[3:] = null_mixing @ changed[3:]
    aligned = align_projection_gauge(
        changed, previous, discriminative_rank=3,
        eigenvalues=torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0, 0.0]),
        use_etf=True,
    )
    assert torch.allclose(
        aligned @ aligned.T, torch.eye(6, dtype=torch.float64), atol=1e-12
    )
    assert torch.linalg.vector_norm(aligned - previous) < torch.linalg.vector_norm(changed - previous)


def test_reconstruction_is_deterministic_antithetic_and_finite():
    torch.manual_seed(11)
    labels = torch.tensor([0] * 12 + [1] * 12)
    features = torch.randn(24, 5, dtype=torch.float64)
    features[labels == 1] += 0.7
    moments = SphericalClassMoments(5)
    moments.update(features, labels)
    reconstructor = SphericalReconstructor(
        moments.snapshot(), covariance_rank=3, shrinkage=0.25, seed=2025
    )
    first = reconstructor.generate(0, 8, heterogeneous=True)
    second = reconstructor.generate(0, 8, heterogeneous=True)
    assert torch.equal(first, second)
    assert bool(torch.isfinite(first).all())
    assert torch.allclose(first.norm(dim=1), torch.ones(8, dtype=torch.float64), atol=1e-12)


def test_support_budget_is_fixed_and_shuffled_control_preserves_values():
    class_ids = [2, 5, 9]
    counts = torch.tensor([10.0, 50.0, 20.0])
    risks = torch.tensor([0.01, 0.8, 0.2])
    allocation = allocate_pseudo_budget(
        class_ids, counts, risks,
        total_budget=60, minimum_per_class=4, risk_floor=1e-3,
    )
    assert set(allocation) == set(class_ids)
    assert sum(allocation.values()) == 60
    assert min(allocation.values()) >= 4
    shuffled = shuffled_risks(risks, seed=2025)
    assert torch.equal(shuffled.sort().values, risks.sort().values)
