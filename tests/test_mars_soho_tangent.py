import torch

from methods.mars_soho.tangent import (
    TangentClassSketch,
    sphere_exp_map,
    sphere_log_map,
)


def _spherical_classes():
    torch.manual_seed(101)
    labels = torch.arange(3).repeat_interleave(24)
    centers = torch.eye(5, dtype=torch.float64)[:3]
    features = centers[labels] + 0.12 * torch.randn(72, 5, dtype=torch.float64)
    return torch.nn.functional.normalize(features, p=2, dim=1), labels


def test_sphere_log_exp_roundtrip_and_tangency():
    features, _ = _spherical_classes()
    base = torch.nn.functional.normalize(features[:24].mean(dim=0), p=2, dim=0)
    tangent = sphere_log_map(features[:8], base)
    assert torch.allclose(tangent @ base, torch.zeros(8, dtype=torch.float64), atol=1e-10)
    reconstructed = sphere_exp_map(tangent, base)
    assert torch.allclose(reconstructed, features[:8], atol=1e-9, rtol=1e-9)


def test_tangent_sketch_is_deterministic_finite_and_exemplar_free():
    features, labels = _spherical_classes()
    first = TangentClassSketch(
        feature_dim=5, rank=3, calibrated=True, seed=2025, dtype=torch.float64
    )
    second = TangentClassSketch(
        feature_dim=5, rank=3, calibrated=True, seed=2025, dtype=torch.float64
    )
    first.fit(features, labels)
    second.fit(features, labels)
    for name in first.persistent_tensors():
        assert torch.equal(
            first.persistent_tensors()[name], second.persistent_tensors()[name]
        )
    generated = first.generate(1, 12)
    assert torch.equal(generated, second.generate(1, 12))
    assert torch.allclose(
        generated.norm(dim=1), torch.ones(12, dtype=torch.float64), atol=1e-12
    )
    assert not any(
        features.shape[0] in tensor.shape
        for tensor in first.persistent_tensors().values()
    )
    first.assert_exemplar_free_state()


def test_resultant_calibration_improves_pseudo_mean_length():
    features, labels = _spherical_classes()
    raw = TangentClassSketch(
        feature_dim=5, rank=3, calibrated=False, seed=2025, dtype=torch.float64
    )
    calibrated = TangentClassSketch(
        feature_dim=5, rank=3, calibrated=True, seed=2025, dtype=torch.float64
    )
    raw.fit(features, labels)
    calibrated.fit(features, labels)
    column = calibrated.class_ids.index(0)
    target = calibrated.resultant_lengths[column]
    raw_error = (raw.generate(0, 256, stream_offset=97).mean(dim=0).norm() - target).abs()
    calibrated_error = (
        calibrated.generate(0, 256, stream_offset=97).mean(dim=0).norm() - target
    ).abs()
    assert calibrated_error <= raw_error + 1e-12
    assert calibrated_error < 1e-5
