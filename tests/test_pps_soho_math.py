import torch

from methods.pps_soho.sketch import FrequentDirections
from methods.pps_soho.solver import compact_factors, solve_compact_ridge
from methods.pps_soho.statistics import ClassProtectedStatistics


DTYPE = torch.float64


def data(seed=17):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randn(42, 8, generator=generator, dtype=DTYPE)
    labels = torch.tensor([9, 2, 5] * 14)
    return codes, labels


def targets(labels, class_ids):
    columns = torch.tensor([class_ids.index(int(value)) for value in labels])
    return torch.nn.functional.one_hot(columns, len(class_ids)).to(DTYPE)


def batch_scatter(codes, labels, class_ids):
    means, counts = [], []
    within = torch.zeros((codes.shape[1], codes.shape[1]), dtype=DTYPE)
    for class_id in class_ids:
        subset = codes[labels == class_id]
        mean = subset.mean(0)
        means.append(mean)
        counts.append(len(subset))
        centered = subset - mean
        within += centered.T @ centered
    return torch.stack(means, 1), torch.tensor(counts, dtype=DTYPE), within


def test_streaming_class_means_and_welford_scatter_equal_batch_oracle():
    codes, labels = data()
    statistics = ClassProtectedStatistics(8, 8, dtype=DTYPE)
    statistics.update(codes[:11], labels[:11])
    statistics.update(codes[11:29], labels[11:29])
    statistics.update(codes[29:], labels[29:])
    means, counts, within = batch_scatter(codes, labels, statistics.class_ids)

    torch.testing.assert_close(statistics.means, means, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(statistics.counts, counts, atol=0, rtol=0)
    torch.testing.assert_close(statistics.sketch.B.T @ statistics.sketch.B, within, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(statistics.cross, codes.T @ targets(labels, statistics.class_ids))
    reconstructed = statistics.between_factor().T @ statistics.between_factor() + within
    torch.testing.assert_close(reconstructed, codes.T @ codes, atol=1e-10, rtol=1e-10)


def test_frequent_directions_psd_error_and_accumulated_bound():
    codes, _ = data()
    sketch = FrequentDirections(8, 3, dtype=DTYPE)
    sketch.update(codes[:13])
    sketch.update(codes[13:])
    error = codes.T @ codes - sketch.B.T @ sketch.B
    eigenvalues = torch.linalg.eigvalsh((error + error.T) * 0.5)

    assert float(eigenvalues.min()) >= -1e-10
    assert float(eigenvalues.max()) <= sketch.covariance_error_bound + 1e-9
    assert sketch.B.shape == (3, 8)
    assert sketch.total_rows == len(codes)


def test_protected_tail_bound_is_no_worse_than_raw_tail_bound():
    codes, labels = data()
    class_ids = sorted(set(labels.tolist()))
    means, _, _ = batch_scatter(codes, labels, class_ids)
    mean_rows = torch.stack([means[:, class_ids.index(int(label))] for label in labels])
    residual = codes - mean_rows
    raw_singular = torch.linalg.svdvals(codes)
    residual_singular = torch.linalg.svdvals(residual)
    k = 2

    assert float(residual_singular[k:].square().sum()) <= float(raw_singular[k:].square().sum()) + 1e-10


def test_compact_woodbury_solver_equals_direct_structured_ridge():
    codes, labels = data()
    statistics = ClassProtectedStatistics(8, 8, dtype=DTYPE)
    statistics.update(codes, labels)
    ridge, gamma = 0.37, 0.6
    weights, relative = solve_compact_ridge(statistics, ridge, gamma)
    factors = compact_factors(statistics, gamma)
    direct = torch.linalg.solve(
        factors.T @ factors + ridge * torch.eye(8, dtype=DTYPE), statistics.cross
    )

    torch.testing.assert_close(weights, direct, atol=1e-11, rtol=1e-11)
    assert relative < 1e-11


def test_full_sketch_gamma_one_recovers_exact_fly_ridge_logits():
    codes, labels = data()
    statistics = ClassProtectedStatistics(8, 8, dtype=DTYPE)
    statistics.update(codes[:21], labels[:21])
    statistics.update(codes[21:], labels[21:])
    ridge = 0.25
    weights, _ = solve_compact_ridge(statistics, ridge, gamma=1.0)
    oracle = torch.linalg.solve(
        codes.T @ codes + ridge * torch.eye(8, dtype=DTYPE),
        codes.T @ targets(labels, statistics.class_ids),
    )

    torch.testing.assert_close(weights, oracle, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(codes @ weights, codes @ oracle, atol=1e-10, rtol=1e-10)


def test_ridge_coefficient_and_logit_errors_obey_certified_bounds():
    codes, labels = data()
    statistics = ClassProtectedStatistics(8, 3, dtype=DTYPE)
    statistics.update(codes, labels)
    ridge = 3.0
    approximate, _ = solve_compact_ridge(statistics, ridge, gamma=1.0)
    target = targets(labels, statistics.class_ids)
    oracle = torch.linalg.solve(
        codes.T @ codes + ridge * torch.eye(8, dtype=DTYPE), codes.T @ target
    )
    coefficient_bound = statistics.sketch.covariance_error_bound / ridge * torch.linalg.matrix_norm(oracle)
    observed = torch.linalg.matrix_norm(approximate - oracle)

    assert float(observed) <= float(coefficient_bound) + 1e-9
    test_codes = codes[:7]
    logit_error = torch.linalg.matrix_norm(test_codes @ (approximate - oracle), ord=2)
    generic_bound = torch.linalg.matrix_norm(test_codes, ord=2) * observed
    assert float(logit_error) <= float(generic_bound) + 1e-9


def test_global_fd_control_keeps_exact_cross_but_not_protected_between_term():
    codes, labels = data()
    statistics = ClassProtectedStatistics(8, 3, mode="global", dtype=DTYPE)
    statistics.update(codes[:19], labels[:19])
    statistics.update(codes[19:], labels[19:])

    torch.testing.assert_close(statistics.cross, codes.T @ targets(labels, statistics.class_ids))
    assert compact_factors(statistics, 1.0).shape == (3, 8)
