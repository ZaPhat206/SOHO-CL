import math

import pytest
import torch

from methods.zi_soho.learner import ZISOHOLearner
from methods.zi_soho.statistics import ZeroInflatedStatistics


DTYPE = torch.float64


def sparse_stream():
    indices = torch.tensor([
        [0, 2], [1, 2], [0, 3], [1, 3], [0, 1], [2, 3],
    ])
    values = torch.tensor([
        [0.0, 1.0], [1.5, 0.5], [1.0, 2.5],
        [0.5, 3.0], [2.2, 1.2], [1.7, 2.7],
    ], dtype=DTYPE)
    labels = torch.tensor([9, 4, 9, 4, 9, 4])
    return indices, values, labels


def dense_codes(indices, values, dimension=4):
    result = torch.zeros((indices.shape[0], dimension), dtype=values.dtype)
    return result.scatter(1, indices, values)


def test_streaming_zero_inflated_statistics_equal_batch_oracle():
    indices, values, labels = sparse_stream()
    stats = ZeroInflatedStatistics(4, dtype=DTYPE)
    stats.update_sparse(indices[:2], values[:2], labels[:2])
    stats.update_sparse(indices[2:5], values[2:5], labels[2:5])
    stats.update_sparse(indices[5:], values[5:], labels[5:])
    assert stats.class_ids == [4, 9]
    torch.testing.assert_close(stats.counts, torch.tensor([3.0, 3.0], dtype=DTYPE))
    for column, class_id in enumerate(stats.class_ids):
        mask = labels == class_id
        selected = indices[mask].reshape(-1)
        amplitudes = values[mask].reshape(-1)
        expected_count = torch.zeros(4, dtype=DTYPE).scatter_add_(
            0, selected, torch.ones_like(amplitudes)
        )
        expected_sum = torch.zeros(4, dtype=DTYPE).scatter_add_(
            0, selected, amplitudes
        )
        expected_square = torch.zeros(4, dtype=DTYPE).scatter_add_(
            0, selected, amplitudes.square()
        )
        torch.testing.assert_close(stats.active_counts[:, column], expected_count)
        torch.testing.assert_close(stats.active_sums[:, column], expected_sum)
        torch.testing.assert_close(stats.active_sq_sums[:, column], expected_square)


def test_statistics_are_partition_and_order_invariant():
    indices, values, labels = sparse_stream()
    batch = ZeroInflatedStatistics(4, dtype=DTYPE)
    batch.update_sparse(indices, values, labels)
    permutation = torch.tensor([5, 1, 3, 0, 4, 2])
    streamed = ZeroInflatedStatistics(4, dtype=DTYPE)
    for part in permutation.split(2):
        streamed.update_sparse(indices[part], values[part], labels[part])

    assert streamed.class_ids == batch.class_ids
    for name in ("counts", "active_counts", "active_sums", "active_sq_sums"):
        torch.testing.assert_close(getattr(streamed, name), getattr(batch, name))


def make_learner(method, chunk=2):
    return ZISOHOLearner(
        raw_dim=3,
        expand_dim=4,
        synaptic_degree=2,
        coding_level=0.5,
        method=method,
        support_alpha=0.5,
        variance_kappa=2.0,
        variance_epsilon=1e-3,
        score_chunk_size=chunk,
        seed=13,
        dtype=DTYPE,
    )


def manual_logits(model, indices, values):
    parameters = model._parameters()
    output = torch.empty((len(indices), len(model.class_ids)), dtype=DTYPE)
    for row in range(len(indices)):
        active = set(indices[row].tolist())
        dense = torch.zeros(model.expand_dim, dtype=DTYPE)
        dense[indices[row]] = values[row]
        for column in range(len(model.class_ids)):
            if model.method == "wta_ncm":
                mean = parameters["wta_mean"][:, column]
                output[row, column] = -(dense - mean).square().sum()
                continue
            score = torch.tensor(0.0, dtype=DTYPE)
            for coordinate in range(model.expand_dim):
                probability = parameters["probability"][coordinate, column]
                if coordinate not in active:
                    if model.method in {"support_only", "hurdle"}:
                        score += torch.log1p(-probability)
                    continue
                if model.method in {"support_only", "hurdle"}:
                    score += torch.log(probability)
                if model.method in {"active_gaussian", "hurdle"}:
                    mean = parameters["mean"][coordinate, column]
                    variance = parameters["variance"][coordinate, column]
                    score += -0.5 * (
                        math.log(2 * math.pi)
                        + torch.log(variance)
                        + (dense[coordinate] - mean).square() / variance
                    )
            output[row, column] = score
    return output


@pytest.mark.parametrize(
    "method", ["wta_ncm", "support_only", "active_gaussian", "hurdle"]
)
def test_sparse_scorers_equal_direct_batch_formulas(method):
    indices, values, labels = sparse_stream()
    model = make_learner(method)
    model.update_from_sparse(indices, values, labels)
    query_indices, query_values = indices[[1, 4]], values[[1, 4]]
    actual = model.predict_logits_from_sparse(query_indices, query_values)
    expected = manual_logits(model, query_indices, query_values)
    if method == "wta_ncm":
        # The implementation omits -||z||^2, a class-independent argmax term.
        expected += query_values.square().sum(1, keepdim=True)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("method", ["support_only", "active_gaussian", "hurdle"])
def test_coordinate_chunking_does_not_change_logits(method):
    indices, values, labels = sparse_stream()
    small = make_learner(method, chunk=1)
    large = make_learner(method, chunk=10)
    small.update_from_sparse(indices, values, labels)
    large.load_state_dict({**small.state_dict(), "score_chunk_size": 10})
    torch.testing.assert_close(
        small.predict_logits_from_sparse(indices, values),
        large.predict_logits_from_sparse(indices, values),
        atol=1e-11,
        rtol=1e-11,
    )
