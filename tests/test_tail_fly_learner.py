import inspect

import pytest
import torch

from methods.tail_fly import TAILFlyLearner


DTYPE = torch.float64


def _learner(*, rank=5, ridge=0.2, seed=2025):
    return TAILFlyLearner(
        feature_dim=7,
        expand_dim=17,
        synaptic_degree=3,
        coding_level=4 / 17,
        max_rank=rank,
        ridge_lambda=ridge,
        seed=seed,
        dtype=DTYPE,
    )


def _codes(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    dense = torch.randn(rows, 17, generator=generator, dtype=DTYPE)
    values, indices = dense.topk(4, dim=1)
    result = torch.zeros_like(dense)
    result.scatter_(1, indices, values)
    return result


def test_streaming_exact_statistics_equal_batch_oracle_with_class_expansion():
    learner = _learner(rank=6)
    first_codes, second_codes = _codes(6, 1), _codes(7, 2)
    first_labels = torch.tensor([9, 3, 9, 3, 9, 3])
    second_labels = torch.tensor([1, 9, 1, 3, 1, 9, 3])
    learner.update_codes(first_codes, first_labels)
    learner.update_codes(second_codes, second_labels)

    codes = torch.cat((first_codes, second_codes))
    labels = torch.cat((first_labels, second_labels))
    class_ids = [1, 3, 9]
    columns = torch.tensor([class_ids.index(int(value)) for value in labels])
    targets = torch.nn.functional.one_hot(columns, num_classes=3).to(DTYPE)
    assert learner.class_ids == class_ids
    torch.testing.assert_close(learner.exact_diagonal, codes.square().sum(0))
    torch.testing.assert_close(learner.Q, codes.T @ targets)
    torch.testing.assert_close(learner.counts, targets.sum(0))


def test_full_rank_learner_matches_batch_fly_logits():
    learner = _learner(rank=17, ridge=0.4)
    first, second, test = _codes(5, 4), _codes(7, 5), _codes(4, 6)
    labels = torch.tensor([5, 2, 5, 2, 5, 8, 2, 8, 5, 2, 8, 5])
    learner.update_codes(first, labels[:5])
    learner.update_codes(second, labels[5:])
    all_codes = torch.cat((first, second))
    class_ids = [2, 5, 8]
    columns = torch.tensor([class_ids.index(int(value)) for value in labels])
    targets = torch.nn.functional.one_hot(columns, num_classes=3).to(DTYPE)
    oracle = torch.linalg.solve(
        all_codes.T @ all_codes + 0.4 * torch.eye(17, dtype=DTYPE),
        all_codes.T @ targets,
    )
    torch.testing.assert_close(
        learner.predict_logits_from_codes(test), test @ oracle, atol=1e-8, rtol=1e-8
    )


def test_checkpoint_round_trip_rebuilds_classifier_without_sample_rows():
    learner = _learner(rank=5)
    codes = _codes(13, 8)
    labels = torch.tensor([2, 4, 7, 2, 4, 7, 2, 4, 7, 2, 4, 7, 2])
    learner.update_codes(codes[:5], labels[:5])
    learner.update_codes(codes[5:], labels[5:])
    expected = learner.predict_logits_from_codes(_codes(3, 9))
    state = learner.state_dict()
    assert "weights" not in state
    assert "codes" not in state and "features" not in state and "labels" not in state

    restored = _learner(rank=5)
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.predict_logits_from_codes(_codes(3, 9)), expected
    )
    restored.assert_exemplar_free_state()
    assert restored.svd.total_rows == 13
    for tensor in restored.persistent_tensors().values():
        assert 13 not in tensor.shape


def test_projection_and_stream_are_deterministic_for_fixed_seed():
    left, right = _learner(rank=4), _learner(rank=4)
    codes, labels = _codes(9, 10), torch.tensor([0, 1, 2] * 3)
    left.update_codes(codes, labels)
    right.update_codes(codes, labels)
    torch.testing.assert_close(
        left.flyhash.projection_matrix.values(), right.flyhash.projection_matrix.values()
    )
    torch.testing.assert_close(left.svd.U, right.svd.U)
    torch.testing.assert_close(left.svd.s, right.svd.s)
    torch.testing.assert_close(left.weights, right.weights)


def test_accumulate_then_finalize_matches_same_stream_with_eager_solves():
    eager, deferred = _learner(rank=4), _learner(rank=4)
    codes = _codes(11, 15)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    eager.update_codes(codes[:5], labels[:5])
    eager.update_codes(codes[5:], labels[5:])
    deferred.accumulate_codes(codes[:5], labels[:5])
    deferred.accumulate_codes(codes[5:], labels[5:])
    assert deferred.weights is None
    deferred.finalize_update()
    torch.testing.assert_close(deferred.svd.U, eager.svd.U)
    torch.testing.assert_close(deferred.svd.s, eager.svd.s)
    torch.testing.assert_close(deferred.weights, eager.weights)


def test_runtime_shapes_and_global_predictions_need_no_task_id():
    learner = _learner(rank=4)
    features = torch.randn(8, 7, generator=torch.Generator().manual_seed(11), dtype=DTYPE)
    labels = torch.tensor([10, 10, 3, 3, 10, 3, 10, 3])
    learner.update(features, labels)
    logits = learner.predict_logits(features[:3])
    assert logits.shape == (3, 2)
    assert learner.predict(features[:3]).shape == (3,)
    assert "task_id" not in inspect.signature(learner.predict_logits).parameters
    assert learner.diagnostics["solver_relative_residual"] < 1e-5


def test_checkpoint_configuration_mismatch_and_nonfinite_codes_fail():
    learner = _learner(rank=4)
    codes = _codes(5, 12)
    learner.update_codes(codes, torch.tensor([0, 1, 0, 1, 0]))
    state = learner.state_dict()
    with pytest.raises(ValueError, match="max_rank"):
        _learner(rank=3).load_state_dict(state)
    codes[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        learner.update_codes(codes, torch.tensor([0, 1, 0, 1, 0]))


def test_rank_zero_learner_keeps_only_diagonal_tail():
    learner = _learner(rank=0)
    codes = _codes(9, 13)
    labels = torch.tensor([0, 1, 2] * 3)
    learner.update_codes(codes, labels)
    targets = torch.nn.functional.one_hot(labels, num_classes=3).to(DTYPE)
    expected = (codes.T @ targets) / (
        codes.square().sum(0) + learner.ridge_lambda
    ).unsqueeze(1)
    torch.testing.assert_close(learner.weights, expected)
    assert learner.svd.U.shape == (17, 0)


def test_persistent_bytes_distinguish_aggregate_and_derived_classifier():
    learner = _learner(rank=3)
    learner.update_codes(_codes(8, 14), torch.tensor([0, 1] * 4))
    aggregate = learner.persistent_state_bytes(include_classifier=False)
    resident = learner.persistent_state_bytes(include_classifier=True)
    assert resident - aggregate == learner.weights.numel() * learner.weights.element_size()
