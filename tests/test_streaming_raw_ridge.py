import torch

from methods.streaming_raw_ridge import StreamingRawRidge


def _baseline(feature_dim=4):
    return StreamingRawRidge(
        backbone=torch.nn.Identity(),
        embedding_dim=feature_dim,
        ridge_lower=-1,
        ridge_upper=1,
        device=torch.device("cpu"),
    )


def test_streaming_classifier_matches_batch_oracle_on_toy_stream():
    torch.manual_seed(31)
    first_X = torch.randn(7, 4)
    first_y = torch.tensor([10, 10, 20, 20, 10, 20, 10])
    second_X = torch.randn(6, 4)
    second_y = torch.tensor([30, 10, 30, 20, 30, 10])
    learner = _baseline()
    learner.update_from_features(first_X, first_y)
    learner.update_from_features(second_X, second_y)

    X = torch.cat((first_X, second_X))
    y = torch.cat((first_y, second_y))
    class_ids = learner.class_ids
    Y = torch.nn.functional.one_hot(torch.tensor([class_ids.index(int(label)) for label in y]), len(class_ids)).float()
    ridge = learner.last_ridge
    W_oracle = torch.linalg.solve(X.T @ X + ridge * torch.eye(4), X.T @ Y)
    query = torch.randn(5, 4)

    torch.testing.assert_close(learner.logits_from_features(query), query @ W_oracle, rtol=0, atol=1e-5)
    assert learner.class_ids == [10, 20, 30]
    assert learner.Q_global.shape == (4, 3)


def test_persistent_state_inventory_has_no_historical_sample_dimension():
    learner = _baseline()
    learner.update_from_features(torch.randn(5, 4), torch.tensor([3, 3, 7, 7, 3]))
    summary = learner.persistent_state_summary()
    assert [entry["name"] for entry in summary["tensors"]] == ["G_global", "Q_global", "Wo"]
    assert [entry["shape"] for entry in summary["tensors"]] == [(4, 4), (4, 2), (4, 2)]
    assert summary["tensor_bytes"] == sum(entry["bytes"] for entry in summary["tensors"])
    learner.assert_exemplar_free_state()
