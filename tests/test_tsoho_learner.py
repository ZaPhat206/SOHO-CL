import inspect
import torch

from methods.t_soho import create_learner
from methods.t_soho.graph import confusion_graph


def _stream():
    torch.manual_seed(41)
    return torch.randn(12, 5), torch.tensor([4, 4, 8, 8, 12, 12, 4, 8, 12, 4, 8, 12])


def test_statistics_match_batch_and_expansion_preserves_old_columns():
    x, y = _stream(); learner = create_learner(method="raw_ridge", feature_dim=5, ridge_lambda=.2)
    learner.update(x[:6], y[:6]); old_q = learner.statistics.Q.clone(); old_ids = learner.class_ids[:]
    learner.update(x[6:], y[6:])
    ids = learner.class_ids; columns = torch.tensor([ids.index(int(v)) for v in y])
    Y = torch.nn.functional.one_hot(columns, len(ids)).double()
    torch.testing.assert_close(learner.statistics.G, x.double().T @ x.double())
    torch.testing.assert_close(learner.statistics.Q, x.double().T @ Y)
    assert old_ids == ids and torch.equal(old_q, learner.statistics.Q[:, :len(old_ids)] - x[6:].double().T @ Y[6:, :len(old_ids)])


def test_counts_sums_sq_sums_and_raw_ridge_match_batch_oracle():
    x, y = _stream(); learner = create_learner(method="raw_ridge", feature_dim=5, ridge_lambda=.3); learner.update(x[:5], y[:5]); learner.update(x[5:], y[5:])
    ids = learner.class_ids; idx = torch.tensor([ids.index(int(v)) for v in y]); Y = torch.nn.functional.one_hot(idx, len(ids)).double()
    assert torch.equal(learner.statistics.counts, Y.sum(0))
    torch.testing.assert_close(learner.statistics.sums, Y.T @ x.double())
    torch.testing.assert_close(learner.statistics.sq_sums, Y.T @ x.double().square())
    W = torch.linalg.solve(x.double().T @ x.double() + .3 * torch.eye(5, dtype=torch.float64), x.double().T @ Y)
    query = torch.randn(4, 5)
    torch.testing.assert_close(learner.predict_logits(query), 2 * query.double() @ W - 1)
    assert torch.equal(learner.predict_logits(query).argmax(1), (query.double() @ W).argmax(1))


def test_graph_codes_logits_state_and_seed_contracts():
    x, y = _stream(); a = create_learner(method="spectral_confusion_code", feature_dim=5, ridge_lambda=.1, requested_rank=8, seed=9); a.update(x, y)
    A, L, _ = confusion_graph(a.statistics.counts, a.statistics.sums, a.statistics.sq_sums)
    torch.testing.assert_close(A, A.T); torch.testing.assert_close(torch.diag(A), torch.zeros(3, dtype=torch.float64)); torch.testing.assert_close(L, L.T); torch.testing.assert_close(L.sum(1), torch.zeros(3, dtype=torch.float64), atol=1e-10, rtol=0)
    assert a.E.shape == (2, 3); torch.testing.assert_close(a.E @ a.E.T, torch.eye(2, dtype=torch.float64), atol=1e-7, rtol=0); torch.testing.assert_close(a.E @ torch.ones(3, dtype=torch.float64), torch.zeros(2, dtype=torch.float64), atol=1e-7, rtol=0)
    logits = a.predict_logits(x[:3]); assert logits.shape == (3, 3) and bool(torch.isfinite(logits).all())
    assert "task_id" not in inspect.signature(a.predict_logits).parameters
    clone = create_learner(method="spectral_confusion_code", feature_dim=5, ridge_lambda=.1, requested_rank=8, seed=9); clone.load_state_dict(a.state_dict()); torch.testing.assert_close(clone.predict_logits(x), a.predict_logits(x))
    random_a = create_learner(method="random_orthogonal_code", feature_dim=5, ridge_lambda=.1, requested_rank=2, seed=1); random_b = create_learner(method="random_orthogonal_code", feature_dim=5, ridge_lambda=.1, requested_rank=2, seed=2); random_a.update(x,y); random_b.update(x,y); assert not torch.allclose(random_a.E, random_b.E)


def test_single_class_and_ill_conditioned_solver_are_defined():
    learner = create_learner(method="spectral_confusion_code", feature_dim=3, ridge_lambda=1e-3, requested_rank=99); learner.update(torch.ones(4,3), torch.tensor([7,7,7,7]))
    assert learner.E is None and learner.predict_logits(torch.ones(2,3)).shape == (2,1)
