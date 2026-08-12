"""Small cache-adapter tests; no image/backbone download is involved."""

import torch

from methods.cached_replay_baselines import CachedFlyCL, CachedSOHOReplay


def _data():
    torch.manual_seed(77)
    return torch.randn(18, 5), torch.tensor([8, 2, 8, 4, 2, 4] * 3)


def test_cached_fly_is_streaming_global_and_resumable():
    x, y = _data()
    learner = CachedFlyCL(5, 16, 3, .25, .2, seed=12)
    learner.update(x[:9], y[:9])
    learner.update(x[9:], y[9:])
    logits = learner.predict_logits(x[:4])
    assert learner.is_exemplar_free is True
    assert learner.class_ids == [2, 4, 8]
    assert logits.shape == (4, 3) and bool(torch.isfinite(logits).all())
    assert learner.persistent_state_bytes() > 0

    clone = CachedFlyCL(5, 16, 3, .25, .2, seed=12)
    clone.load_state_dict(learner.state_dict())
    torch.testing.assert_close(clone.predict_logits(x[:4]), logits, rtol=0, atol=1e-6)


def test_cached_soho_replay_is_explicitly_disclosed_and_resumable():
    x, y = _data()
    learner = CachedSOHOReplay(5, 16, .4, 5, True, .25, .2, seed=13)
    learner.update(x[:9], y[:9])
    learner.update(x[9:], y[9:])
    logits = learner.predict_logits(x[:4])
    assert learner.is_exemplar_free is False
    assert learner.class_ids == [2, 4, 8]
    assert logits.shape == (4, 3) and bool(torch.isfinite(logits).all())
    assert sum(item.shape[0] for item in learner.feature_history) == len(x)

    clone = CachedSOHOReplay(5, 16, .4, 5, True, .25, .2, seed=13)
    clone.load_state_dict(learner.state_dict())
    torch.testing.assert_close(clone.predict_logits(x[:4]), logits, rtol=0, atol=1e-5)
    clone.update(x[:3], y[:3])
    assert clone.predict_logits(x[:2]).shape == (2, 3)
