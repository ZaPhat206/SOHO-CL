"""Small cache-adapter tests; no image/backbone download is involved."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from methods.cached_replay_baselines import (
    CachedFlyCL,
    CachedFlyCLFidelity,
    CachedSOHOReplay,
    CachedSOHOReplayFidelity,
)
from methods.flycl import FlyCL
from methods.sohocl import SOHOCL
from models.flyhash import FlyHash
from models.soho import SOHO


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


def _loader(features, labels):
    return DataLoader(TensorDataset(features, labels), batch_size=len(features), shuffle=False)


def test_fly_fidelity_adapter_matches_original_update_logits_and_gcv():
    generator = torch.Generator().manual_seed(101)
    first = torch.randn(12, 5, generator=generator)
    second = torch.randn(12, 5, generator=generator)
    first_labels = torch.tensor([0, 1] * 6)
    second_labels = torch.tensor([2, 3] * 6)
    query = torch.randn(7, 5, generator=generator)
    seed = 29

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        original = FlyCL(
            torch.nn.Identity(), FlyHash(5, 16, 3), 4, .25, -1, 2,
            torch.device("cpu"),
        )
        original_ridges = [
            float(original.train_task(0, _loader(first, first_labels))[0].item()),
            float(original.train_task(1, _loader(second, second_labels))[0].item()),
        ]
        original_logits = original.flyhash(query, .25, absolute_wta=False) @ original.Wo
        original_projection = original.flyhash.projection_matrix.to_dense()

    cached = CachedFlyCLFidelity(
        5, 16, 3, .25, 4, -1, 2, seed=seed, dtype=torch.float32
    )
    cached.update(first, first_labels)
    cached_ridges = [cached.last_ridge]
    cached.update(second, second_labels)
    cached_ridges.append(cached.last_ridge)

    assert cached_ridges == original_ridges
    assert cached.class_ids == [0, 1, 2, 3]
    assert cached.predict_logits(query).shape == (len(query), 4)
    torch.testing.assert_close(
        cached.flyhash.projection_matrix.to_dense(), original_projection,
        rtol=0, atol=0,
    )
    torch.testing.assert_close(cached.G, original.G_global, rtol=0, atol=1e-6)
    torch.testing.assert_close(cached.Q, original.Q_global, rtol=0, atol=1e-6)
    torch.testing.assert_close(cached.weights, original.Wo, rtol=0, atol=1e-6)
    torch.testing.assert_close(cached.predict_logits(query), original_logits, rtol=0, atol=1e-6)
    assert torch.equal(cached.predict(query), original_logits.argmax(1))
    assert not any("history" in name for name in cached.persistent_tensors())

    clone = CachedFlyCLFidelity(5, 16, 3, .25, 4, -1, 2, seed=seed)
    clone.load_state_dict(cached.state_dict())
    torch.testing.assert_close(clone.predict_logits(query), original_logits, rtol=0, atol=1e-6)


def test_soho_fidelity_adapter_matches_original_replay_logits_and_gcv():
    generator = torch.Generator().manual_seed(131)
    first = torch.randn(12, 5, generator=generator)
    second = torch.randn(12, 5, generator=generator)
    first_labels = torch.tensor([0, 1] * 6)
    second_labels = torch.tensor([2, 3] * 6)
    query = torch.randn(7, 5, generator=generator)
    seed = 37

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        original = SOHOCL(
            torch.nn.Identity(),
            SOHO(5, 16, torch.device("cpu"), density=.4, olda_dim=5, use_etf=True),
            4, .25, -1, 2, torch.device("cpu"),
        )
        original_ridges = [
            float(original.train_task(0, _loader(first, first_labels))[0].item()),
            float(original.train_task(1, _loader(second, second_labels))[0].item()),
        ]
        original_logits = original.soho(query, .25, absolute_wta=False) @ original.Wo
        original_r = original.soho.R.detach().clone()
        original_w = original.soho.W.detach().clone()

    cached = CachedSOHOReplayFidelity(
        5, 16, .4, 5, True, .25, 4, -1, 2,
        seed=seed, replay_chunk_size=2000, gcv_sample_size=3000,
    )
    cached.update(first, first_labels)
    cached_ridges = [cached.last_ridge]
    cached.update(second, second_labels)
    cached_ridges.append(cached.last_ridge)

    assert cached_ridges == original_ridges
    assert cached.class_ids == [0, 1, 2, 3]
    assert cached.is_exemplar_free is False
    assert cached.diagnostics["retained_sample_count"] == 24
    assert sum(len(value) for value in cached.feature_history) == 24
    torch.testing.assert_close(cached.soho.R, original_r, rtol=0, atol=1e-6)
    torch.testing.assert_close(cached.soho.W, original_w, rtol=0, atol=0)
    torch.testing.assert_close(cached.weights, original.Wo, rtol=0, atol=1e-5)
    torch.testing.assert_close(cached.predict_logits(query), original_logits, rtol=0, atol=1e-5)
    assert torch.equal(cached.predict(query), original_logits.argmax(1))

    clone = CachedSOHOReplayFidelity(
        5, 16, .4, 5, True, .25, 4, -1, 2,
        seed=seed, replay_chunk_size=2000, gcv_sample_size=3000,
    )
    clone.load_state_dict(cached.state_dict())
    torch.testing.assert_close(clone.predict_logits(query), original_logits, rtol=0, atol=1e-5)
    assert clone.diagnostics["retained_sample_count"] == 24

    continuation = torch.randn(6, 5, generator=generator)
    continuation_labels = torch.tensor([0, 1, 2, 3, 0, 1])
    cached.update(continuation, continuation_labels)
    clone.update(continuation, continuation_labels)
    assert clone.last_ridge == cached.last_ridge
    torch.testing.assert_close(
        clone.predict_logits(query), cached.predict_logits(query), rtol=0, atol=1e-5
    )
