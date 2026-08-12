from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import models.backbone as backbone_module


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 768)
        self.reset_called = False

    def reset_classifier(self, num_classes):
        assert num_classes == 0
        self.reset_called = True

    def forward(self, inputs):
        return self.projection(inputs)


def _checkpoint(path: Path, state_dict=None):
    if state_dict is None:
        state_dict = TinyBackbone().state_dict()
    save_file(state_dict, str(path))
    return path


def test_missing_checkpoint_path_fails_clearly():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        backbone_module.verify_safetensors_checkpoint("does-not-exist.safetensors")


def test_checkpoint_hash_mismatch_fails(tmp_path):
    checkpoint = _checkpoint(tmp_path / "tiny.safetensors")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        backbone_module.verify_safetensors_checkpoint(checkpoint, expected_sha256="0" * 64)


def test_local_checkpoint_never_requests_pretrained_network(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path / "tiny.safetensors")
    calls = []

    def create_model(name, pretrained):
        calls.append((name, pretrained))
        return TinyBackbone()

    monkeypatch.setattr(backbone_module.timm, "create_model", create_model)
    model = backbone_module.load_model("tiny", checkpoint_path=checkpoint)

    assert calls == [("tiny", False)]
    assert model.reset_called
    assert model.checkpoint_load_info == {"missing_keys": [], "unexpected_keys": []}


def test_backbone_mismatch_fails_strictly(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path / "wrong.safetensors", {"wrong.weight": torch.ones(1)})
    monkeypatch.setattr(backbone_module.timm, "create_model", lambda name, pretrained: TinyBackbone())
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        backbone_module.load_model("tiny", checkpoint_path=checkpoint)


def test_local_backbone_features_are_768d_and_finite(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path / "tiny.safetensors")
    monkeypatch.setattr(backbone_module.timm, "create_model", lambda name, pretrained: TinyBackbone())
    model = backbone_module.load_model("tiny", checkpoint_path=checkpoint)
    features = model(torch.randn(3, 4))
    assert features.shape == (3, 768)
    assert bool(torch.isfinite(features).all())
