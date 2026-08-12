"""Frozen backbone construction and verified local checkpoint loading."""

from hashlib import sha256
from pathlib import Path

import timm
import torch
import torch.nn as nn
from safetensors import safe_open


def verify_safetensors_checkpoint(checkpoint_path, expected_size=None, expected_sha256=None):
    """Validate a local safetensors file before any model is constructed/loaded."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Backbone checkpoint does not exist: {path}")

    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Checkpoint size mismatch: expected {expected_size}, got {size} bytes.")

    digest = sha256()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 mismatch.")

    finite = True
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                finite = False
                raise ValueError(f"Checkpoint contains NaN or Inf values: {key}")

    return {"path": str(path.resolve()), "size": size, "sha256": actual_sha256, "tensor_count": len(keys), "finite": finite}


def _load_verified_state_dict(checkpoint_path):
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def load_model(
    model_name: str,
    pretrained: bool = True,
    checkpoint_path=None,
    expected_checkpoint_size=None,
    expected_checkpoint_sha256=None,
) -> nn.Module:
    """Load a timm backbone, optionally from a verified local safetensors artifact.

    Local checkpoints never trigger a network lookup and must match the complete
    pre-classifier architecture exactly.  The classifier is removed only after
    successful strict loading to preserve the full checkpoint compatibility gate.
    """
    if checkpoint_path is None:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        model.checkpoint_verification = None
        model.checkpoint_load_info = {"missing_keys": [], "unexpected_keys": []}
        return model

    verification = verify_safetensors_checkpoint(
        checkpoint_path,
        expected_size=expected_checkpoint_size,
        expected_sha256=expected_checkpoint_sha256,
    )
    model = timm.create_model(model_name, pretrained=False)
    state_dict = _load_verified_state_dict(checkpoint_path)
    # strict=True intentionally raises for every backbone or classifier mismatch.
    model.load_state_dict(state_dict, strict=True)
    model.reset_classifier(0)
    model.checkpoint_verification = verification
    model.checkpoint_load_info = {"missing_keys": [], "unexpected_keys": []}
    return model
