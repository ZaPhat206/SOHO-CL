"""Exact tensor-payload accounting for analytic learner state."""

from __future__ import annotations

import torch


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Return bytes owned by a supported dense or sparse tensor payload."""
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return (
            tensor.values().numel() * tensor.values().element_size()
            + tensor.ccol_indices().numel() * tensor.ccol_indices().element_size()
            + tensor.row_indices().numel() * tensor.row_indices().element_size()
        )
    raise ValueError(f"unsupported tensor layout: {tensor.layout}")


def persistent_tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(tensor_bytes(tensor) for tensor in tensors.values())
