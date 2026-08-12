"""Streaming statistics for a fixed frozen-feature representation.

This module deliberately stores class aggregates only.  It never stores a
batch, a sample index, a historical feature matrix, or a replay buffer.
"""

from __future__ import annotations

import torch


class FixedFeatureStatistics:
    """Exact `G, Q, counts` statistics in a fixed D-dimensional space."""

    def __init__(self, feature_dim: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64):
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self.G = torch.zeros((feature_dim, feature_dim), device=self.device, dtype=dtype)
        self.Q = torch.zeros((feature_dim, 0), device=self.device, dtype=dtype)
        self.counts = torch.zeros(0, device=self.device, dtype=dtype)
        self.class_ids: list[int] = []

    def _expand_classes(self, labels: torch.Tensor) -> None:
        """Insert new IDs while keeping every statistic column canonically sorted."""
        ordered_ids = sorted(set(self.class_ids) | set(map(int, labels.detach().cpu().tolist())))
        if ordered_ids == self.class_ids:
            return
        old_columns = {class_id: column for column, class_id in enumerate(self.class_ids)}
        expanded_q = torch.zeros((self.feature_dim, len(ordered_ids)), device=self.device, dtype=self.dtype)
        expanded_counts = torch.zeros(len(ordered_ids), device=self.device, dtype=self.dtype)
        for new_column, class_id in enumerate(ordered_ids):
            if class_id in old_columns:
                old_column = old_columns[class_id]
                expanded_q[:, new_column] = self.Q[:, old_column]
                expanded_counts[new_column] = self.counts[old_column]
        self.class_ids = ordered_ids
        self.Q = expanded_q
        self.counts = expanded_counts

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate one streaming batch without retaining it."""
        x = features.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device, dtype=torch.long)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        if labels.ndim != 1 or labels.shape[0] != x.shape[0]:
            raise ValueError("labels must have shape (B,) aligned with features")
        if labels.numel() == 0:
            return
        if not bool(torch.isfinite(x).all()):
            raise ValueError("features contain NaN or Inf")

        self._expand_classes(labels)
        class_to_column = {class_id: column for column, class_id in enumerate(self.class_ids)}
        columns = torch.tensor([class_to_column[int(label)] for label in labels.detach().cpu().tolist()], device=self.device)
        targets = torch.nn.functional.one_hot(columns, num_classes=len(self.class_ids)).to(dtype=self.dtype)
        self.G.add_(x.T @ x)
        self.Q.add_(x.T @ targets)
        self.counts.scatter_add_(0, columns, torch.ones_like(columns, dtype=self.dtype))

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    @property
    def total_count(self) -> int:
        return int(self.counts.sum().item())

    def means(self) -> torch.Tensor:
        if not self.class_ids:
            return torch.empty((self.feature_dim, 0), device=self.device, dtype=self.dtype)
        return self.Q / self.counts.clamp_min(1).unsqueeze(0)

    def state_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "G": self.G.clone(),
            "Q": self.Q.clone(),
            "counts": self.counts.clone(),
            "class_ids": list(self.class_ids),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("feature dimension mismatch")
        self.G = state["G"].to(device=self.device, dtype=self.dtype)
        self.Q = state["Q"].to(device=self.device, dtype=self.dtype)
        self.counts = state["counts"].to(device=self.device, dtype=self.dtype)
        self.class_ids = [int(class_id) for class_id in state["class_ids"]]
        if self.G.shape != (self.feature_dim, self.feature_dim):
            raise ValueError("invalid G shape")
        if self.Q.shape != (self.feature_dim, len(self.class_ids)) or self.counts.shape != (len(self.class_ids),):
            raise ValueError("invalid class statistic shapes")
