"""Streaming class-protected WTA statistics."""

from __future__ import annotations

import torch

from .sketch import FrequentDirections


class ClassProtectedStatistics:
    """Exact class means/counts plus a sketch of within-class scatter."""

    def __init__(
        self,
        feature_dim: int,
        sketch_size: int,
        *,
        mode: str = "class_protected",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if mode not in {"class_protected", "global"}:
            raise ValueError("mode must be 'class_protected' or 'global'")
        self.feature_dim = int(feature_dim)
        self.sketch_size = int(sketch_size)
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = dtype
        self.means = torch.zeros((feature_dim, 0), device=self.device, dtype=dtype)
        self.counts = torch.zeros(0, device=self.device, dtype=dtype)
        self.class_ids: list[int] = []
        self.sketch = FrequentDirections(
            feature_dim, sketch_size, device=self.device, dtype=self.dtype
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    @property
    def total_count(self) -> int:
        return int(self.counts.sum().item())

    @property
    def cross(self) -> torch.Tensor:
        """Exact ``Phi.T @ Y`` reconstructed from class means and counts."""
        return self.means * self.counts[None, :]

    def between_factor(self) -> torch.Tensor:
        """Rows A with ``A.T @ A = sum_c n_c m_c m_c.T``."""
        return self.counts.sqrt()[:, None] * self.means.T

    def _expand_classes(self, labels: torch.Tensor) -> None:
        ordered = sorted(set(self.class_ids) | set(map(int, labels.detach().cpu().tolist())))
        if ordered == self.class_ids:
            return
        old_columns = {class_id: column for column, class_id in enumerate(self.class_ids)}
        means = torch.zeros((self.feature_dim, len(ordered)), device=self.device, dtype=self.dtype)
        counts = torch.zeros(len(ordered), device=self.device, dtype=self.dtype)
        for column, class_id in enumerate(ordered):
            if class_id in old_columns:
                old = old_columns[class_id]
                means[:, column] = self.means[:, old]
                counts[column] = self.counts[old]
        self.class_ids, self.means, self.counts = ordered, means, counts

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values = features.to(device=self.device, dtype=self.dtype)
        targets = labels.to(device=self.device, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (N, {self.feature_dim})")
        if targets.ndim != 1 or targets.shape[0] != values.shape[0]:
            raise ValueError("labels must have shape (N,) aligned with features")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        if values.shape[0] == 0:
            return
        self._expand_classes(targets)
        if self.mode == "global":
            self.sketch.update(values)

        class_to_column = {class_id: column for column, class_id in enumerate(self.class_ids)}
        for class_id in sorted(set(map(int, targets.detach().cpu().tolist()))):
            column = class_to_column[class_id]
            batch = values[targets == class_id]
            batch_count = int(batch.shape[0])
            batch_mean = batch.mean(dim=0)
            old_count = float(self.counts[column].item())
            old_mean = self.means[:, column].clone()
            combined_count = old_count + batch_count

            if self.mode == "class_protected":
                centered = batch - batch_mean
                if old_count > 0:
                    correction = torch.sqrt(torch.tensor(
                        old_count * batch_count / combined_count,
                        device=self.device,
                        dtype=self.dtype,
                    )) * (batch_mean - old_mean)
                    centered = torch.cat((centered, correction[None, :]), dim=0)
                self.sketch.update(centered)

            if old_count == 0:
                merged_mean = batch_mean
            else:
                merged_mean = old_mean + (batch_count / combined_count) * (batch_mean - old_mean)
            self.means[:, column] = merged_mean
            self.counts[column] = combined_count

    def state_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "sketch_size": self.sketch_size,
            "mode": self.mode,
            "means": self.means.clone(),
            "counts": self.counts.clone(),
            "class_ids": list(self.class_ids),
            "sketch": self.sketch.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("statistics feature dimension mismatch")
        if int(state["sketch_size"]) != self.sketch_size:
            raise ValueError("statistics sketch size mismatch")
        if state["mode"] != self.mode:
            raise ValueError("statistics mode mismatch")
        self.class_ids = [int(value) for value in state["class_ids"]]
        if self.class_ids != sorted(set(self.class_ids)):
            raise ValueError("class_ids must be sorted and unique")
        self.means = state["means"].to(device=self.device, dtype=self.dtype)
        self.counts = state["counts"].to(device=self.device, dtype=self.dtype)
        if self.means.shape != (self.feature_dim, self.num_classes):
            raise ValueError("invalid means shape")
        if self.counts.shape != (self.num_classes,):
            raise ValueError("invalid counts shape")
        if not bool(torch.isfinite(self.means).all() and torch.isfinite(self.counts).all()):
            raise ValueError("statistics contain NaN or Inf")
        if bool((self.counts < 0).any()):
            raise ValueError("counts must be non-negative")
        self.sketch.load_state_dict(state["sketch"])
