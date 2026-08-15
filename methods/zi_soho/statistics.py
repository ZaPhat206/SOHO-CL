"""Streaming class-conditional moments for fixed sparse WTA codes."""

from __future__ import annotations

import torch


class ZeroInflatedStatistics:
    """Counts and active-amplitude moments with no retained observation."""

    def __init__(self, feature_dim: int, *, device="cpu", dtype=torch.float32):
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self.class_ids: list[int] = []
        self.counts = torch.zeros(0, device=self.device, dtype=self.dtype)
        self.active_counts = torch.zeros(
            (self.feature_dim, 0), device=self.device, dtype=self.dtype
        )
        self.active_sums = torch.zeros_like(self.active_counts)
        self.active_sq_sums = torch.zeros_like(self.active_counts)

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    @property
    def total_count(self) -> int:
        return int(self.counts.sum().item())

    def _expand_classes(self, labels: torch.Tensor) -> None:
        ordered = sorted(set(self.class_ids) | set(map(int, labels.detach().cpu().tolist())))
        if ordered == self.class_ids:
            return
        old_columns = {class_id: column for column, class_id in enumerate(self.class_ids)}
        counts = torch.zeros(len(ordered), device=self.device, dtype=self.dtype)
        moments = [
            torch.zeros((self.feature_dim, len(ordered)), device=self.device, dtype=self.dtype)
            for _ in range(3)
        ]
        for column, class_id in enumerate(ordered):
            if class_id not in old_columns:
                continue
            old = old_columns[class_id]
            counts[column] = self.counts[old]
            moments[0][:, column] = self.active_counts[:, old]
            moments[1][:, column] = self.active_sums[:, old]
            moments[2][:, column] = self.active_sq_sums[:, old]
        self.class_ids = ordered
        self.counts = counts
        self.active_counts, self.active_sums, self.active_sq_sums = moments

    def update_sparse(
        self, indices: torch.Tensor, values: torch.Tensor, labels: torch.Tensor
    ) -> None:
        index = indices.to(device=self.device, dtype=torch.long)
        amplitude = values.to(device=self.device, dtype=self.dtype)
        target = labels.to(device=self.device, dtype=torch.long)
        if index.ndim != 2 or amplitude.shape != index.shape:
            raise ValueError("indices and values must have the same (B,k) shape")
        if target.ndim != 1 or target.shape[0] != index.shape[0]:
            raise ValueError("labels must have shape (B,) aligned with sparse codes")
        if index.numel() and bool(((index < 0) | (index >= self.feature_dim)).any()):
            raise ValueError("sparse code index out of range")
        if not bool(torch.isfinite(amplitude).all()):
            raise ValueError("sparse code values contain NaN or Inf")
        if index.shape[0] == 0:
            return
        self._expand_classes(target)
        class_to_column = {class_id: column for column, class_id in enumerate(self.class_ids)}
        for class_id in sorted(set(map(int, target.detach().cpu().tolist()))):
            column = class_to_column[class_id]
            mask = target == class_id
            selected = index[mask].reshape(-1)
            selected_values = amplitude[mask].reshape(-1)
            ones = torch.ones_like(selected_values)
            self.active_counts[:, column].scatter_add_(0, selected, ones)
            self.active_sums[:, column].scatter_add_(0, selected, selected_values)
            self.active_sq_sums[:, column].scatter_add_(
                0, selected, selected_values.square()
            )
            self.counts[column] += int(mask.sum().item())

    def state_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "class_ids": list(self.class_ids),
            "counts": self.counts.detach().cpu().clone(),
            "active_counts": self.active_counts.detach().cpu().clone(),
            "active_sums": self.active_sums.detach().cpu().clone(),
            "active_sq_sums": self.active_sq_sums.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("statistics feature dimension mismatch")
        class_ids = [int(value) for value in state["class_ids"]]
        if class_ids != sorted(set(class_ids)):
            raise ValueError("class_ids must be sorted and unique")
        count = len(class_ids)
        counts = state["counts"].to(self.device, self.dtype)
        moments = [
            state[name].to(self.device, self.dtype)
            for name in ("active_counts", "active_sums", "active_sq_sums")
        ]
        if counts.shape != (count,) or any(
            value.shape != (self.feature_dim, count) for value in moments
        ):
            raise ValueError("invalid zero-inflated statistics shape")
        if not all(bool(torch.isfinite(value).all()) for value in [counts, *moments]):
            raise ValueError("statistics contain NaN or Inf")
        if bool((counts < 0).any() or (moments[0] < 0).any()):
            raise ValueError("counts must be non-negative")
        self.class_ids, self.counts = class_ids, counts
        self.active_counts, self.active_sums, self.active_sq_sums = moments
