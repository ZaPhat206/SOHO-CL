"""Streaming spherical class moments with no retained observation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MomentSnapshot:
    """Immutable tensor snapshot used transiently during one learner update."""

    feature_dim: int
    class_ids: tuple[int, ...]
    counts: torch.Tensor
    sums: torch.Tensor
    squared_sums: torch.Tensor
    within_scatter: torch.Tensor
    global_sum: torch.Tensor
    total_count: int

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    @property
    def means(self) -> torch.Tensor:
        if not self.class_ids:
            return self.sums
        return self.sums / self.counts.clamp_min(1).unsqueeze(0)

    @property
    def variances(self) -> torch.Tensor:
        if not self.class_ids:
            return self.squared_sums
        values = self.squared_sums / self.counts.clamp_min(1).unsqueeze(0)
        return (values - self.means.square()).clamp_min(0)

    @property
    def global_mean(self) -> torch.Tensor:
        if self.total_count <= 0:
            return torch.zeros_like(self.global_sum)
        return self.global_sum / self.total_count

    def class_column(self, class_id: int) -> int:
        try:
            return self.class_ids.index(int(class_id))
        except ValueError as error:
            raise KeyError(f"unknown class ID {class_id}") from error

    def pooled_covariance(self) -> torch.Tensor:
        degrees = max(self.total_count - self.num_classes, 1)
        return self.within_scatter / degrees


class SphericalClassMoments:
    """Exact streaming moments of row-normalized frozen features.

    Persistent tensor shapes depend only on feature dimension and class count.
    No batch, sample index, historical feature, or per-sample label is retained.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self.class_ids: list[int] = []
        self.counts = torch.zeros(0, device=self.device, dtype=self.dtype)
        self.sums = torch.zeros(
            (self.feature_dim, 0), device=self.device, dtype=self.dtype
        )
        self.squared_sums = torch.zeros_like(self.sums)
        self.within_scatter = torch.zeros(
            (self.feature_dim, self.feature_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.global_sum = torch.zeros(
            self.feature_dim, device=self.device, dtype=self.dtype
        )
        self.total_count = 0

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    def _expand_classes(self, labels: torch.Tensor) -> None:
        ordered = sorted(
            set(self.class_ids) | set(map(int, labels.detach().cpu().tolist()))
        )
        if ordered == self.class_ids:
            return
        old_columns = {value: index for index, value in enumerate(self.class_ids)}
        counts = torch.zeros(len(ordered), device=self.device, dtype=self.dtype)
        sums = torch.zeros(
            (self.feature_dim, len(ordered)), device=self.device, dtype=self.dtype
        )
        squared = torch.zeros_like(sums)
        for column, class_id in enumerate(ordered):
            if class_id not in old_columns:
                continue
            old = old_columns[class_id]
            counts[column] = self.counts[old]
            sums[:, column] = self.sums[:, old]
            squared[:, column] = self.squared_sums[:, old]
        self.class_ids = ordered
        self.counts = counts
        self.sums = sums
        self.squared_sums = squared

    def snapshot(self) -> MomentSnapshot:
        return MomentSnapshot(
            feature_dim=self.feature_dim,
            class_ids=tuple(self.class_ids),
            counts=self.counts.clone(),
            sums=self.sums.clone(),
            squared_sums=self.squared_sums.clone(),
            within_scatter=self.within_scatter.clone(),
            global_sum=self.global_sum.clone(),
            total_count=self.total_count,
        )

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
        values = torch.nn.functional.normalize(values, p=2, dim=1)
        self._expand_classes(targets)
        columns = {value: index for index, value in enumerate(self.class_ids)}
        for class_id in sorted(set(map(int, targets.detach().cpu().tolist()))):
            column = columns[class_id]
            batch = values[targets == class_id]
            batch_count = int(batch.shape[0])
            batch_sum = batch.sum(dim=0)
            batch_mean = batch_sum / batch_count
            centered = batch - batch_mean
            batch_scatter = centered.T @ centered
            old_count = float(self.counts[column].item())
            if old_count:
                old_mean = self.sums[:, column] / old_count
                correction_scale = old_count * batch_count / (old_count + batch_count)
                delta = batch_mean - old_mean
                batch_scatter = batch_scatter + correction_scale * torch.outer(delta, delta)
            self.within_scatter.add_(batch_scatter)
            self.counts[column] += batch_count
            self.sums[:, column].add_(batch_sum)
            self.squared_sums[:, column].add_(batch.square().sum(dim=0))
        self.global_sum.add_(values.sum(dim=0))
        self.total_count += int(values.shape[0])

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "class_counts": self.counts,
            "class_sums": self.sums,
            "class_squared_sums": self.squared_sums,
            "pooled_within_scatter": self.within_scatter,
            "global_sum": self.global_sum,
        }

    def state_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "class_ids": list(self.class_ids),
            "counts": self.counts.detach().cpu().clone(),
            "sums": self.sums.detach().cpu().clone(),
            "squared_sums": self.squared_sums.detach().cpu().clone(),
            "within_scatter": self.within_scatter.detach().cpu().clone(),
            "global_sum": self.global_sum.detach().cpu().clone(),
            "total_count": self.total_count,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("moment feature dimension mismatch")
        class_ids = [int(value) for value in state["class_ids"]]
        if class_ids != sorted(set(class_ids)):
            raise ValueError("class_ids must be sorted and unique")
        count = len(class_ids)
        tensors = {
            "counts": state["counts"].to(self.device, self.dtype),
            "sums": state["sums"].to(self.device, self.dtype),
            "squared_sums": state["squared_sums"].to(self.device, self.dtype),
            "within_scatter": state["within_scatter"].to(self.device, self.dtype),
            "global_sum": state["global_sum"].to(self.device, self.dtype),
        }
        expected = {
            "counts": (count,),
            "sums": (self.feature_dim, count),
            "squared_sums": (self.feature_dim, count),
            "within_scatter": (self.feature_dim, self.feature_dim),
            "global_sum": (self.feature_dim,),
        }
        for name, value in tensors.items():
            if value.shape != expected[name] or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid {name} in moment checkpoint")
        total_count = int(state["total_count"])
        if total_count < 0 or bool((tensors["counts"] < 0).any()):
            raise ValueError("moment counts must be non-negative")
        if abs(float(tensors["counts"].sum().item()) - total_count) > 1e-6:
            raise ValueError("moment total_count does not match class counts")
        self.class_ids = class_ids
        self.total_count = total_count
        for name, value in tensors.items():
            setattr(self, name, value)
