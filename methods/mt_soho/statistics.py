"""Minimal exact sufficient statistics for fixed-anchor MT-SOHO."""

from __future__ import annotations

import torch


class MomentTransportStatistics:
    def __init__(self, raw_dim: int, anchor_dim: int, device="cpu", dtype=torch.float32):
        if raw_dim <= 0 or anchor_dim <= 0:
            raise ValueError("raw_dim and anchor_dim must be positive")
        self.raw_dim, self.anchor_dim = int(raw_dim), int(anchor_dim)
        self.device, self.dtype = torch.device(device), dtype
        self.G_u = torch.zeros((anchor_dim, anchor_dim), device=self.device, dtype=dtype)
        self.G_x = torch.zeros((raw_dim, raw_dim), device=self.device, dtype=dtype)
        self.Q_u = torch.zeros((anchor_dim, 0), device=self.device, dtype=dtype)
        self.Q_x = torch.zeros((raw_dim, 0), device=self.device, dtype=dtype)
        self.counts = torch.zeros(0, device=self.device, dtype=dtype)
        self.class_ids: list[int] = []

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    @property
    def total_count(self) -> int:
        return int(self.counts.sum().item())

    def _expand(self, labels: torch.Tensor) -> None:
        ordered = sorted(set(self.class_ids) | set(map(int, labels.detach().cpu().tolist())))
        if ordered == self.class_ids:
            return
        previous = {class_id: column for column, class_id in enumerate(self.class_ids)}
        q_u = torch.zeros((self.anchor_dim, len(ordered)), device=self.device, dtype=self.dtype)
        q_x = torch.zeros((self.raw_dim, len(ordered)), device=self.device, dtype=self.dtype)
        counts = torch.zeros(len(ordered), device=self.device, dtype=self.dtype)
        for column, class_id in enumerate(ordered):
            if class_id in previous:
                old = previous[class_id]
                q_u[:, column], q_x[:, column], counts[column] = self.Q_u[:, old], self.Q_x[:, old], self.counts[old]
        self.class_ids, self.Q_u, self.Q_x, self.counts = ordered, q_u, q_x, counts

    def update(self, raw: torch.Tensor, anchor: torch.Tensor, labels: torch.Tensor) -> None:
        x = raw.to(self.device, self.dtype)
        u = anchor.to(self.device, self.dtype)
        y = labels.to(self.device, torch.long)
        if x.ndim != 2 or x.shape[1] != self.raw_dim:
            raise ValueError(f"raw must have shape (B, {self.raw_dim})")
        if u.ndim != 2 or u.shape != (x.shape[0], self.anchor_dim):
            raise ValueError(f"anchor must have shape (B, {self.anchor_dim})")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("labels must align with features")
        if y.numel() == 0:
            return
        if not bool(torch.isfinite(x).all() and torch.isfinite(u).all()):
            raise ValueError("features contain NaN or Inf")
        self._expand(y)
        mapping = {class_id: column for column, class_id in enumerate(self.class_ids)}
        columns = torch.tensor([mapping[int(value)] for value in y.detach().cpu().tolist()], device=self.device)
        targets = torch.nn.functional.one_hot(columns, num_classes=self.num_classes).to(self.dtype)
        self.G_u.add_(u.T @ u)
        self.G_x.add_(x.T @ x)
        self.Q_u.add_(u.T @ targets)
        self.Q_x.add_(x.T @ targets)
        self.counts.scatter_add_(0, columns, torch.ones_like(columns, dtype=self.dtype))

    def state_dict(self) -> dict:
        return {
            "raw_dim": self.raw_dim,
            "anchor_dim": self.anchor_dim,
            "G_u": self.G_u.clone(),
            "G_x": self.G_x.clone(),
            "Q_u": self.Q_u.clone(),
            "Q_x": self.Q_x.clone(),
            "counts": self.counts.clone(),
            "class_ids": list(self.class_ids),
        }

    def load_state_dict(self, state: dict) -> None:
        if (int(state["raw_dim"]), int(state["anchor_dim"])) != (self.raw_dim, self.anchor_dim):
            raise ValueError("moment-statistic dimension mismatch")
        for name in ("G_u", "G_x", "Q_u", "Q_x", "counts"):
            setattr(self, name, state[name].to(self.device, self.dtype))
        self.class_ids = [int(value) for value in state["class_ids"]]
        expected = {
            "G_u": (self.anchor_dim, self.anchor_dim),
            "G_x": (self.raw_dim, self.raw_dim),
            "Q_u": (self.anchor_dim, self.num_classes),
            "Q_x": (self.raw_dim, self.num_classes),
            "counts": (self.num_classes,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid {name}")
