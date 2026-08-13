"""Exact dual-view streaming sufficient statistics for CRT-SOHO."""

from __future__ import annotations

import torch


class DualViewStatistics:
    """Statistics for fixed raw features X and fixed nonlinear anchor Phi."""

    def __init__(self, raw_dim: int, anchor_dim: int, device="cpu", dtype=torch.float32):
        if raw_dim <= 0 or anchor_dim <= 0:
            raise ValueError("raw_dim and anchor_dim must be positive")
        self.raw_dim, self.anchor_dim = int(raw_dim), int(anchor_dim)
        self.device, self.dtype = torch.device(device), dtype
        self.G_pp = torch.zeros((anchor_dim, anchor_dim), device=self.device, dtype=dtype)
        self.G_xx = torch.zeros((raw_dim, raw_dim), device=self.device, dtype=dtype)
        self.H_px = torch.zeros((anchor_dim, raw_dim), device=self.device, dtype=dtype)
        self.Q_p = torch.zeros((anchor_dim, 0), device=self.device, dtype=dtype)
        self.Q_x = torch.zeros((raw_dim, 0), device=self.device, dtype=dtype)
        self.counts = torch.zeros(0, device=self.device, dtype=dtype)
        self.class_ids: list[int] = []

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    @property
    def total_count(self) -> int:
        return int(self.counts.sum().item())

    def _expand_classes(self, labels: torch.Tensor) -> None:
        ordered_ids = sorted(set(self.class_ids) | set(map(int, labels.detach().cpu().tolist())))
        if ordered_ids == self.class_ids:
            return
        old_columns = {class_id: column for column, class_id in enumerate(self.class_ids)}
        q_p = torch.zeros((self.anchor_dim, len(ordered_ids)), device=self.device, dtype=self.dtype)
        q_x = torch.zeros((self.raw_dim, len(ordered_ids)), device=self.device, dtype=self.dtype)
        counts = torch.zeros(len(ordered_ids), device=self.device, dtype=self.dtype)
        for column, class_id in enumerate(ordered_ids):
            if class_id in old_columns:
                old = old_columns[class_id]
                q_p[:, column], q_x[:, column], counts[column] = self.Q_p[:, old], self.Q_x[:, old], self.counts[old]
        self.class_ids, self.Q_p, self.Q_x, self.counts = ordered_ids, q_p, q_x, counts

    def update(self, raw_features: torch.Tensor, anchor_features: torch.Tensor, labels: torch.Tensor) -> None:
        x = raw_features.to(device=self.device, dtype=self.dtype)
        phi = anchor_features.to(device=self.device, dtype=self.dtype)
        y = labels.to(device=self.device, dtype=torch.long)
        if x.ndim != 2 or x.shape[1] != self.raw_dim:
            raise ValueError(f"raw_features must have shape (B, {self.raw_dim})")
        if phi.ndim != 2 or phi.shape != (x.shape[0], self.anchor_dim):
            raise ValueError(f"anchor_features must have shape (B, {self.anchor_dim})")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("labels must have shape (B,) aligned with both views")
        if y.numel() == 0:
            return
        if not bool(torch.isfinite(x).all() and torch.isfinite(phi).all()):
            raise ValueError("features contain NaN or Inf")

        self._expand_classes(y)
        class_to_column = {class_id: column for column, class_id in enumerate(self.class_ids)}
        columns = torch.tensor([class_to_column[int(label)] for label in y.detach().cpu().tolist()], device=self.device)
        targets = torch.nn.functional.one_hot(columns, num_classes=self.num_classes).to(self.dtype)
        self.G_pp.add_(phi.T @ phi)
        self.G_xx.add_(x.T @ x)
        self.H_px.add_(phi.T @ x)
        self.Q_p.add_(phi.T @ targets)
        self.Q_x.add_(x.T @ targets)
        self.counts.scatter_add_(0, columns, torch.ones_like(columns, dtype=self.dtype))

    def state_dict(self) -> dict:
        return {
            "raw_dim": self.raw_dim,
            "anchor_dim": self.anchor_dim,
            "G_pp": self.G_pp.clone(),
            "G_xx": self.G_xx.clone(),
            "H_px": self.H_px.clone(),
            "Q_p": self.Q_p.clone(),
            "Q_x": self.Q_x.clone(),
            "counts": self.counts.clone(),
            "class_ids": list(self.class_ids),
        }

    def load_state_dict(self, state: dict) -> None:
        if (int(state["raw_dim"]), int(state["anchor_dim"])) != (self.raw_dim, self.anchor_dim):
            raise ValueError("dual-view feature dimension mismatch")
        for name in ("G_pp", "G_xx", "H_px", "Q_p", "Q_x", "counts"):
            setattr(self, name, state[name].to(device=self.device, dtype=self.dtype))
        self.class_ids = [int(class_id) for class_id in state["class_ids"]]
        if self.class_ids != sorted(set(self.class_ids)):
            raise ValueError("class_ids must be sorted and unique")
        expected = {
            "G_pp": (self.anchor_dim, self.anchor_dim), "G_xx": (self.raw_dim, self.raw_dim),
            "H_px": (self.anchor_dim, self.raw_dim), "Q_p": (self.anchor_dim, self.num_classes),
            "Q_x": (self.raw_dim, self.num_classes), "counts": (self.num_classes,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"invalid {name} shape")
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if bool((self.counts < 0).any()):
            raise ValueError("counts must be non-negative")
