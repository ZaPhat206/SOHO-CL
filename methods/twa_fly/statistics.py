"""Fixed-width streaming sufficient statistics for paired raw/WTA views."""

from __future__ import annotations

import torch


class TWAStatistics:
    """Aggregate paired-view moments without retaining sample-level rows."""

    def __init__(self, raw_dim: int, fly_dim: int, num_classes: int, *, device="cpu", dtype=torch.float32):
        if min(raw_dim, fly_dim, num_classes) <= 0:
            raise ValueError("raw_dim, fly_dim, and num_classes must be positive")
        self.raw_dim = int(raw_dim)
        self.fly_dim = int(fly_dim)
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.dtype = dtype
        self.G_xx = torch.zeros((self.raw_dim, self.raw_dim), device=self.device, dtype=dtype)
        self.G_zz = torch.zeros((self.fly_dim, self.fly_dim), device=self.device, dtype=dtype)
        self.R_xz = torch.zeros((self.raw_dim, self.fly_dim), device=self.device, dtype=dtype)
        self.Q_x = torch.zeros((self.raw_dim, self.num_classes), device=self.device, dtype=dtype)
        self.Q_z = torch.zeros((self.fly_dim, self.num_classes), device=self.device, dtype=dtype)
        self.counts = torch.zeros(self.num_classes, device=self.device, dtype=dtype)

    @property
    def total_count(self) -> int:
        return int(self.counts.sum().item())

    @property
    def class_ids(self) -> list[int]:
        return list(range(self.num_classes))

    def update(
        self,
        raw_features: torch.Tensor,
        fly_features: torch.Tensor,
        labels: torch.Tensor,
        *,
        cross_fly_features: torch.Tensor | None = None,
    ) -> None:
        x = raw_features.to(device=self.device, dtype=self.dtype)
        z = fly_features.to(device=self.device, dtype=self.dtype)
        y = labels.to(device=self.device, dtype=torch.long)
        if x.ndim != 2 or x.shape[1] != self.raw_dim:
            raise ValueError(f"raw_features must have shape (B, {self.raw_dim})")
        if z.ndim != 2 or z.shape != (x.shape[0], self.fly_dim):
            raise ValueError(f"fly_features must have shape (B, {self.fly_dim})")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("labels must have shape (B,) aligned with both views")
        if y.numel() == 0:
            return
        if bool(((y < 0) | (y >= self.num_classes)).any()):
            raise ValueError("labels must be global class IDs in [0, num_classes)")
        if not bool(torch.isfinite(x).all() and torch.isfinite(z).all()):
            raise ValueError("features contain NaN or Inf")
        cross_z = z if cross_fly_features is None else cross_fly_features.to(
            device=self.device, dtype=self.dtype
        )
        if cross_z.shape != z.shape or not bool(torch.isfinite(cross_z).all()):
            raise ValueError("cross_fly_features must be finite and match fly_features")
        targets = torch.nn.functional.one_hot(y, self.num_classes).to(self.dtype)
        self.G_xx.add_(x.T @ x)
        self.G_zz.add_(z.T @ z)
        self.R_xz.add_(x.T @ cross_z)
        self.Q_x.add_(x.T @ targets)
        self.Q_z.add_(z.T @ targets)
        self.counts.add_(torch.bincount(y, minlength=self.num_classes).to(self.dtype))

    def state_dict(self) -> dict:
        return {
            "raw_dim": self.raw_dim,
            "fly_dim": self.fly_dim,
            "num_classes": self.num_classes,
            **{name: getattr(self, name).detach().cpu().clone() for name in (
                "G_xx", "G_zz", "R_xz", "Q_x", "Q_z", "counts"
            )},
        }

    def load_state_dict(self, state: dict) -> None:
        observed = (int(state["raw_dim"]), int(state["fly_dim"]), int(state["num_classes"]))
        expected = (self.raw_dim, self.fly_dim, self.num_classes)
        if observed != expected:
            raise ValueError("TWA statistic dimension mismatch")
        shapes = {
            "G_xx": (self.raw_dim, self.raw_dim),
            "G_zz": (self.fly_dim, self.fly_dim),
            "R_xz": (self.raw_dim, self.fly_dim),
            "Q_x": (self.raw_dim, self.num_classes),
            "Q_z": (self.fly_dim, self.num_classes),
            "counts": (self.num_classes,),
        }
        for name, shape in shapes.items():
            value = state[name].to(device=self.device, dtype=self.dtype)
            if value.shape != shape:
                raise ValueError(f"invalid {name} shape")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or Inf")
            setattr(self, name, value)
        if bool((self.counts < 0).any()):
            raise ValueError("counts must be non-negative")
