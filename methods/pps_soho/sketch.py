"""Deterministic Frequent-Directions covariance sketch."""

from __future__ import annotations

import torch


class FrequentDirections:
    """Maintain an ``ell x d`` deterministic covariance sketch.

    The implementation accepts rows in blocks and never retains an input row
    after ``update`` returns.  ``covariance_error_bound`` is the accumulated
    shrinkage and certifies ``0 <= A.T A - B.T B <= bound * I`` up to floating
    point error.
    """

    def __init__(
        self,
        feature_dim: int,
        sketch_size: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if sketch_size <= 0:
            raise ValueError("sketch_size must be positive")
        if sketch_size > feature_dim:
            raise ValueError("sketch_size cannot exceed feature_dim")
        self.feature_dim = int(feature_dim)
        self.sketch_size = int(sketch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.B = torch.zeros(
            (self.sketch_size, self.feature_dim), device=self.device, dtype=self.dtype
        )
        self.occupied_rows = 0
        self.total_rows = 0
        self.covariance_error_bound = 0.0

    def _compress(self, rows: torch.Tensor) -> None:
        if rows.shape[0] <= self.sketch_size:
            self.B.zero_()
            self.B[: rows.shape[0]].copy_(rows)
            self.occupied_rows = int(rows.shape[0])
            return
        _, singular_values, right = torch.linalg.svd(rows, full_matrices=False)
        if self.sketch_size == self.feature_dim:
            self.B.copy_(singular_values[: self.sketch_size, None] * right[: self.sketch_size])
            self.occupied_rows = self.sketch_size
            return
        delta = singular_values[self.sketch_size - 1].square()
        shrunk = torch.sqrt(
            torch.clamp(singular_values[: self.sketch_size].square() - delta, min=0)
        )
        self.B.copy_(shrunk[:, None] * right[: self.sketch_size])
        self.occupied_rows = self.sketch_size
        self.covariance_error_bound += float(delta.item())

    def update(self, rows: torch.Tensor) -> None:
        values = rows.to(device=self.device, dtype=self.dtype)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"rows must have shape (N, {self.feature_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("sketch rows contain NaN or Inf")
        if values.shape[0] == 0:
            return
        self.total_rows += int(values.shape[0])
        cursor = 0
        if self.occupied_rows < self.sketch_size:
            available = self.sketch_size - self.occupied_rows
            take = min(available, values.shape[0])
            self.B[self.occupied_rows : self.occupied_rows + take].copy_(values[:take])
            self.occupied_rows += take
            cursor += take
        # Compress at most 2*ell rows at once. This avoids shrinking the first
        # ell rows merely because the persistent matrix is zero padded.
        while cursor < values.shape[0]:
            block = values[cursor : cursor + self.sketch_size]
            self._compress(torch.cat((self.B[: self.occupied_rows], block), dim=0))
            cursor += int(block.shape[0])

    def state_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "sketch_size": self.sketch_size,
            "B": self.B.clone(),
            "occupied_rows": self.occupied_rows,
            "total_rows": self.total_rows,
            "covariance_error_bound": self.covariance_error_bound,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("sketch feature dimension mismatch")
        if int(state["sketch_size"]) != self.sketch_size:
            raise ValueError("sketch size mismatch")
        matrix = state["B"].to(device=self.device, dtype=self.dtype)
        if matrix.shape != (self.sketch_size, self.feature_dim):
            raise ValueError("invalid sketch matrix shape")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("sketch matrix contains NaN or Inf")
        bound = float(state["covariance_error_bound"])
        rows = int(state["total_rows"])
        occupied = int(state["occupied_rows"])
        if bound < 0 or rows < 0 or not 0 <= occupied <= self.sketch_size:
            raise ValueError("invalid sketch metadata")
        self.B.copy_(matrix)
        self.covariance_error_bound = bound
        self.occupied_rows = occupied
        self.total_rows = rows
