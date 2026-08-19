"""Deterministic continual truncated SVD without sample replay."""

from __future__ import annotations

import torch


def _canonicalize_columns(matrix: torch.Tensor) -> torch.Tensor:
    """Resolve the independent sign ambiguity of singular vectors."""
    if not matrix.shape[1]:
        return matrix
    pivots = matrix.abs().argmax(dim=0, keepdim=True)
    signs = matrix.gather(0, pivots).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return matrix * signs


class StreamingTruncatedSVD:
    """Maintain left singular factors of a streamed row design matrix.

    The update SVD is computed from a QR factorization of
    ``[U diag(s), rows.T]``. Historical rows are represented only by ``U,s``
    and are never retained.
    """

    def __init__(
        self,
        feature_dim: int,
        max_rank: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if max_rank < 0 or max_rank > feature_dim:
            raise ValueError("max_rank must be in [0, feature_dim]")
        self.feature_dim = int(feature_dim)
        self.max_rank = int(max_rank)
        self.device = torch.device(device)
        self.dtype = dtype
        self.U = torch.empty(
            (self.feature_dim, 0), device=self.device, dtype=self.dtype
        )
        self.s = torch.empty((0,), device=self.device, dtype=self.dtype)
        self.total_rows = 0

    @property
    def effective_rank(self) -> int:
        return int(self.s.numel())

    def update(self, rows: torch.Tensor) -> None:
        values = rows.to(device=self.device, dtype=self.dtype)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"rows must have shape (N, {self.feature_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("SVD rows contain NaN or Inf")
        if values.shape[0] == 0:
            return
        if self.max_rank == 0:
            self.total_rows += int(values.shape[0])
            return

        previous = self.U * self.s.unsqueeze(0)
        factor = torch.cat((previous, values.T), dim=1)
        basis, core = torch.linalg.qr(factor, mode="reduced")
        left, singular_values, _ = torch.linalg.svd(core, full_matrices=False)
        keep = min(self.max_rank, int(singular_values.numel()))
        updated = basis @ left[:, :keep]
        self.U = _canonicalize_columns(updated)
        self.s = singular_values[:keep].clamp_min(0)

        if not bool(torch.isfinite(self.U).all() and torch.isfinite(self.s).all()):
            raise RuntimeError("streaming SVD produced NaN or Inf")
        self.total_rows += int(values.shape[0])

    def state_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "max_rank": self.max_rank,
            "U": self.U.detach().cpu().clone(),
            "s": self.s.detach().cpu().clone(),
            "total_rows": self.total_rows,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("SVD feature dimension mismatch")
        if int(state["max_rank"]) != self.max_rank:
            raise ValueError("SVD rank mismatch")
        U = state["U"].to(device=self.device, dtype=self.dtype)
        s = state["s"].to(device=self.device, dtype=self.dtype)
        rows = int(state["total_rows"])
        if U.ndim != 2 or U.shape[0] != self.feature_dim:
            raise ValueError("invalid U shape")
        if s.ndim != 1 or U.shape[1] != s.shape[0] or len(s) > self.max_rank:
            raise ValueError("invalid singular-value shape")
        if rows < 0 or bool((s < 0).any()):
            raise ValueError("invalid SVD metadata")
        if not bool(torch.isfinite(U).all() and torch.isfinite(s).all()):
            raise ValueError("SVD checkpoint contains NaN or Inf")
        if U.shape[1]:
            identity = torch.eye(U.shape[1], device=self.device, dtype=self.dtype)
            tolerance = 1e-8 if self.dtype == torch.float64 else 1e-4
            if not torch.allclose(U.T @ U, identity, atol=tolerance, rtol=tolerance):
                raise ValueError("checkpoint U is not orthonormal")
        self.U, self.s, self.total_rows = U, s, rows
