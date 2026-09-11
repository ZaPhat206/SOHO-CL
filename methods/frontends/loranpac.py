"""Official-code-faithful continual TSVD backend used by LoRanPAC controls.

The update follows ``TSVDNet.update_svd`` from the public LoRanPAC code: old
left singular vectors/values are combined with the current feature block via a
residual QR and a small-core SVD.  Only aggregate state is retained.
"""

from __future__ import annotations

import torch

from methods.analytic_ridge import persistent_tensor_bytes


def official_loranpac_rank(
    total_rows: int,
    *,
    dimension: int,
    truncate_percent: float,
    max_rank: int,
) -> int:
    """Return the rank selected by the released LoRanPAC implementation.

    The paper states ``ceil`` whereas the released code uses Python ``round``.
    This helper intentionally reproduces the code and caps the result by the
    available matrix rank.
    """

    if total_rows < 0 or dimension <= 0 or max_rank <= 0:
        raise ValueError("invalid LoRanPAC rank arguments")
    if not 0.0 <= truncate_percent < 100.0:
        raise ValueError("truncate_percent must be in [0, 100)")
    requested = round(total_rows * (1.0 - truncate_percent / 100.0))
    return min(int(requested), int(max_rank), int(dimension), int(total_rows))


def maximum_loranpac_rank_for_backend_budget(
    *,
    dimension: int,
    num_classes: int,
    target_bytes: int,
    statistics_element_size: int = 4,
    solver_element_size: int = 4,
) -> int:
    """Largest final TSVD rank fitting an existing backend byte budget.

    The accounting includes the state actually kept by this implementation:
    ``Q``, class counts, classifier weights, left singular vectors, and
    singular values.  A common frontend projection is intentionally excluded
    from both sides of a backend-only comparison.
    """

    if min(dimension, num_classes, target_bytes) <= 0:
        raise ValueError("dimension, num_classes, and target_bytes must be positive")
    if min(statistics_element_size, solver_element_size) <= 0:
        raise ValueError("element sizes must be positive")
    base = (
        dimension * num_classes * statistics_element_size
        + num_classes * statistics_element_size
        + dimension * num_classes * solver_element_size
    )
    per_rank = dimension * statistics_element_size + statistics_element_size
    if target_bytes < base + per_rank:
        return 0
    return min(dimension, (target_bytes - base) // per_rank)


class LoRanPACTSVBackend:
    """Continual truncated-SVD analytic classifier without sample replay.

    This is a backend for already-expanded random-ReLU codes.  It deliberately
    keeps LoRanPAC's code-level rank rule and projected classifier equation,
    while exposing the same prediction/state contract as the repository's
    additive Ridge backends.
    """

    is_exemplar_free = True

    def __init__(
        self,
        *,
        dimension: int,
        truncate_percent: float = 25.0,
        max_rank: int = 10_000,
        ridge_lambda: float = 0.0,
        device: str | torch.device = "cpu",
        statistics_dtype: torch.dtype = torch.float32,
        solver_dtype: torch.dtype = torch.float32,
    ) -> None:
        if dimension <= 0 or max_rank <= 0 or max_rank > dimension:
            raise ValueError("max_rank must be in [1, dimension]")
        if not 0.0 <= truncate_percent < 100.0:
            raise ValueError("truncate_percent must be in [0, 100)")
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        if statistics_dtype not in {torch.float32, torch.float64}:
            raise ValueError("invalid statistics dtype")
        if solver_dtype not in {torch.float32, torch.float64}:
            raise ValueError("invalid solver dtype")
        self.dimension = int(dimension)
        self.truncate_percent = float(truncate_percent)
        self.max_rank = int(max_rank)
        self.ridge_lambda = float(ridge_lambda)
        self.device = torch.device(device)
        self.statistics_dtype = statistics_dtype
        self.solver_dtype = solver_dtype
        self.Q = torch.zeros(
            (self.dimension, 0), device=self.device, dtype=self.statistics_dtype
        )
        self.counts = torch.zeros(0, device=self.device, dtype=self.statistics_dtype)
        self.class_ids: list[int] = []
        self.weights: torch.Tensor | None = None
        self.U = torch.empty(
            (self.dimension, 0), device=self.device, dtype=self.statistics_dtype
        )
        self.s = torch.empty(0, device=self.device, dtype=self.statistics_dtype)
        self.total_rows = 0
        self.diagnostics: dict[str, object] = {
            "method": "loranpac_continual_tsvd",
            "rank_rule": "official_code_python_round",
            "paper_rank_rule": "ceil",
            "truncate_percent": self.truncate_percent,
            "max_rank": self.max_rank,
            "ridge_lambda": self.ridge_lambda,
            "task_id_required": False,
        }

    @property
    def effective_rank(self) -> int:
        return int(self.s.numel())

    def _validated_batch(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = features.to(device=self.device, dtype=self.statistics_dtype)
        target_labels = labels.to(device=self.device, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"features must have shape (B, {self.dimension})")
        if target_labels.ndim != 1 or len(target_labels) != len(values) or not len(values):
            raise ValueError("labels must align with a non-empty feature matrix")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return values, target_labels

    def _expanded_statistics(
        self, labels: torch.Tensor
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        class_ids = sorted(set(self.class_ids) | set(map(int, labels.cpu().tolist())))
        old_columns = {value: index for index, value in enumerate(self.class_ids)}
        new_columns = {value: index for index, value in enumerate(class_ids)}
        cross = torch.zeros(
            (self.dimension, len(class_ids)),
            device=self.device,
            dtype=self.statistics_dtype,
        )
        counts = torch.zeros(
            len(class_ids), device=self.device, dtype=self.statistics_dtype
        )
        for class_id, old_column in old_columns.items():
            new_column = new_columns[class_id]
            cross[:, new_column] = self.Q[:, old_column]
            counts[new_column] = self.counts[old_column]
        columns = torch.tensor(
            [new_columns[int(value)] for value in labels.cpu().tolist()],
            device=self.device,
            dtype=torch.long,
        )
        targets = torch.nn.functional.one_hot(
            columns, num_classes=len(class_ids)
        ).to(self.statistics_dtype)
        return class_ids, cross, counts, targets

    def _update_factor(self, values: torch.Tensor, keep: int) -> None:
        if keep <= 0:
            raise RuntimeError("LoRanPAC rank rule selected an empty factor")
        transposed = values.T
        if not self.effective_rank:
            left, singular_values, _ = torch.linalg.svd(
                transposed, full_matrices=False
            )
            self.U = left[:, :keep]
            self.s = singular_values[:keep].clamp_min(0)
            return

        upper_off_diagonal = self.U.T @ transposed
        residual = transposed - self.U @ upper_off_diagonal
        residual_basis, residual_factor = torch.linalg.qr(residual, mode="reduced")
        old_rank = self.effective_rank
        residual_rank = residual_basis.shape[1]
        upper = torch.cat(
            (
                torch.diag(self.s),
                upper_off_diagonal,
            ),
            dim=1,
        )
        lower = torch.cat(
            (
                torch.zeros(
                    (residual_rank, old_rank),
                    device=self.device,
                    dtype=self.statistics_dtype,
                ),
                residual_factor,
            ),
            dim=1,
        )
        core = torch.cat((upper, lower), dim=0)
        core_left, singular_values, _ = torch.linalg.svd(core, full_matrices=False)
        keep = min(keep, int(singular_values.numel()))
        basis = torch.cat((self.U, residual_basis), dim=1)
        updated = basis @ core_left[:, :keep]
        # The released implementation re-orthogonalizes U after truncation.
        self.U, _ = torch.linalg.qr(updated, mode="reduced")
        self.s = singular_values[:keep].clamp_min(0)

    def _solve(
        self, cross: torch.Tensor, *, ridge_lambda: float | None = None
    ) -> tuple[torch.Tensor, float]:
        work_U = self.U.to(self.solver_dtype)
        work_s = self.s.to(self.solver_dtype)
        work_cross = cross.to(self.solver_dtype)
        ridge = self.ridge_lambda if ridge_lambda is None else float(ridge_lambda)
        if ridge < 0:
            raise ValueError("ridge_lambda must be non-negative")
        denominator = work_s.square() + ridge
        threshold = torch.finfo(self.solver_dtype).tiny
        if bool((denominator <= threshold).any()):
            raise RuntimeError("LoRanPAC retained a zero singular direction")
        projected_cross = work_U.T @ work_cross
        weights = work_U @ (projected_cross / denominator.unsqueeze(1))
        projected_residual = (
            denominator.unsqueeze(1) * (work_U.T @ weights) - projected_cross
        )
        scale = max(float(torch.linalg.vector_norm(projected_cross).item()), 1.0)
        residual = float(torch.linalg.vector_norm(projected_residual).item()) / scale
        return weights, residual

    def solve_weights(self, ridge_lambda: float | None = None) -> tuple[torch.Tensor, float]:
        """Solve the current projected system for a preregistered Ridge value.

        This supports a non-persistent matched-Ridge diagnostic without running
        or storing a second TSVD trajectory.
        """

        if not self.total_rows:
            raise RuntimeError("backend has not been updated")
        return self._solve(self.Q, ridge_lambda=ridge_lambda)

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values, target_labels = self._validated_batch(features, labels)
        class_ids, cross, counts, targets = self._expanded_statistics(target_labels)
        new_total = self.total_rows + len(values)
        keep = official_loranpac_rank(
            new_total,
            dimension=self.dimension,
            truncate_percent=self.truncate_percent,
            max_rank=self.max_rank,
        )
        self._update_factor(values, keep)
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        weights, residual = self._solve(new_cross)
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = weights
        self.total_rows = new_total
        identity = torch.eye(
            self.effective_rank, device=self.device, dtype=self.statistics_dtype
        )
        orthogonality = torch.linalg.vector_norm(self.U.T @ self.U - identity)
        self.diagnostics.update(
            requested_rank=keep,
            effective_rank=self.effective_rank,
            total_rows=self.total_rows,
            solver_relative_residual=residual,
            orthogonality_residual=float(orthogonality.item()),
            minimum_singular_value=float(self.s.min().item()),
            maximum_singular_value=float(self.s.max().item()),
        )
        if not bool(
            torch.isfinite(self.U).all()
            and torch.isfinite(self.s).all()
            and torch.isfinite(self.weights).all()
        ):
            raise RuntimeError("LoRanPAC update produced NaN or Inf")
        self.assert_exemplar_free_state()

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("backend has not been updated")
        values = features.to(device=self.device, dtype=self.weights.dtype)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"features must have shape (B, {self.dimension})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        return values @ self.weights

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(1).cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns])

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "Q": self.Q,
            "counts": self.counts,
            "factor.U": self.U,
            "factor.s": self.s,
        }
        if self.weights is not None:
            tensors["weights"] = self.weights
        return tensors

    def persistent_state_bytes(self) -> int:
        return persistent_tensor_bytes(self.persistent_tensors())

    def assert_exemplar_free_state(self) -> None:
        classes = len(self.class_ids)
        if self.Q.shape != (self.dimension, classes):
            raise AssertionError("invalid Q shape")
        if self.counts.shape != (classes,):
            raise AssertionError("invalid class-count shape")
        if self.U.shape != (self.dimension, self.effective_rank):
            raise AssertionError("invalid left-factor shape")
        if self.weights is not None and self.weights.shape != self.Q.shape:
            raise AssertionError("invalid classifier shape")
        if self.effective_rank > self.max_rank:
            raise AssertionError("factor exceeds the locked rank cap")
        forbidden = ("history", "sample", "feature_cache", "labels", "codes")
        for name, tensor in self.persistent_tensors().items():
            if any(token in name.lower() for token in forbidden):
                raise AssertionError(f"forbidden sample-level state: {name}")
            if (
                name != "factor.U"
                and
                tensor.ndim >= 2
                and self.total_rows not in {self.dimension, classes}
                and self.total_rows in tensor.shape
            ):
                raise AssertionError(f"historical sample dimension in {name}")

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "method": "loranpac_continual_tsvd",
            "dimension": self.dimension,
            "truncate_percent": self.truncate_percent,
            "max_rank": self.max_rank,
            "ridge_lambda": self.ridge_lambda,
            "statistics_dtype": str(self.statistics_dtype).removeprefix("torch."),
            "solver_dtype": str(self.solver_dtype).removeprefix("torch."),
            "Q": self.Q.detach().cpu().clone(),
            "counts": self.counts.detach().cpu().clone(),
            "class_ids": list(self.class_ids),
            "weights": None if self.weights is None else self.weights.detach().cpu().clone(),
            "U": self.U.detach().cpu().clone(),
            "s": self.s.detach().cpu().clone(),
            "total_rows": self.total_rows,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or state.get("method") != "loranpac_continual_tsvd":
            raise ValueError("unsupported LoRanPAC checkpoint")
        expected = {
            "dimension": self.dimension,
            "truncate_percent": self.truncate_percent,
            "max_rank": self.max_rank,
            "ridge_lambda": self.ridge_lambda,
            "statistics_dtype": str(self.statistics_dtype).removeprefix("torch."),
            "solver_dtype": str(self.solver_dtype).removeprefix("torch."),
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"LoRanPAC checkpoint mismatch for {field}")
        self.class_ids = [int(value) for value in state["class_ids"]]
        self.total_rows = int(state["total_rows"])
        self.Q = state["Q"].to(device=self.device, dtype=self.statistics_dtype)
        self.counts = state["counts"].to(
            device=self.device, dtype=self.statistics_dtype
        )
        self.U = state["U"].to(device=self.device, dtype=self.statistics_dtype)
        self.s = state["s"].to(device=self.device, dtype=self.statistics_dtype)
        stored_weights = state["weights"]
        self.weights = None if stored_weights is None else stored_weights.to(
            device=self.device, dtype=self.solver_dtype
        )
        if self.class_ids != sorted(set(self.class_ids)) or self.total_rows < 0:
            raise ValueError("invalid LoRanPAC checkpoint metadata")
        if bool((self.s < 0).any()) or not all(
            bool(torch.isfinite(tensor).all()) for tensor in self.persistent_tensors().values()
        ):
            raise ValueError("invalid LoRanPAC checkpoint tensors")
        if abs(float(self.counts.sum().item()) - self.total_rows) > 1e-3:
            raise ValueError("checkpoint counts do not match total_rows")
        if self.effective_rank:
            identity = torch.eye(
                self.effective_rank, device=self.device, dtype=self.statistics_dtype
            )
            tolerance = 1e-8 if self.statistics_dtype == torch.float64 else 1e-3
            if not torch.allclose(
                self.U.T @ self.U, identity, atol=tolerance, rtol=tolerance
            ):
                raise ValueError("checkpoint LoRanPAC basis is not orthonormal")
        self.assert_exemplar_free_state()


__all__ = [
    "LoRanPACTSVBackend",
    "maximum_loranpac_rank_for_backend_budget",
    "official_loranpac_rank",
]
