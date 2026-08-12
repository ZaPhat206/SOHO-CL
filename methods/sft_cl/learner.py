"""Exemplar-free global analytic learners based on Fisher transport."""

from __future__ import annotations

import inspect

import torch

from .geometry import (
    analytic_confusion_affinity,
    confusion_between_scatter,
    fisher_transport,
    raw_ridge_weights,
    scatter_matrices,
    shuffled_affinity,
)
from .statistics import FixedFeatureStatistics


METHODS = {
    "raw_ridge",
    "fisher_hard",
    "confusion_fisher_hard",
    "fisher_soft",
    "confusion_fisher_soft",
    "shuffled_confusion_fisher_soft",
}


class SFTLearner:
    """Task-free classifier reconstructed entirely from bounded statistics.

    The only tensors retained by this learner have dimensions based on feature
    dimension D and seen-class count C.  No stored object grows with N_seen.
    """

    def __init__(
        self,
        method: str,
        feature_dim: int,
        ridge_lambda: float,
        requested_rank: int = 64,
        kappa: float = 1.0,
        delta: float = 0.1,
        scatter_epsilon: float = 1e-4,
        seed: int = 1993,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if method not in METHODS:
            raise ValueError(f"unknown SFT method {method!r}; choices: {sorted(METHODS)}")
        if ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        self.method = method
        self.ridge_lambda = float(ridge_lambda)
        self.requested_rank = int(requested_rank)
        self.kappa = float(kappa)
        self.delta = float(delta)
        self.scatter_epsilon = float(scatter_epsilon)
        self.seed = int(seed)
        self.statistics = FixedFeatureStatistics(feature_dim, device=device, dtype=dtype)
        self.weights: torch.Tensor | None = None  # (D, C), global raw-class logits
        self.diagnostics: dict = {}

    @property
    def class_ids(self) -> list[int]:
        return self.statistics.class_ids

    def _recompute(self) -> None:
        raw_weights = raw_ridge_weights(self.statistics, self.ridge_lambda)
        if self.method == "raw_ridge":
            self.weights = raw_weights
            self.diagnostics = {"transport": "identity", "effective_rank": self.statistics.feature_dim}
            return

        within, standard_between, means = scatter_matrices(self.statistics)
        use_confusion = "confusion" in self.method
        affinity = None
        between = standard_between
        if use_confusion:
            affinity = analytic_confusion_affinity(
                means,
                within,
                self.statistics.counts,
                raw_weights,
                self.scatter_epsilon,
            )
            if self.method == "shuffled_confusion_fisher_soft":
                affinity = shuffled_affinity(affinity, self.seed)
            between = confusion_between_scatter(means, self.statistics.counts, affinity)

        mode = "hard" if self.method.endswith("hard") else "soft"
        transport, geometry = fisher_transport(
            within,
            between,
            total_count=self.statistics.total_count,
            scatter_epsilon=self.scatter_epsilon,
            mode=mode,
            requested_rank=self.requested_rank,
            kappa=self.kappa,
            delta=self.delta,
        )
        gram_z = transport.T @ self.statistics.G @ transport
        cross_z = transport.T @ self.statistics.Q
        eye = torch.eye(gram_z.shape[0], dtype=gram_z.dtype, device=gram_z.device)
        projector = torch.linalg.solve(gram_z + self.ridge_lambda * eye, cross_z)
        self.weights = transport @ projector
        residual = (gram_z + self.ridge_lambda * eye) @ projector - cross_z
        self.diagnostics = {
            "transport": f"{mode}_fisher",
            "effective_rank": geometry.effective_rank,
            "eigenvalues": geometry.eigenvalues.detach().clone(),
            "gains": geometry.gains.detach().clone(),
            "solver_residual_max": float(residual.abs().max().item()),
            "affinity": affinity.detach().clone() if affinity is not None else None,
        }

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.statistics.update(features, labels)
        self._recompute()
        self.assert_exemplar_free_state()

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        x = features.to(device=self.statistics.device, dtype=self.statistics.dtype)
        if x.ndim != 2 or x.shape[1] != self.statistics.feature_dim:
            raise ValueError(f"features must have shape (B, {self.statistics.feature_dim})")
        return x @ self.weights

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(dim=1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns], dtype=torch.long)

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        result = {
            "G": self.statistics.G,
            "Q": self.statistics.Q,
            "counts": self.statistics.counts,
        }
        if self.weights is not None:
            result["weights"] = self.weights
        return result

    def persistent_state_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.persistent_tensors().values())

    def assert_exemplar_free_state(self) -> None:
        forbidden = ("dataset", "loader", "cache", "memory", "historical", "sample", "image", "replay", "batch")
        names = tuple(self.__dict__) + tuple(self.statistics.__dict__)
        offending = [name for name in names if any(token in name.lower() for token in forbidden)]
        if offending:
            raise AssertionError(f"forbidden sample-level state names: {offending}")
        total = self.statistics.total_count
        if total > max(self.statistics.feature_dim, self.statistics.num_classes):
            for name, tensor in self.persistent_tensors().items():
                if tensor.ndim and tensor.shape[0] == total:
                    raise AssertionError(f"{name} has a historical sample-count dimension")

    def state_dict(self) -> dict:
        """Checkpoint only sufficient statistics and config; rebuild weights on load."""
        return {
            "version": 1,
            "method": self.method,
            "ridge_lambda": self.ridge_lambda,
            "requested_rank": self.requested_rank,
            "kappa": self.kappa,
            "delta": self.delta,
            "scatter_epsilon": self.scatter_epsilon,
            "seed": self.seed,
            "statistics": self.statistics.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        for field in ("method", "ridge_lambda", "requested_rank", "kappa", "delta", "scatter_epsilon", "seed"):
            if field not in state:
                raise ValueError(f"checkpoint missing {field}")
        if state["method"] != self.method:
            raise ValueError("method mismatch")
        expected = (self.ridge_lambda, self.requested_rank, self.kappa, self.delta, self.scatter_epsilon, self.seed)
        actual = (float(state["ridge_lambda"]), int(state["requested_rank"]), float(state["kappa"]), float(state["delta"]), float(state["scatter_epsilon"]), int(state["seed"]))
        if expected != actual:
            raise ValueError("checkpoint configuration mismatch")
        self.statistics.load_state_dict(state["statistics"])
        self._recompute()
        self.assert_exemplar_free_state()


def create_learner(**kwargs) -> SFTLearner:
    return SFTLearner(**kwargs)


assert "task_id" not in inspect.signature(SFTLearner.predict_logits).parameters
