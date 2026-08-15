"""Fixed-WTA zero-inflated analytic class-incremental learner."""

from __future__ import annotations

import inspect
import math

import torch

from models.flyhash import FlyHash
from .statistics import ZeroInflatedStatistics


def _tensor_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return sum(
            part.numel() * part.element_size()
            for part in (tensor.ccol_indices(), tensor.row_indices(), tensor.values())
        )
    raise ValueError(f"unsupported tensor layout {tensor.layout}")


class ZISOHOLearner:
    """Global task-free scorer over one immutable sparse WTA representation."""

    METHODS = {"wta_ncm", "support_only", "active_gaussian", "hurdle"}
    is_exemplar_free = True

    def __init__(
        self,
        *,
        raw_dim: int,
        expand_dim: int,
        synaptic_degree: int,
        coding_level: float,
        method: str = "hurdle",
        support_alpha: float = 0.5,
        variance_kappa: float = 50.0,
        variance_epsilon: float = 1e-4,
        score_chunk_size: int = 256,
        seed: int = 1993,
        device="cpu",
        dtype=torch.float32,
        projection: torch.Tensor | None = None,
    ):
        if raw_dim <= 0 or expand_dim <= 0:
            raise ValueError("raw_dim and expand_dim must be positive")
        if not 0 < synaptic_degree <= raw_dim:
            raise ValueError("synaptic_degree must be in [1,raw_dim]")
        if not 0 < coding_level <= 1 or int(expand_dim * coding_level) < 1:
            raise ValueError("coding_level must retain at least one coordinate")
        if method not in self.METHODS:
            raise ValueError(f"unknown method {method!r}")
        if support_alpha <= 0 or variance_kappa <= 0 or variance_epsilon <= 0:
            raise ValueError("smoothing parameters must be positive")
        if score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive")
        self.raw_dim, self.expand_dim = int(raw_dim), int(expand_dim)
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.active_size = max(1, int(self.expand_dim * self.coding_level))
        self.method = method
        self.support_alpha = float(support_alpha)
        self.variance_kappa = float(variance_kappa)
        self.variance_epsilon = float(variance_epsilon)
        self.score_chunk_size = int(score_chunk_size)
        self.seed = int(seed)
        self.device, self.dtype = torch.device(device), dtype
        if projection is None:
            devices = [] if self.device.type != "cuda" else [self.device.index or torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                anchor = FlyHash(self.raw_dim, self.expand_dim, self.synaptic_degree)
            anchor.to_sparse()
            self.projection = anchor.projection_matrix.to(self.device)
        else:
            if projection.shape != (self.expand_dim, self.raw_dim):
                raise ValueError("projection shape mismatch")
            if projection.layout != torch.sparse_csc:
                raise ValueError("projection must be sparse CSC")
            self.projection = projection.to(self.device)
        self.statistics = ZeroInflatedStatistics(
            self.expand_dim, device=self.device, dtype=self.dtype
        )
        self.diagnostics = {
            "model": "fixed_wta_zero_inflated",
            "method": self.method,
            "active_size": self.active_size,
        }

    @property
    def class_ids(self) -> list[int]:
        return self.statistics.class_ids

    def encode_sparse(self, raw_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = raw_features.to(self.device, self.projection.dtype)
        if features.ndim != 2 or features.shape[1] != self.raw_dim:
            raise ValueError(f"features must have shape (B,{self.raw_dim})")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features contain NaN or Inf")
        expanded = (self.projection @ features.T).T
        values, indices = expanded.topk(self.active_size, dim=1, largest=True)
        return indices, values.to(self.dtype)

    def update(self, raw_features: torch.Tensor, labels: torch.Tensor) -> None:
        self.update_from_sparse(*self.encode_sparse(raw_features), labels)

    def update_from_sparse(
        self, indices: torch.Tensor, values: torch.Tensor, labels: torch.Tensor
    ) -> None:
        self.statistics.update_sparse(indices, values, labels)
        self.diagnostics.update(
            seen_classes=self.statistics.num_classes,
            total_count=self.statistics.total_count,
        )
        self.assert_exemplar_free_state()

    def _parameters(self) -> dict[str, torch.Tensor]:
        stats = self.statistics
        if stats.num_classes == 0:
            raise RuntimeError("update() must be called before prediction")
        counts = stats.counts.clamp_min(1)[None, :]
        means = stats.active_sums / stats.active_counts.clamp_min(1)
        wta_means = stats.active_sums / counts
        class_numerator = (
            stats.active_sq_sums
            - stats.active_sums.square() / stats.active_counts.clamp_min(1)
        ).clamp_min(0)
        class_variance = class_numerator / (stats.active_counts - 1).clamp_min(1)
        pooled_count = stats.active_counts.sum(dim=1, keepdim=True)
        pooled_sum = stats.active_sums.sum(dim=1, keepdim=True)
        pooled_sq = stats.active_sq_sums.sum(dim=1, keepdim=True)
        pooled_numerator = (
            pooled_sq - pooled_sum.square() / pooled_count.clamp_min(1)
        ).clamp_min(0)
        pooled_variance = pooled_numerator / (pooled_count - 1).clamp_min(1)
        rho = stats.active_counts / (stats.active_counts + self.variance_kappa)
        variance = (
            rho * class_variance + (1 - rho) * pooled_variance
        ).clamp_min(self.variance_epsilon)
        probability = (
            (stats.active_counts + self.support_alpha)
            / (counts + 2 * self.support_alpha)
        ).clamp(1e-7, 1 - 1e-7)
        return {
            "mean": means,
            "wta_mean": wta_means,
            "variance": variance,
            "probability": probability,
        }

    def predict_logits_from_sparse(
        self, indices: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        index = indices.to(self.device, torch.long)
        amplitude = values.to(self.device, self.dtype)
        if index.ndim != 2 or amplitude.shape != index.shape:
            raise ValueError("indices and values must have the same (B,k) shape")
        if index.shape[1] != self.active_size:
            raise ValueError(f"sparse code must retain exactly {self.active_size} entries")
        if index.numel() and bool(((index < 0) | (index >= self.expand_dim)).any()):
            raise ValueError("sparse code index out of range")
        if not bool(torch.isfinite(amplitude).all()):
            raise ValueError("sparse code values contain NaN or Inf")
        parameters = self._parameters()
        means = parameters["mean"]
        if self.method == "wta_ncm":
            wta_means = parameters["wta_mean"]
            logits = -wta_means.square().sum(dim=0).expand(index.shape[0], -1).clone()
            for start in range(0, self.active_size, self.score_chunk_size):
                stop = min(start + self.score_chunk_size, self.active_size)
                logits += 2 * (
                    amplitude[:, start:stop, None]
                    * wta_means[index[:, start:stop]]
                ).sum(dim=1)
            return logits
        variance = parameters["variance"]
        probability = parameters["probability"]
        if self.method in {"support_only", "hurdle"}:
            logits = torch.log1p(-probability).sum(dim=0).expand(index.shape[0], -1).clone()
            support_correction = torch.log(probability) - torch.log1p(-probability)
        else:
            logits = torch.zeros(
                (index.shape[0], self.statistics.num_classes),
                device=self.device,
                dtype=self.dtype,
            )
            support_correction = None
        if self.method == "support_only":
            for start in range(0, self.active_size, self.score_chunk_size):
                stop = min(start + self.score_chunk_size, self.active_size)
                logits += support_correction[index[:, start:stop]].sum(dim=1)
            return logits
        gaussian_constant = -0.5 * (
            math.log(2 * math.pi) + torch.log(variance) + means.square() / variance
        )
        if support_correction is not None:
            gaussian_constant = gaussian_constant + support_correction
        gaussian_linear = means / variance
        gaussian_quadratic = -0.5 / variance
        for start in range(0, self.active_size, self.score_chunk_size):
            stop = min(start + self.score_chunk_size, self.active_size)
            selected = index[:, start:stop]
            selected_values = amplitude[:, start:stop, None]
            logits += (
                gaussian_constant[selected]
                + selected_values * gaussian_linear[selected]
                + selected_values.square() * gaussian_quadratic[selected]
            ).sum(dim=1)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("ZI-SOHO produced non-finite logits")
        return logits

    def predict_logits(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.predict_logits_from_sparse(*self.encode_sparse(raw_features))

    def predict(self, raw_features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(raw_features).argmax(dim=1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns], dtype=torch.long)

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "projection": self.projection,
            "class_counts": self.statistics.counts,
            "active_counts": self.statistics.active_counts,
            "active_sums": self.statistics.active_sums,
            "active_sq_sums": self.statistics.active_sq_sums,
        }

    def persistent_state_bytes(self) -> int:
        return sum(_tensor_bytes(value) for value in self.persistent_tensors().values())

    def assert_exemplar_free_state(self) -> None:
        forbidden = ("dataset", "loader", "cache", "historical", "sample", "image", "replay", "batch")
        names = tuple(self.__dict__) + tuple(self.statistics.__dict__)
        offending = [name for name in names if any(token in name.lower() for token in forbidden)]
        if offending:
            raise AssertionError(f"forbidden state names: {offending}")
        total = self.statistics.total_count
        if total > max(self.raw_dim, self.expand_dim, self.statistics.num_classes):
            for name, tensor in self.persistent_tensors().items():
                if tensor.ndim and total in tensor.shape:
                    raise AssertionError(f"{name} has a historical sample-count dimension")

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "raw_dim": self.raw_dim,
            "expand_dim": self.expand_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "method": self.method,
            "support_alpha": self.support_alpha,
            "variance_kappa": self.variance_kappa,
            "variance_epsilon": self.variance_epsilon,
            "score_chunk_size": self.score_chunk_size,
            "seed": self.seed,
            "projection": self.projection.detach().cpu(),
            "statistics": self.statistics.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported checkpoint version")
        for field in (
            "raw_dim", "expand_dim", "synaptic_degree", "coding_level", "method",
            "support_alpha", "variance_kappa", "variance_epsilon", "score_chunk_size", "seed",
        ):
            if state.get(field) != getattr(self, field):
                raise ValueError(f"checkpoint configuration mismatch for {field}")
        projection = state["projection"].to(self.device)
        if projection.shape != (self.expand_dim, self.raw_dim) or projection.layout != torch.sparse_csc:
            raise ValueError("invalid checkpoint projection")
        self.projection = projection
        self.statistics.load_state_dict(state["statistics"])
        self.diagnostics.update(
            seen_classes=self.statistics.num_classes,
            total_count=self.statistics.total_count,
        )
        self.assert_exemplar_free_state()


def create_learner(**kwargs) -> ZISOHOLearner:
    return ZISOHOLearner(**kwargs)


assert "task_id" not in inspect.signature(ZISOHOLearner.predict_logits).parameters
