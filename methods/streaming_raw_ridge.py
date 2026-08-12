"""Exemplar-free streaming raw-feature Ridge baseline.

This is deliberately a baseline, not a T-SOHO implementation: it learns no
code, graph, ETF, random projection, or task-specific classifier head.
"""

import time

import numpy as np
import torch

from methods.base_cl import BaseCL
from utils.train_utils import feature_extract


def select_ridge_parameter(features, targets, ridge_lower, ridge_upper):
    """Match the existing FlyCL GCV policy on the current task's features."""
    U, singular_values, _ = torch.linalg.svd(features, full_matrices=False)
    singular_values_sq = singular_values.square()
    UTY = U.T @ targets
    ridges = torch.tensor(
        10.0 ** np.arange(ridge_lower, ridge_upper),
        dtype=features.dtype,
        device=features.device,
    )
    n_samples = features.shape[0]
    scores = []
    for ridge in ridges:
        diagonal = singular_values_sq / (singular_values_sq + ridge)
        degrees_of_freedom = diagonal.sum()
        prediction = U @ (diagonal[:, None] * UTY)
        residual = torch.linalg.vector_norm(targets - prediction).square()
        scores.append(((residual / n_samples) / (1 - degrees_of_freedom / n_samples).square()).item())
    return ridges[int(np.argmin(scores))]


class StreamingRawRidge(BaseCL):
    """Global Ridge classifier whose state is only G, Q, class IDs, and metadata."""

    def __init__(self, backbone, embedding_dim, ridge_lower, ridge_upper, device):
        super().__init__(backbone, device)
        self.embedding_dim = embedding_dim
        self.ridge_lower = ridge_lower
        self.ridge_upper = ridge_upper
        self.G_global = torch.zeros((embedding_dim, embedding_dim), device=device)
        self.Q_global = torch.zeros((embedding_dim, 0), device=device)
        self.Wo = None
        self.class_ids = []
        self._class_to_column = {}
        self.last_ridge = None

    def _ensure_classes(self, labels):
        new_ids = sorted(set(labels.detach().cpu().tolist()) - set(self._class_to_column))
        if not new_ids:
            return
        self.class_ids.extend(new_ids)
        self._class_to_column = {class_id: index for index, class_id in enumerate(self.class_ids)}
        extra = torch.zeros((self.embedding_dim, len(new_ids)), dtype=self.G_global.dtype, device=self.device)
        self.Q_global = torch.cat((self.Q_global, extra), dim=1)

    def _targets_for_seen_classes(self, labels):
        columns = torch.tensor(
            [self._class_to_column[class_id] for class_id in labels.detach().cpu().tolist()],
            dtype=torch.long,
            device=self.device,
        )
        targets = torch.zeros((labels.shape[0], len(self.class_ids)), dtype=self.G_global.dtype, device=self.device)
        targets.scatter_(1, columns.unsqueeze(1), 1.0)
        return targets

    def update_from_features(self, features, labels):
        """Update sufficient statistics and solve; exposed for synthetic integration tests."""
        if features.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim} feature dimensions, got {features.shape[1]}.")
        features = features.to(self.device)
        labels = labels.to(self.device)
        self._ensure_classes(labels)
        targets = self._targets_for_seen_classes(labels)

        self.G_global += features.T @ features
        self.Q_global += features.T @ targets
        self.last_ridge = float(select_ridge_parameter(features, targets, self.ridge_lower, self.ridge_upper))
        regularized = self.G_global + self.last_ridge * torch.eye(
            self.embedding_dim, dtype=self.G_global.dtype, device=self.device
        )
        self.Wo = torch.linalg.solve(regularized, self.Q_global)
        self.assert_exemplar_free_state()
        return self.last_ridge

    def logits_from_features(self, features):
        if self.Wo is None:
            raise RuntimeError("Cannot infer before the first streaming update.")
        return features.to(self.device) @ self.Wo

    def train_task(self, task_id, train_loader):
        training_start = time.time()
        extract_start = time.time()
        embeddings, labels = feature_extract(self.backbone, train_loader, self.device)
        extract_time = time.time() - extract_start
        ridge = self.update_from_features(embeddings, labels)
        return ridge, extract_time, time.time() - training_start

    def eval_task(self, task_id, test_loader):
        # task_id selects only the loader supplied by the outer evaluation loop.
        extract_start = time.time()
        embeddings, labels = feature_extract(self.backbone, test_loader, self.device)
        self.last_eval_feature_extract_time = time.time() - extract_start
        classifier_start = time.time()
        logits = self.logits_from_features(embeddings)
        predicted_columns = logits.argmax(dim=1).detach().cpu().tolist()
        predicted_ids = torch.tensor([self.class_ids[column] for column in predicted_columns], device=labels.device)
        self.last_eval_classifier_time = time.time() - classifier_start
        return np.mean(predicted_ids.cpu().numpy() == labels.cpu().numpy()) * 100

    def persistent_state_tensors(self):
        tensors = {"G_global": self.G_global, "Q_global": self.Q_global}
        if self.Wo is not None:
            tensors["Wo"] = self.Wo
        return tensors

    def persistent_state_summary(self):
        tensors = self.persistent_state_tensors()
        entries = [
            {"name": name, "shape": tuple(tensor.shape), "bytes": tensor.numel() * tensor.element_size()}
            for name, tensor in tensors.items()
        ]
        return {
            "tensors": entries,
            "tensor_bytes": sum(entry["bytes"] for entry in entries),
            "class_ids": tuple(self.class_ids),
            "metadata": {"embedding_dim": self.embedding_dim, "last_ridge": self.last_ridge},
        }

    def assert_exemplar_free_state(self):
        allowed = {"G_global", "Q_global", "Wo"}
        unexpected_tensors = [
            name for name, value in self.__dict__.items() if isinstance(value, torch.Tensor) and name not in allowed
        ]
        if unexpected_tensors:
            raise AssertionError(f"Unexpected persistent tensors: {unexpected_tensors}")
        for entry in self.persistent_state_summary()["tensors"]:
            if entry["shape"] and entry["shape"][0] not in {self.embedding_dim, len(self.class_ids)}:
                raise AssertionError(f"Sample-shaped persistent tensor detected: {entry}")
