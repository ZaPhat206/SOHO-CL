import inspect
import torch

from .codes import effective_rank, random_code, simplex_code, spectral_code
from .graph import confusion_graph
from .statistics import StreamingStatistics


METHODS = {"raw_ridge", "random_orthogonal_code", "truncated_simplex_code", "spectral_confusion_code"}


class StreamingCodeLearner:
    def __init__(self, method, feature_dim, ridge_lambda, requested_rank=8, seed=1993, device="cpu", dtype=torch.float64):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}; choices: {sorted(METHODS)}")
        self.method, self.ridge_lambda, self.requested_rank, self.seed = method, float(ridge_lambda), int(requested_rank), int(seed)
        self.statistics = StreamingStatistics(feature_dim, device, dtype)
        self.W = None; self.P = None; self.E = None; self.diagnostics = {}

    @property
    def class_ids(self): return self.statistics.class_ids

    def _solve(self, rhs):
        G = self.statistics.G + self.ridge_lambda * torch.eye(self.statistics.feature_dim, dtype=self.statistics.dtype, device=self.statistics.device)
        return torch.linalg.solve(G, rhs)

    def _recompute(self):
        c = len(self.class_ids)
        self.W = self._solve(self.statistics.Q)
        r = effective_rank(self.requested_rank, c)
        # Raw Ridge is the same code-regression objective with the complete
        # one-hot code E=I_C.  Its nearest-code logits are 2XW-1, which have
        # the same argmax as conventional XW but make scorer semantics common.
        if self.method == "raw_ridge":
            self.E = torch.eye(c, dtype=self.statistics.dtype, device=self.statistics.device)
            self.P = self.W
            self.diagnostics = {"effective_rank": c, "code_type": "identity"}
            return
        if r == 0:
            self.E = None; self.P = None; self.diagnostics = {"effective_rank": r, "fallback": "raw_ridge_identity"}; return
        if self.method == "random_orthogonal_code": self.E = random_code(c, r, self.seed, self.statistics.dtype, self.statistics.device); self.diagnostics = {"effective_rank": r}
        elif self.method == "truncated_simplex_code": self.E = simplex_code(c, r, self.statistics.dtype, self.statistics.device); self.diagnostics = {"effective_rank": r}
        else:
            _, laplacian, diagnostics = confusion_graph(self.statistics.counts, self.statistics.sums, self.statistics.sq_sums)
            self.E, spectrum = spectral_code(laplacian, r); self.diagnostics = {**diagnostics, "effective_rank": r, "eigenvalues": spectrum}
        self.P = self._solve(self.statistics.Q @ self.E.T)

    def update(self, features, labels):
        self.statistics.update(features, labels); self._recompute(); self.assert_exemplar_free_state()

    def predict_logits(self, features):
        x = features.to(self.statistics.device, self.statistics.dtype)
        if self.W is None: raise RuntimeError("update() must be called before prediction")
        if self.E is None: return x @ self.W
        z = x @ self.P
        return 2 * z @ self.E - self.E.square().sum(0)

    def predict(self, features):
        columns = self.predict_logits(features).argmax(1).cpu().tolist()
        return torch.tensor([self.class_ids[index] for index in columns], dtype=torch.long)

    def persistent_state_bytes(self):
        return sum(t.numel() * t.element_size() for t in self._persistent_tensors().values())

    def _persistent_tensors(self):
        result = {"G": self.statistics.G, "Q": self.statistics.Q, "counts": self.statistics.counts, "sums": self.statistics.sums, "sq_sums": self.statistics.sq_sums}
        if self.W is not None: result["W"] = self.W
        if self.P is not None: result["P"] = self.P
        if self.E is not None: result["E"] = self.E
        return result

    def assert_exemplar_free_state(self):
        forbidden = ("dataset", "loader", "cache", "memory", "historical", "samples", "images")
        names = list(self.__dict__) + list(self.statistics.__dict__)
        if [name for name in names if any(token in name.lower() for token in forbidden)]:
            raise AssertionError("forbidden replay-like state name")

    def state_dict(self):
        return {"method": self.method, "ridge_lambda": self.ridge_lambda, "requested_rank": self.requested_rank, "seed": self.seed,
                "statistics": self.statistics.state_dict(), "W": self.W, "P": self.P, "E": self.E, "diagnostics": self.diagnostics}

    def load_state_dict(self, state):
        if state["method"] != self.method: raise ValueError("method mismatch")
        self.ridge_lambda, self.requested_rank, self.seed = state["ridge_lambda"], state["requested_rank"], state["seed"]
        self.statistics.load_state_dict(state["statistics"])
        self.W, self.P, self.E, self.diagnostics = state["W"], state["P"], state["E"], state["diagnostics"]
        self.assert_exemplar_free_state()


def create_learner(**kwargs):
    return StreamingCodeLearner(**kwargs)


assert "task_id" not in inspect.signature(StreamingCodeLearner.predict_logits).parameters
