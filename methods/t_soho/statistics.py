import torch


class StreamingStatistics:
    """Bounded, exemplar-free sufficient statistics in a fixed feature space."""

    def __init__(self, feature_dim, device="cpu", dtype=torch.float64):
        self.feature_dim, self.device, self.dtype = feature_dim, torch.device(device), dtype
        self.G = torch.zeros((feature_dim, feature_dim), device=self.device, dtype=dtype)
        self.Q = torch.zeros((feature_dim, 0), device=self.device, dtype=dtype)
        self.counts = torch.zeros(0, device=self.device, dtype=dtype)
        self.sums = torch.zeros((0, feature_dim), device=self.device, dtype=dtype)
        self.sq_sums = torch.zeros((0, feature_dim), device=self.device, dtype=dtype)
        self.class_ids = []

    def _expand(self, labels):
        new = sorted(set(map(int, labels.detach().cpu().tolist())) - set(self.class_ids))
        if not new:
            return
        n = len(new)
        self.class_ids.extend(new)
        self.Q = torch.cat((self.Q, torch.zeros((self.feature_dim, n), device=self.device, dtype=self.dtype)), 1)
        self.counts = torch.cat((self.counts, torch.zeros(n, device=self.device, dtype=self.dtype)))
        self.sums = torch.cat((self.sums, torch.zeros((n, self.feature_dim), device=self.device, dtype=self.dtype)))
        self.sq_sums = torch.cat((self.sq_sums, torch.zeros((n, self.feature_dim), device=self.device, dtype=self.dtype)))

    def update(self, features, labels):
        x, labels = features.to(self.device, self.dtype), labels.to(self.device)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError("features must have shape (B, feature_dim)")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("features contain NaN/Inf")
        self._expand(labels)
        index = torch.tensor([self.class_ids.index(int(v)) for v in labels.cpu().tolist()], device=self.device)
        y = torch.nn.functional.one_hot(index, len(self.class_ids)).to(self.dtype)
        self.G += x.T @ x
        self.Q += x.T @ y
        self.counts.scatter_add_(0, index, torch.ones_like(index, dtype=self.dtype))
        self.sums.index_add_(0, index, x)
        self.sq_sums.index_add_(0, index, x.square())

    def state_dict(self):
        return {k: getattr(self, k).clone() if isinstance(getattr(self, k), torch.Tensor) else list(getattr(self, k))
                for k in ("G", "Q", "counts", "sums", "sq_sums", "class_ids")}

    def load_state_dict(self, state):
        for name in ("G", "Q", "counts", "sums", "sq_sums"):
            setattr(self, name, state[name].to(self.device, self.dtype))
        self.class_ids = list(state["class_ids"])
