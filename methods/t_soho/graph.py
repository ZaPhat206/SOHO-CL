import torch


def confusion_graph(counts, sums, sq_sums, variance_epsilon=1e-6, tau_epsilon=1e-6):
    classes = counts.numel()
    if classes == 0:
        empty = torch.empty((0, 0), dtype=sums.dtype, device=sums.device)
        return empty, empty, {"tau": None, "variance": torch.empty(0, dtype=sums.dtype, device=sums.device), "distances": empty}
    means = sums / counts.clamp_min(1).unsqueeze(1)
    numerator = sq_sums.sum(0) - (counts.unsqueeze(1) * means.square()).sum(0)
    variance = (numerator / max(float(counts.sum() - classes), 1.0)).clamp_min(variance_epsilon)
    distances = ((means[:, None, :] - means[None, :, :]).square() / variance).sum(-1)
    off_diag = distances[~torch.eye(classes, dtype=torch.bool, device=distances.device)]
    finite = off_diag[torch.isfinite(off_diag)]
    tau = torch.median(finite) if finite.numel() else torch.tensor(1.0, dtype=sums.dtype, device=sums.device)
    if not bool(torch.isfinite(tau)) or tau <= tau_epsilon:
        tau = torch.tensor(1.0, dtype=sums.dtype, device=sums.device)
    affinity = torch.exp(-distances / tau.clamp_min(tau_epsilon))
    affinity.fill_diagonal_(0)
    affinity = (affinity + affinity.T) / 2
    laplacian = torch.diag(affinity.sum(1)) - affinity
    return affinity, laplacian, {"tau": float(tau), "variance": variance, "distances": distances, "means": means}
