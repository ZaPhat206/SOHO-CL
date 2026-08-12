import torch


def effective_rank(requested_rank, classes):
    return max(0, min(int(requested_rank), max(classes - 1, 0)))


def _orient(rows):
    rows = rows.clone()
    for i in range(rows.shape[0]):
        pivot = torch.argmax(rows[i].abs())
        if rows[i, pivot] < 0:
            rows[i].mul_(-1)
    return rows


def simplex_code(classes, rank, dtype=torch.float64, device="cpu"):
    rank = effective_rank(rank, classes)
    if rank == 0:
        return torch.empty((0, classes), dtype=dtype, device=device)
    centering = torch.eye(classes, dtype=dtype, device=device) - torch.ones((classes, classes), dtype=dtype, device=device) / classes
    _, vectors = torch.linalg.eigh(centering)
    return _orient(vectors[:, -rank:].T)


def random_code(classes, rank, seed, dtype=torch.float64, device="cpu"):
    rank = effective_rank(rank, classes)
    if rank == 0:
        return torch.empty((0, classes), dtype=dtype, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn((classes, classes), generator=generator, dtype=dtype).to(device)
    raw = raw - raw.mean(0, keepdim=True)
    q, _ = torch.linalg.qr(raw)
    # The last C-1 columns span the centered subspace because qr is deterministic for this raw matrix.
    centered = q[:, :classes - 1].T
    centered = centered - centered.mean(1, keepdim=True)
    centered = torch.linalg.qr(centered.T).Q[:, :classes - 1].T
    return _orient(centered[:rank])


def spectral_code(laplacian, rank):
    classes = laplacian.shape[0]
    rank = effective_rank(rank, classes)
    if rank == 0:
        return torch.empty((0, classes), dtype=laplacian.dtype, device=laplacian.device), torch.empty(0, dtype=laplacian.dtype, device=laplacian.device)
    eigenvalues, vectors = torch.linalg.eigh((laplacian + laplacian.T) / 2)
    # Largest non-trivial eigenvectors; for a Laplacian the zero/constant mode is excluded.
    selected = vectors[:, -rank:].T
    return _orient(selected), eigenvalues
