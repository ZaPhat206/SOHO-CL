"""Mathematical sanity checks for the pre-implementation T-SOHO specification.

These tests intentionally use only synthetic tensors.  They are not a learner
implementation and do not exercise the dataset, backbone, SOHO, or FlyCL code.
"""

import torch


DTYPE = torch.float64
TOL = 1e-5


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).to(DTYPE)


def _ridge_weights(G: torch.Tensor, Q: torch.Tensor, ridge: float) -> torch.Tensor:
    eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return torch.linalg.solve(G + ridge * eye, Q)


def _positive_topk(values: torch.Tensor, k: int) -> torch.Tensor:
    """Per-row positive WTA, matching the relevant SOHO/FlyCL convention."""
    top_values, top_indices = values.topk(k, dim=1, largest=True)
    output = torch.zeros_like(values)
    return output.scatter(1, top_indices, top_values)


def _simplex_etf(num_classes: int) -> torch.Tensor:
    """Return E:(C-1,C), EE^T=I and E^TE=I-11^T/C."""
    centering = torch.eye(num_classes, dtype=DTYPE) - torch.ones(
        (num_classes, num_classes), dtype=DTYPE
    ) / num_classes
    eigenvalues, eigenvectors = torch.linalg.eigh(centering)
    basis = eigenvectors[:, eigenvalues > 0.5]  # (C, C-1), deterministic enough for algebra
    return basis.T


def test_streaming_G_and_Q_equal_batch_statistics():
    torch.manual_seed(7)
    n, d, c = 23, 6, 5
    features = torch.randn(n, d, dtype=DTYPE)
    labels = torch.randint(c, (n,))
    targets = _one_hot(labels, c)

    G_batch = features.T @ features
    Q_batch = features.T @ targets

    G_stream = torch.zeros((d, d), dtype=DTYPE)
    Q_stream = torch.zeros((d, c), dtype=DTYPE)
    for start, end in ((0, 4), (4, 13), (13, n)):
        X_chunk, Y_chunk = features[start:end], targets[start:end]
        G_stream += X_chunk.T @ X_chunk
        Q_stream += X_chunk.T @ Y_chunk

    torch.testing.assert_close(G_stream, G_batch, rtol=0, atol=TOL)
    torch.testing.assert_close(Q_stream, Q_batch, rtol=0, atol=TOL)


def test_streaming_ridge_logits_equal_batch_ridge_logits():
    torch.manual_seed(11)
    n, d, c, n_query = 31, 7, 4, 9
    ridge = 0.37
    features = torch.randn(n, d, dtype=DTYPE)
    labels = torch.randint(c, (n,))
    targets = _one_hot(labels, c)
    query = torch.randn(n_query, d, dtype=DTYPE)

    W_batch = _ridge_weights(features.T @ features, features.T @ targets, ridge)

    G_stream = torch.zeros((d, d), dtype=DTYPE)
    Q_stream = torch.zeros((d, c), dtype=DTYPE)
    for start, end in ((0, 8), (8, 17), (17, n)):
        X_chunk, Y_chunk = features[start:end], targets[start:end]
        G_stream += X_chunk.T @ X_chunk
        Q_stream += X_chunk.T @ Y_chunk
    W_stream = _ridge_weights(G_stream, Q_stream, ridge)

    torch.testing.assert_close(query @ W_stream, query @ W_batch, rtol=0, atol=TOL)


def test_orthogonal_transport_with_isotropic_ridge_preserves_logits():
    torch.manual_seed(19)
    n, d, c, n_query = 29, 8, 5, 10
    ridge = 0.25
    features = torch.randn(n, d, dtype=DTYPE)
    targets = _one_hot(torch.randint(c, (n,)), c)
    query = torch.randn(n_query, d, dtype=DTYPE)

    # Row features transform as X' = X U.  The correctly transported Ridge
    # weights are W' = U^T W, and isotropic regularization is invariant.
    U, _ = torch.linalg.qr(torch.randn(d, d, dtype=DTYPE))
    W_raw = _ridge_weights(features.T @ features, features.T @ targets, ridge)
    features_rotated = features @ U
    W_rotated = _ridge_weights(
        features_rotated.T @ features_rotated,
        features_rotated.T @ targets,
        ridge,
    )

    torch.testing.assert_close(W_rotated, U.T @ W_raw, rtol=0, atol=TOL)
    torch.testing.assert_close(query @ W_raw, (query @ U) @ W_rotated, rtol=0, atol=TOL)


def test_dynamic_topk_has_no_shared_linear_transport_in_general():
    # Both samples have identical old WTA outputs, so every linear transport T
    # maps them to the same vector.  A changed linear projection followed by
    # WTA gives distinct outputs; therefore no sample-independent T can make
    # the old sparse features equal these new sparse features for both samples.
    samples = torch.tensor([[1.0, 0.0], [1.0, 0.6]], dtype=DTYPE)
    old_projection = torch.eye(2, dtype=DTYPE)
    new_projection = torch.tensor([[0.0, 2.0], [1.0, 0.0]], dtype=DTYPE)

    old_wta = _positive_topk(samples @ old_projection.T, k=1)
    new_wta = _positive_topk(samples @ new_projection.T, k=1)
    torch.testing.assert_close(old_wta[0], old_wta[1], rtol=0, atol=0)
    assert not torch.allclose(new_wta[0], new_wta[1], rtol=0, atol=0)

    # Directly verify the contradiction: equal inputs to a linear map have
    # equal outputs, whereas the required target vectors are unequal.
    candidate_transport = torch.linalg.lstsq(old_wta, new_wta).solution
    residual = old_wta @ candidate_transport - new_wta
    assert torch.linalg.vector_norm(residual).item() > 0.1


def test_full_rank_simplex_etf_preserves_raw_ridge_argmax():
    torch.manual_seed(23)
    n, d, c, n_query = 37, 6, 5, 41
    ridge = 0.4
    features = torch.randn(n, d, dtype=DTYPE)
    targets = _one_hot(torch.randint(c, (n,)), c)
    query = torch.randn(n_query, d, dtype=DTYPE)

    W_raw = _ridge_weights(features.T @ features, features.T @ targets, ridge)
    raw_logits = query @ W_raw
    E = _simplex_etf(c)  # full simplex rank C-1, not strict low rank
    P = W_raw @ E.T
    decoded_logits = 2.0 * (query @ P) @ E - (E.square().sum(dim=0)).unsqueeze(0)

    expected_projector = torch.eye(c, dtype=DTYPE) - torch.ones((c, c), dtype=DTYPE) / c
    torch.testing.assert_close(E.T @ E, expected_projector, rtol=0, atol=TOL)
    assert torch.equal(raw_logits.argmax(dim=1), decoded_logits.argmax(dim=1))


def test_strict_low_rank_code_has_expected_EtE_properties():
    classes, rank = 6, 3
    assert rank < classes - 1
    full_etf = _simplex_etf(classes)
    E = full_etf[:rank]  # any orthonormal subset is a valid strict-low-rank code
    projector = E.T @ E

    torch.testing.assert_close(E @ E.T, torch.eye(rank, dtype=DTYPE), rtol=0, atol=TOL)
    torch.testing.assert_close(projector, projector.T, rtol=0, atol=TOL)
    torch.testing.assert_close(projector @ projector, projector, rtol=0, atol=TOL)
    assert torch.linalg.matrix_rank(projector).item() == rank

    centering = torch.eye(classes, dtype=DTYPE) - torch.ones((classes, classes), dtype=DTYPE) / classes
    assert not torch.allclose(projector, torch.eye(classes, dtype=DTYPE), rtol=0, atol=TOL)
    assert not torch.allclose(projector, centering, rtol=0, atol=TOL)
