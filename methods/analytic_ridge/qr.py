"""Representation-agnostic square-root rank updates."""

from __future__ import annotations

import torch


def blocked_qr_rank_update(
    upper: torch.Tensor,
    update_rows: torch.Tensor,
    *,
    panel_size: int,
    trailing_chunk_size: int | None = None,
    preserve_update_rows: bool = True,
) -> torch.Tensor:
    """Return the positive-diagonal QR factor of ``[upper; update_rows]``.

    The routine knows nothing about the feature map that produced
    ``update_rows``.  It reuses the old factor as the output buffer and bounds
    trailing workspace when ``trailing_chunk_size`` is supplied.
    """
    if upper.ndim != 2 or upper.shape[0] != upper.shape[1] or not len(upper):
        raise ValueError("upper must be a non-empty square matrix")
    if update_rows.ndim != 2 or update_rows.shape[1] != len(upper):
        raise ValueError("update rows must align with the factor dimension")
    if not len(update_rows) or panel_size <= 0:
        raise ValueError("rank and panel size must be positive")
    if trailing_chunk_size is not None and trailing_chunk_size <= 0:
        raise ValueError("trailing chunk size must be positive when provided")
    if upper.device != update_rows.device or upper.dtype != update_rows.dtype:
        raise ValueError("factor and update rows must share device and dtype")

    dimension = len(upper)
    residual = update_rows.clone() if preserve_update_rows else update_rows
    for start in range(0, dimension, panel_size):
        end = min(start + panel_size, dimension)
        width = end - start
        panel = torch.cat(
            (upper[start:end, start:end], residual[:, start:end]), dim=0
        )
        reflectors, tau = torch.geqrf(panel)
        diagonal_block = torch.triu(reflectors[:width])
        signs = torch.where(
            diagonal_block.diagonal() < 0,
            -torch.ones((), device=upper.device, dtype=upper.dtype),
            torch.ones((), device=upper.device, dtype=upper.dtype),
        )

        trailing_width = (
            dimension - end
            if trailing_chunk_size is None
            else trailing_chunk_size
        )
        if end < dimension:
            for column_start in range(end, dimension, trailing_width):
                column_end = min(column_start + trailing_width, dimension)
                trailing = torch.cat(
                    (
                        upper[start:end, column_start:column_end],
                        residual[:, column_start:column_end],
                    ),
                    dim=0,
                )
                transformed = torch.ormqr(
                    reflectors, tau, trailing, left=True, transpose=True
                )
                upper[start:end, column_start:column_end].copy_(
                    signs[:, None] * transformed[:width]
                )
                residual[:, column_start:column_end].copy_(transformed[width:])

        upper[start:end, start:end].copy_(signs[:, None] * diagonal_block)
    return upper
