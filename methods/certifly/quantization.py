"""Certified symmetric block quantization for FLY normal equations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


SUPPORTED_BITS = (8, 16)


def projected_all_int8_state_bytes(
    *,
    feature_dim: int,
    expand_dim: int,
    synaptic_degree: int,
    num_classes: int,
    block_size: int,
    scalar_bytes: int = 4,
    sparse_index_bytes: int = 8,
) -> dict[str, int | float]:
    """Analytical persistent-state projection under repository accounting."""
    if min(
        feature_dim,
        expand_dim,
        synaptic_degree,
        num_classes,
        block_size,
        scalar_bytes,
        sparse_index_bytes,
    ) <= 0:
        raise ValueError("state projection inputs must be positive")
    if synaptic_degree > feature_dim:
        raise ValueError("synaptic_degree cannot exceed feature_dim")
    nonzeros = expand_dim * synaptic_degree
    projection = (
        nonzeros * scalar_bytes
        + nonzeros * sparse_index_bytes
        + (feature_dim + 1) * sparse_index_bytes
    )
    block_count = math.ceil(expand_dim / block_size)
    gram = (
        expand_dim * (expand_dim - 1) // 2
        + expand_dim * scalar_bytes
        + block_count * (block_count + 1) // 2 * scalar_bytes
    )
    cross_and_classifier = 2 * expand_dim * num_classes * scalar_bytes
    counts = num_classes * scalar_bytes
    compressed = projection + gram + cross_and_classifier + counts
    exact = (
        projection
        + expand_dim * expand_dim * scalar_bytes
        + cross_and_classifier
        + counts
    )
    return {
        "projection_bytes": projection,
        "gram_bytes": gram,
        "cross_classifier_count_bytes": cross_and_classifier + counts,
        "compressed_total_bytes": compressed,
        "exact_fly_total_bytes": exact,
        "state_fraction": compressed / exact,
    }


def _integer_dtype(bits: int) -> torch.dtype:
    if bits == 8:
        return torch.int8
    if bits == 16:
        return torch.int16
    raise ValueError(f"unsupported quantization width: {bits}")


def _qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1


@dataclass(frozen=True)
class QuantizedBlock:
    row_block: int
    col_block: int
    bits: int
    scale: torch.Tensor
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.row_block < 0 or self.col_block < self.row_block:
            raise ValueError("block must belong to the upper triangle")
        if self.bits not in SUPPORTED_BITS:
            raise ValueError("block bit width must be 8 or 16")
        if self.scale.ndim != 0 or self.scale.dtype != torch.float32:
            raise ValueError("block scale must be a scalar float32 tensor")
        if self.values.dtype != _integer_dtype(self.bits):
            raise ValueError("block tensor dtype disagrees with bit width")
        if not bool(torch.isfinite(self.scale)) or float(self.scale.item()) <= 0:
            raise ValueError("block scale must be finite and positive")


def _quantize_values(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic symmetric quantization and a float32 scale."""
    if values.ndim != 1:
        values = values.reshape(-1)
    maximum = float(values.abs().max().item()) if values.numel() else 0.0
    scale_value = maximum / _qmax(bits) if maximum > 0 else 1.0
    scale = torch.tensor(scale_value, device=values.device, dtype=torch.float32)
    quantized = torch.round(values / scale_value).clamp(
        -_qmax(bits), _qmax(bits)
    ).to(_integer_dtype(bits))
    return quantized, scale


def _decode(values: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return values.to(dtype) * scale.to(dtype)


class QuantizedSymmetricGram:
    """Upper-triangular, coordinate-complete approximation of a Gram matrix.

    The object stores the exact diagonal and quantized strict-upper entries of
    a diagonally normalized correlation matrix. ``error_bound`` is cumulative
    across streaming requantizations and bounds the spectral Gram error through
    a Frobenius/triangle-inequality certificate.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        dimension: int,
        block_size: int,
        diagonal: torch.Tensor,
        blocks: list[QuantizedBlock],
        error_bound: float,
        last_quantization_error: float,
        merge_count: int,
        max_bits: int = 16,
    ) -> None:
        if dimension <= 0 or block_size <= 0:
            raise ValueError("dimension and block_size must be positive")
        if diagonal.shape != (dimension,) or diagonal.dtype not in {
            torch.float32, torch.float64
        }:
            raise ValueError("invalid exact diagonal")
        if not bool(torch.isfinite(diagonal).all()) or bool((diagonal < 0).any()):
            raise ValueError("Gram diagonal must be finite and non-negative")
        if not math.isfinite(error_bound) or error_bound < 0:
            raise ValueError("invalid cumulative error bound")
        if not math.isfinite(last_quantization_error) or last_quantization_error < 0:
            raise ValueError("invalid quantization error")
        if merge_count <= 0:
            raise ValueError("merge_count must be positive")
        if max_bits not in SUPPORTED_BITS:
            raise ValueError("max_bits must be 8 or 16")
        self.dimension = int(dimension)
        self.block_size = int(block_size)
        self.diagonal = diagonal
        self.blocks = blocks
        self.error_bound = float(error_bound)
        self.last_quantization_error = float(last_quantization_error)
        self.merge_count = int(merge_count)
        self.max_bits = int(max_bits)
        self._validate_blocks()

    @property
    def device(self) -> torch.device:
        return self.diagonal.device

    @property
    def dtype(self) -> torch.dtype:
        return self.diagonal.dtype

    @property
    def block_count(self) -> int:
        return math.ceil(self.dimension / self.block_size)

    def _bounds(self, block: int) -> tuple[int, int]:
        start = block * self.block_size
        return start, min(start + self.block_size, self.dimension)

    def _validate_blocks(self) -> None:
        expected = self.block_count * (self.block_count + 1) // 2
        if len(self.blocks) != expected:
            raise ValueError("quantized upper triangle is incomplete")
        locations = set()
        for block in self.blocks:
            if block.bits > self.max_bits:
                raise ValueError("quantized block exceeds configured maximum bit width")
            if block.scale.device != self.device or block.values.device != self.device:
                raise ValueError("quantized tensors must share the diagonal device")
            if block.col_block >= self.block_count:
                raise ValueError("block index outside Gram matrix")
            location = (block.row_block, block.col_block)
            if location in locations:
                raise ValueError("duplicate quantized block")
            locations.add(location)
            row_start, row_stop = self._bounds(block.row_block)
            col_start, col_stop = self._bounds(block.col_block)
            rows, cols = row_stop - row_start, col_stop - col_start
            expected_values = rows * cols if block.row_block != block.col_block else rows * (rows - 1) // 2
            if block.values.numel() != expected_values:
                raise ValueError("quantized block shape mismatch")

    @classmethod
    def from_dense(
        cls,
        gram: torch.Tensor,
        *,
        block_size: int,
        ridge_lambda: float,
        error_fraction: float,
        previous_error_bound: float = 0.0,
        merge_count: int = 1,
        max_bits: int = 16,
    ) -> "QuantizedSymmetricGram":
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not len(gram):
            raise ValueError("gram must be a non-empty square matrix")
        if gram.dtype not in {torch.float32, torch.float64}:
            raise ValueError("gram must use float32 or float64")
        if not bool(torch.isfinite(gram).all()):
            raise ValueError("gram contains NaN or Inf")
        if ridge_lambda <= 0 or not 0 < error_fraction < 1:
            raise ValueError("ridge_lambda and error_fraction must be positive")
        if max_bits not in SUPPORTED_BITS:
            raise ValueError("max_bits must be 8 or 16")
        if not math.isfinite(previous_error_bound) or previous_error_bound < 0:
            raise ValueError("previous error bound must be finite and non-negative")
        asymmetry = torch.linalg.vector_norm(gram - gram.T)
        scale = max(float(torch.linalg.vector_norm(gram).item()), 1.0)
        if float(asymmetry.item()) > 10 * torch.finfo(gram.dtype).eps * scale:
            raise ValueError("gram must be symmetric")
        gram = (gram + gram.T) * 0.5
        diagonal = torch.diagonal(gram).clone()
        if bool((diagonal < -10 * torch.finfo(gram.dtype).eps).any()):
            raise ValueError("gram has a negative diagonal")
        diagonal.clamp_min_(0)
        roots = diagonal.sqrt()
        reciprocal = torch.where(roots > 0, roots.reciprocal(), torch.zeros_like(roots))
        correlation = reciprocal[:, None] * gram * reciprocal[None, :]
        correlation.fill_diagonal_(1.0)

        target_total = error_fraction * ridge_lambda
        available = target_total - float(previous_error_bound)
        if available <= 0:
            raise RuntimeError("previous quantization error already exhausts certificate budget")

        candidates = []
        block_count = math.ceil(len(diagonal) / block_size)
        for row_block in range(block_count):
            row_start = row_block * block_size
            row_stop = min(row_start + block_size, len(diagonal))
            for col_block in range(row_block, block_count):
                col_start = col_block * block_size
                col_stop = min(col_start + block_size, len(diagonal))
                if row_block == col_block:
                    local = correlation[row_start:row_stop, col_start:col_stop]
                    indices = torch.triu_indices(
                        len(local), len(local), offset=1, device=gram.device
                    )
                    values = local[indices[0], indices[1]]
                    weights = roots[row_start:row_stop][indices[0]] * roots[col_start:col_stop][indices[1]]
                else:
                    values = correlation[row_start:row_stop, col_start:col_stop].reshape(-1)
                    weights = (
                        roots[row_start:row_stop, None]
                        * roots[col_start:col_stop][None, :]
                    ).reshape(-1)
                options = {}
                for bits in SUPPORTED_BITS:
                    quantized, quant_scale = _quantize_values(values, bits)
                    decoded = _decode(quantized, quant_scale, gram.dtype)
                    # Every stored entry has a mirrored lower-triangular entry.
                    squared_error = float((2.0 * ((decoded - values) * weights).square().sum()).item())
                    options[bits] = (quantized, quant_scale, squared_error)
                candidates.append((row_block, col_block, options))

        chosen = [8] * len(candidates)
        squared_total = sum(item[2][8][2] for item in candidates)
        if max_bits == 16 and math.sqrt(max(squared_total, 0.0)) > available:
            order = sorted(
                range(len(candidates)),
                key=lambda index: (
                    candidates[index][2][8][2] - candidates[index][2][16][2],
                    -candidates[index][0],
                    -candidates[index][1],
                ),
                reverse=True,
            )
            for index in order:
                old = candidates[index][2][8][2]
                new = candidates[index][2][16][2]
                squared_total += new - old
                chosen[index] = 16
                if math.sqrt(max(squared_total, 0.0)) <= available:
                    break
        quantization_error = math.sqrt(max(squared_total, 0.0))
        if quantization_error > available * (1 + 1e-6):
            raise RuntimeError(
                f"int{max_bits} blocks cannot satisfy the Gram error certificate"
            )

        blocks = []
        for choice, (row_block, col_block, options) in zip(chosen, candidates):
            quantized, quant_scale, _ = options[choice]
            blocks.append(
                QuantizedBlock(
                    row_block=row_block,
                    col_block=col_block,
                    bits=choice,
                    scale=quant_scale,
                    values=quantized,
                )
            )
        return cls(
            dimension=len(diagonal),
            block_size=block_size,
            diagonal=diagonal,
            blocks=blocks,
            error_bound=float(previous_error_bound) + quantization_error,
            last_quantization_error=quantization_error,
            merge_count=merge_count,
            max_bits=max_bits,
        )

    def reconstruct(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        output_dtype = self.dtype if dtype is None else dtype
        if output_dtype not in {torch.float32, torch.float64}:
            raise ValueError("reconstruction dtype must be float32 or float64")
        roots = self.diagonal.to(output_dtype).sqrt()
        gram = torch.zeros(
            (self.dimension, self.dimension), device=self.device, dtype=output_dtype
        )
        gram.diagonal().copy_(self.diagonal.to(output_dtype))
        for block in self.blocks:
            row_start, row_stop = self._bounds(block.row_block)
            col_start, col_stop = self._bounds(block.col_block)
            decoded = _decode(block.values, block.scale, output_dtype)
            if block.row_block == block.col_block:
                size = row_stop - row_start
                indices = torch.triu_indices(size, size, offset=1, device=self.device)
                values = decoded * roots[row_start:row_stop][indices[0]] * roots[col_start:col_stop][indices[1]]
                local = gram[row_start:row_stop, col_start:col_stop]
                local[indices[0], indices[1]] = values
                local[indices[1], indices[0]] = values
            else:
                values = decoded.reshape(row_stop - row_start, col_stop - col_start)
                values = values * roots[row_start:row_stop, None] * roots[col_start:col_stop][None, :]
                gram[row_start:row_stop, col_start:col_stop] = values
                gram[col_start:col_stop, row_start:row_stop] = values.T
        return gram

    def merge(
        self,
        delta_gram: torch.Tensor,
        *,
        ridge_lambda: float,
        error_fraction: float,
    ) -> "QuantizedSymmetricGram":
        if delta_gram.shape != (self.dimension, self.dimension):
            raise ValueError("delta Gram shape mismatch")
        updated = self.reconstruct(dtype=delta_gram.dtype) + delta_gram.to(self.device)
        return self.from_dense(
            updated,
            block_size=self.block_size,
            ridge_lambda=ridge_lambda,
            error_fraction=error_fraction,
            previous_error_bound=self.error_bound,
            merge_count=self.merge_count + 1,
            max_bits=self.max_bits,
        )

    def bit_histogram(self) -> dict[int, int]:
        return {bits: sum(block.bits == bits for block in self.blocks) for bits in SUPPORTED_BITS}

    def persistent_tensors(self, prefix: str = "gram") -> dict[str, torch.Tensor]:
        tensors = {f"{prefix}.diagonal": self.diagonal}
        for index, block in enumerate(self.blocks):
            tensors[f"{prefix}.block_{index}.values"] = block.values
            tensors[f"{prefix}.block_{index}.scale"] = block.scale
        return tensors

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "dimension": self.dimension,
            "block_size": self.block_size,
            "diagonal": self.diagonal.detach().cpu().clone(),
            "blocks": [
                {
                    "row_block": block.row_block,
                    "col_block": block.col_block,
                    "bits": block.bits,
                    "scale": block.scale.detach().cpu().clone(),
                    "values": block.values.detach().cpu().clone(),
                }
                for block in self.blocks
            ],
            "error_bound": self.error_bound,
            "last_quantization_error": self.last_quantization_error,
            "merge_count": self.merge_count,
            "max_bits": self.max_bits,
        }

    @classmethod
    def load_state_dict(
        cls, state: dict, *, device: str | torch.device
    ) -> "QuantizedSymmetricGram":
        if state.get("version") != cls.VERSION:
            raise ValueError("unsupported quantized Gram checkpoint")
        target = torch.device(device)
        blocks = [
            QuantizedBlock(
                row_block=int(item["row_block"]),
                col_block=int(item["col_block"]),
                bits=int(item["bits"]),
                scale=item["scale"].to(device=target, dtype=torch.float32),
                values=item["values"].to(device=target, dtype=_integer_dtype(int(item["bits"]))),
            )
            for item in state["blocks"]
        ]
        return cls(
            dimension=int(state["dimension"]),
            block_size=int(state["block_size"]),
            diagonal=state["diagonal"].to(target),
            blocks=blocks,
            error_bound=float(state["error_bound"]),
            last_quantization_error=float(state["last_quantization_error"]),
            merge_count=int(state["merge_count"]),
            max_bits=int(state.get("max_bits", 16)),
        )
