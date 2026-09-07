"""M2 gates for unquantized Exact-Gram/QR equivalence."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from methods.analytic_ridge import DenseSquareRootBackend
from tools import srq_generalization_m2


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "srq_generalization_m2_equivalence.json"


def test_dense_square_root_checkpoint_round_trip():
    backend = DenseSquareRootBackend(
        dimension=11,
        ridge_lambda=3.0,
        update_backend="blocked_qr",
        update_panel_size=4,
        update_trailing_chunk_size=5,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(2025)
    features = torch.randn(17, 11, generator=generator, dtype=torch.float64)
    labels = torch.tensor([index % 3 for index in range(17)])
    backend.update(features, labels)
    restored = DenseSquareRootBackend(
        dimension=11,
        ridge_lambda=3.0,
        update_backend="blocked_qr",
        update_panel_size=4,
        update_trailing_chunk_size=5,
        statistics_dtype=torch.float64,
        solver_dtype=torch.float64,
    )
    restored.load_state_dict(backend.state_dict())
    torch.testing.assert_close(restored.factor, backend.factor, rtol=0, atol=0)
    torch.testing.assert_close(restored.weights, backend.weights, rtol=0, atol=0)


def test_m2_config_is_test_free_and_has_both_precisions():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["dtypes"] == ["float64", "float32"]
    assert set(config["cases"]) == {"gaussian", "column_scaled", "sparse"}


def test_m2_end_to_end_passes_all_locked_gates(tmp_path):
    output = tmp_path / "m2.json"
    payload = srq_generalization_m2.run(CONFIG, output)
    assert payload["status"] == "PASS_M2_UNQUANTIZED_EQUIVALENCE"
    assert payload["uses_test_set"] is False
    assert output.is_file()
    assert len(payload["results"]) == 6
    assert all(item["passed"] for item in payload["results"])
