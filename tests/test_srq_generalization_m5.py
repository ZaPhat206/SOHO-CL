"""M5 gates for equal-budget analytic Ridge alternatives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest
import torch

from methods.analytic_ridge import (
    ExactGramBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from methods.analytic_ridge.compressed_upper import CompressedUpper
from tools import srq_generalization_m5 as m5


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "srq_generalization_m5_equal_budget_train_only.json"
NOTEBOOK = ROOT / "notebooks" / "srq_generalization_m5_equal_budget_colab.ipynb"


def test_m5_config_locks_train_only_byte_derived_comparison():
    config = m5._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["budget"]["derive_dimensions_before_accuracy"] is True
    assert config["countsketch"]["family"] == "one_nonzero_signed_hash"
    assert config["ridge_selection"]["policy"] == (
        "per_representation_train_only_calibration"
    )


def test_m5_rejects_test_use_and_accuracy_dimension_selection(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["uses_test_set"] = True
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        m5._read_config(path)
    config["uses_test_set"] = False
    config["accuracy_based_selection"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="accuracy"):
        m5._read_config(path)


def test_m5_locked_dimensions_exactly_follow_total_tensor_byte_budget():
    config = m5._read_config(CONFIG)
    lock = m5._derive_budget_lock(config)
    assert lock["target_p2b_total_persistent_bytes"] == 91_880_088
    assert lock["byte_matched_exact_dimension"] == 4_333
    assert lock["byte_matched_exact_total_persistent_bytes"] == 91_877_332
    assert lock["countsketch_dimension"] == 3_809
    assert lock["countsketch_total_persistent_bytes"] == 91_851_524
    assert lock["raw_ridge_total_persistent_bytes"] == 2_974_096
    assert lock["byte_matched_exact_underfill_fraction"] < 0.001
    assert lock["countsketch_underfill_fraction"] < 0.001


@pytest.mark.parametrize("mode", ["int8", "float16"])
def test_m5_symbolic_compressed_factor_bytes_match_real_layout(mode):
    dimension, block_size, group_size = 23, 7, 5
    matrix = torch.triu(torch.randn(dimension, dimension))
    matrix.diagonal().copy_(torch.arange(1, dimension + 1, dtype=torch.float32))
    compressed = CompressedUpper.from_upper(
        matrix,
        block_size=block_size,
        group_size=group_size,
        mode=mode,
    )
    actual = persistent_tensor_bytes(compressed.persistent_tensors("factor"))
    expected = m5._compressed_upper_bytes(
        dimension,
        block_size=block_size,
        group_size=group_size,
        mode=mode,
    )
    assert actual == expected


def test_m5_small_full_width_group_matches_symbolic_state():
    feature_dimension, dimension, classes = 8, 24, 6
    block_size, group_size = 6, 5
    generator = torch.Generator().manual_seed(71)
    features = torch.randn(72, feature_dimension, generator=generator)
    labels = torch.tensor([index % classes for index in range(72)])
    projection = torch.randn(
        feature_dimension, dimension, generator=torch.Generator().manual_seed(2025)
    )
    encoder = lambda values: torch.relu(values @ projection)
    common = m5._common_backend(dimension, 100.0, torch.device("cpu"))
    backends = {
        "full_width_exact": ExactGramBackend(**common),
        "fp16_square_root": SquareRootBackend(
            storage_mode="float16",
            block_size=block_size,
            group_size=group_size,
            update_panel_size=7,
            first_update_backend="gram_cholesky",
            quantization_backend="streaming",
            quantization_batch_blocks=3,
            **common,
        ),
        "p2b_int8": SquareRootBackend(
            storage_mode="int8",
            block_size=block_size,
            group_size=group_size,
            update_panel_size=7,
            first_update_backend="gram_cholesky",
            quantization_backend="streaming",
            quantization_batch_blocks=3,
            **common,
        ),
    }
    training_parts, validation_parts = [], []
    for task in range(3):
        indices = torch.where((labels == 2 * task) | (labels == 2 * task + 1))[0]
        training_parts.append(indices[:16])
        validation_parts.append(indices[16:])
    projection_bytes = projection.numel() * projection.element_size()
    expected = {
        "full_width_exact": projection_bytes
        + m5._exact_backend_bytes(dimension, classes),
        "fp16_square_root": projection_bytes
        + m5._square_root_backend_bytes(
            dimension,
            classes,
            block_size=block_size,
            group_size=group_size,
            mode="float16",
        ),
        "p2b_int8": projection_bytes
        + m5._square_root_backend_bytes(
            dimension,
            classes,
            block_size=block_size,
            group_size=group_size,
            mode="int8",
        ),
    }
    result = m5._run_group(
        encoder=encoder,
        backends=backends,
        extra_tensors={"projection": projection},
        expected_total_bytes=expected,
        features=features,
        labels=labels,
        training_parts=training_parts,
        validation_parts=validation_parts,
        encode_batch_size=7,
        evaluation_batch_size=8,
        device=torch.device("cpu"),
    )
    assert len(result["records"]) == 3
    for name, expected_bytes in expected.items():
        assert result["records"][-1]["state"][name][
            "total_persistent_bytes"
        ] == expected_bytes


def test_m5_summary_detects_a_faster_equal_state_accuracy_dominator():
    config = m5._read_config(CONFIG)
    config["gates"]["maximum_p2b_validation_aia_loss_pp"] = 100.0
    config["gates"]["maximum_equal_or_lower_state_aia_advantage_pp"] = 100.0
    names = [
        "full_width_exact",
        "fp16_square_root",
        "p2b_int8",
        "byte_matched_exact",
        "countsketch_exact",
        "raw_feature_ridge",
    ]
    accuracies = {name: 80.0 for name in names}
    accuracies["full_width_exact"] = 80.1
    accuracies["countsketch_exact"] = 80.2
    accuracies["raw_feature_ridge"] = 79.0
    states = {name: 200 for name in names}
    states["p2b_int8"] = 100
    states["byte_matched_exact"] = 100
    states["countsketch_exact"] = 99
    states["raw_feature_ridge"] = 10
    group = {
        "records": [
            {
                "accuracy_percent": accuracies,
                "state": {
                    name: {
                        "total_persistent_bytes": states[name],
                        "solver_relative_residual": 0.0,
                    }
                    for name in names
                },
            }
        ],
        "encoding_seconds": 1.0,
        "analytic_update_seconds": {
            name: (0.5 if name == "countsketch_exact" else 1.0)
            for name in names
        },
    }
    lock = m5._derive_budget_lock(config)
    summary = m5._summarize([group], config, lock)
    assert summary["pareto_dominators_of_p2b"] == ["countsketch_exact"]
    assert summary["gates"]["not_pareto_dominated"] is False


def test_m5_notebook_is_source_locked_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "--extract-train-only" in code
    assert "test.pt').exists()" in code
    assert "--require-clean-git" in code
    assert "m5_results.json" in code
    assert "files.download(archive)" in code
    locked = dict(re.findall(r"'([^']+)':'([0-9a-f]{64})'", code))
    assert "tools/experiment_runner.py" in locked
    for relative_path, expected in locked.items():
        # Colab checks out canonical LF text.  read_text() applies universal
        # newline translation, so this catches Windows-only CRLF source locks.
        canonical = (ROOT / relative_path).read_text(encoding="utf-8")
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert actual == expected, relative_path
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
