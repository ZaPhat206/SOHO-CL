"""M3 gates for the source-locked generic FLY regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools import srq_generalization_m3 as m3


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "srq_generalization_m3_fly_regression.json"
NOTEBOOK = ROOT / "notebooks" / "srq_generalization_m3_fly_regression_colab.ipynb"


def _config():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["num_classes"] = 6
    config["num_tasks"] = 3
    config["representation"] = {
        "expand_dim": 24,
        "synaptic_degree": 4,
        "coding_level": 0.25,
        "encode_batch_size": 8,
        "evaluation_batch_size": 7,
    }
    config["p2b"].update(
        block_size=6,
        group_size=5,
        update_panel_size=7,
        quantization_batch_blocks=3,
    )
    return config


def _synthetic_stream():
    config = _config()
    generator = torch.Generator().manual_seed(2025)
    features = torch.randn(60, 7, generator=generator)
    labels = torch.tensor([index % 6 for index in range(60)])
    projection_learner = m3._new_learners(config, 7, None, "cpu")[1]
    codes = projection_learner.encode(features)
    active = int(config["representation"]["expand_dim"] * 0.25)
    values, indices = torch.topk(codes, active, dim=1)
    train = {"features": features, "labels": labels}
    training_parts = []
    validation_parts = []
    for task in range(3):
        task_indices = torch.where((labels == 2 * task) | (labels == 2 * task + 1))[0]
        training_parts.append(task_indices[:14])
        validation_parts.append(task_indices[14:])
    return (
        config,
        train,
        indices.to(torch.int16),
        values,
        projection_learner.flyhash.projection_matrix,
        training_parts,
        validation_parts,
    )


def test_m3_config_locks_p2b_and_refuses_test_use():
    config = m3._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["p2b"]["storage_mode"] == "int8"
    assert config["p2b"]["update_backend"] == "blocked_qr"
    assert config["p2b"]["quantization_backend"] == "streaming"


def test_m3_rejects_changed_method_or_relaxed_identity(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["p2b"]["storage_mode"] = "float16"
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen P2B"):
        m3._read_config(path)
    config["p2b"]["storage_mode"] = "int8"
    config["gates"]["require_tensor_identity"] = False
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="tensor identity"):
        m3._read_config(path)


def test_m3_synthetic_end_to_end_has_exact_legacy_generic_identity():
    (
        config,
        train,
        code_indices,
        code_values,
        projection,
        training_parts,
        validation_parts,
    ) = _synthetic_stream()
    result = m3._run_tasks(
        config=config,
        train=train,
        code_indices=code_indices,
        code_values=code_values,
        projection=projection,
        training_parts=training_parts,
        validation_parts=validation_parts,
        device=torch.device("cpu"),
    )
    assert all(result["gates"].values())
    assert result["summary"]["minimum_prediction_agreement"] == 1.0
    assert result["summary"]["maximum_relative_logit_error"] == 0.0
    assert all(
        record[method]["legacy_persistent_state_bytes"]
        == record[method]["generic_persistent_state_bytes"]
        for record in result["records"]
        for method in ("exact", "p2b")
    )


def test_m3_notebook_is_train_only_source_locked_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "--extract-train-only" in code
    assert "--code-cache-dir" in code
    assert "test.pt').exists()" in code
    assert "files.download(archive)" in code
    assert "WTA_CACHE_DIR,bundle" not in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
