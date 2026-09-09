"""M11 adaptive-precision protocol and runner gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest
import torch

from tools import srq_generalization_m11 as m11


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m11_adaptive_precision_train_only.json"
NOTEBOOK = ROOT / "notebooks/srq_generalization_m11_adaptive_precision_colab.ipynb"


def test_m11_config_locks_label_free_quarter_budget_and_no_test_use():
    config = m11._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["diagnostic_widths"] == [10000, 20000]
    assert config["adaptive"]["budget_fraction_between_int8_and_fp16"] == 0.25
    assert config["adaptive"]["selection_signal"].endswith(
        "no_labels_or_accuracy"
    )


def test_m11_rejects_accuracy_selection_and_policy_mutation(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["accuracy_based_selection"] = True
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="accuracy"):
        m11._read_config(path)
    config["accuracy_based_selection"] = False
    config["adaptive"]["budget_fraction_between_int8_and_fp16"] = 0.5
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="preregistered"):
        m11._read_config(path)


def test_m11_source_loader_requires_isolated_m6_retention_failure(tmp_path):
    result = {
        "status": "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY",
        "gates": {
            "all_widths_complete": True,
            "p2b_accuracy_retention": False,
            "solver_residual": True,
        },
    }
    payload = json.dumps(result).encode()
    artifact = tmp_path / "source.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("m6_results.json", payload)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["source_m6"].update(
        artifact_filename=artifact.name,
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        result_sha256=hashlib.sha256(payload).hexdigest(),
    )
    loaded = m11._load_source(config, artifact)
    assert loaded["gates"]["p2b_accuracy_retention"] is False

    result["gates"]["solver_residual"] = False
    payload = json.dumps(result).encode()
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("m6_results.json", payload)
    config["source_m6"].update(
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        result_sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(ValueError, match="isolated"):
        m11._load_source(config, artifact)


def test_m11_small_adaptive_width_records_budget_error_and_state():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["ranpac"].update(
        feature_dimension=9,
        encode_batch_size=8,
        evaluation_batch_size=7,
    )
    config["p2b"].update(
        block_size=6,
        group_size=5,
        update_panel_size=7,
        quantization_batch_blocks=3,
    )
    generator = torch.Generator().manual_seed(1111)
    features = torch.randn(72, 9, generator=generator)
    labels = torch.arange(6).repeat_interleave(12)
    train = {"features": features, "labels": labels}
    training_parts = [torch.arange(0, 24), torch.arange(24, 48)]
    validation_parts = [torch.arange(48, 60), torch.arange(60, 72)]
    projection = torch.randn(9, 24, generator=generator)
    result = m11._run_adaptive_width(
        config=config,
        width=24,
        ridge=100.0,
        projection=projection,
        train=train,
        training_parts=training_parts,
        validation_parts=validation_parts,
        device=torch.device("cpu"),
    )
    assert len(result["records"]) == 2
    for record in result["records"]:
        assert record["factor_persistent_bytes"] <= record[
            "factor_budget_ceiling_bytes"
        ]
        assert record["adaptive_relative_factor_error"] <= record[
            "all_int8_relative_factor_error"
        ]
        assert record["selected_fp16_blocks"] > 0
        assert len(record["precision_mask_sha256"]) == 64


def test_m11_notebook_is_source_locked_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "--extract-train-only" in code
    assert "test.pt').exists()" in code
    assert "--source-m6-artifact" in code
    assert "--require-clean-git" in code
    assert "m11_results.json" in code
    assert "PASS_M11_ADAPTIVE_PRECISION_TRAIN_ONLY" in code
    locked_paths = (
        "configs/srq_generalization_m11_adaptive_precision_train_only.json",
        "tools/srq_generalization_m11.py",
        "methods/analytic_ridge/adaptive_upper.py",
        "methods/analytic_ridge/backends.py",
        "methods/analytic_ridge/__init__.py",
        "tools/srq_generalization_m6.py",
        "tools/srq_generalization_m5.py",
        "tools/srq_generalization_m4.py",
        "tools/experiment_runner.py",
        "models/backbone.py",
        "utils/data_utils.py",
        "utils/train_utils.py",
    )
    for relative_path in locked_paths:
        canonical = (ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(canonical).hexdigest()
        assert f"'{relative_path}':'{digest}'" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
