"""M11b scale-refinement protocol and runner gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest
import torch

from tools import srq_generalization_m11b as m11b


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m11b_scale_refined_train_only.json"
NOTEBOOK = ROOT / "notebooks/srq_generalization_m11b_scale_refined_colab.ipynb"


def test_m11b_config_locks_same_byte_label_free_four_step_rule():
    config = m11b._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["diagnostic_widths"] == [10000, 20000]
    assert config["scale_refinement"] == {
        "initializer": "max_abs_over_127",
        "update_rule": "least_squares_scale_then_nearest_int8_assignment",
        "iterations": 4,
        "selection_signal": "current_group_values_only_no_labels_or_accuracy",
        "persistent_payload": "identical_shapes_and_dtypes_to_p2b_int8",
    }


def test_m11b_rejects_accuracy_selection_and_iteration_mutation(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["accuracy_based_selection"] = True
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="accuracy"):
        m11b._read_config(path)
    config["accuracy_based_selection"] = False
    config["scale_refinement"]["iterations"] = 5
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="preregistered"):
        m11b._read_config(path)


def test_m11b_locked_result_loader_checks_member_hash(tmp_path):
    payload = json.dumps({"status": "PASS"}).encode()
    artifact = tmp_path / "source.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("result.json", payload)
    lock = {
        "artifact_filename": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "result_member": "result.json",
        "result_sha256": hashlib.sha256(payload).hexdigest(),
        "required_status": "PASS",
    }
    assert m11b._load_locked_result(lock, artifact, "fixture")["status"] == "PASS"
    lock["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="result SHA-256"):
        m11b._load_locked_result(lock, artifact, "fixture")


def test_m11b_small_width_records_same_input_error_and_state():
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
    generator = torch.Generator().manual_seed(1112)
    features = torch.randn(72, 9, generator=generator)
    labels = torch.arange(6).repeat_interleave(12)
    train = {"features": features, "labels": labels}
    result = m11b._run_refined_width(
        config=config,
        width=24,
        ridge=100.0,
        projection=torch.randn(9, 24, generator=generator),
        train=train,
        training_parts=[torch.arange(0, 24), torch.arange(24, 48)],
        validation_parts=[torch.arange(48, 60), torch.arange(60, 72)],
        device=torch.device("cpu"),
    )
    assert len(result["records"]) == 2
    for record in result["records"]:
        assert record["refined_relative_factor_error"] <= record[
            "maxabs_same_input_relative_factor_error"
        ] + 1e-12
        assert record["factor_persistent_bytes"] > 0


def test_m11b_notebook_is_locked_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "--extract-train-only" in code
    assert "test.pt').exists()" in code
    assert "--source-m6-artifact" in code
    assert "--source-m11-artifact" in code
    assert "--require-clean-git" in code
    assert "m11b_results.json" in code
    assert "PASS_M11B_SCALE_REFINED_INT8_TRAIN_ONLY" in code
    locked_paths = (
        "configs/srq_generalization_m11b_scale_refined_train_only.json",
        "tools/srq_generalization_m11b.py",
        "methods/analytic_ridge/refined_upper.py",
        "methods/analytic_ridge/refined_backend.py",
        "tools/srq_generalization_m11.py",
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
