"""M13 LoRanPAC equal-budget protocol and runner gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest
import torch

from tools import srq_generalization_m13 as m13


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m13_loranpac_train_only.json"
NOTEBOOK = ROOT / "notebooks/srq_generalization_m13_loranpac_colab.ipynb"
KAGGLE_NOTEBOOK = ROOT / "notebooks/srq_generalization_m13_loranpac_kaggle.ipynb"


def test_m13_config_locks_train_only_equal_byte_loranpac_screen():
    config = m13._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["widths"] == [10000, 20000]
    assert config["budget_targets"] == ["p2b_int8", "adaptive_int8_fp16"]
    assert config["loranpac"]["rank_rule"] == "official_code_python_round"
    assert config["loranpac"]["paper_rank_rule"] == "ceil"
    assert config["gates"]["accuracy_gate"] is None


def test_m13_rejects_test_use_accuracy_selection_and_rank_policy_mutation(tmp_path):
    original = json.loads(CONFIG.read_text(encoding="utf-8"))
    for field in ("uses_test_set", "accuracy_based_selection"):
        broken = json.loads(json.dumps(original))
        broken[field] = True
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="train-only|nonselective"):
            m13._read_config(path)
    broken = json.loads(json.dumps(original))
    broken["loranpac"]["rank_cap_selection_signal"] = "best_validation_accuracy"
    path = tmp_path / "rank.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="source-locked"):
        m13._read_config(path)


def test_m13_source_loader_requires_filename_archive_member_hash_and_status(tmp_path):
    result = {"status": "PASS_SOURCE"}
    payload = json.dumps(result).encode()
    artifact = tmp_path / "source.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("result.json", payload)
    lock = {
        "artifact_filename": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "result_member": "result.json",
        "result_sha256": hashlib.sha256(payload).hexdigest(),
        "required_status": "PASS_SOURCE",
    }
    assert m13._load_source(lock, artifact) == result
    lock["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="embedded result"):
        m13._load_source(lock, artifact)


def _small_sources(*, dimension: int, classes: int, projection_bytes: int) -> dict:
    base = 8 * dimension * classes + 4 * classes
    per_rank = 4 * (dimension + 1)
    p2b_total = projection_bytes + base + 5 * per_rank + 3
    adaptive_total = projection_bytes + base + 7 * per_rank + 1
    source_common = {
        "selected_ridge_lambda": 10.0,
        "validation_aia_percent": {
            "exact": 90.0, "fp16_square_root": 89.9, "p2b_int8": 89.7,
        },
        "final_validation_accuracy_percent": {
            "exact": 88.0, "fp16_square_root": 87.9, "p2b_int8": 87.7,
        },
        "final_total_persistent_bytes": {
            "exact": p2b_total * 4,
            "fp16_square_root": adaptive_total * 2,
            "p2b_int8": p2b_total,
        },
    }
    adaptive = {
        "width": dimension,
        "selected_ridge_lambda": 10.0,
        "validation_aia_percent": 89.8,
        "final_validation_accuracy_percent": 87.8,
        "final_total_persistent_bytes": adaptive_total,
    }
    return {
        "m6": {"width_results": [{"width": dimension, **source_common}]},
        "m11": {"width_results": [adaptive]},
    }


def test_m13_small_unit_derives_rank_from_bytes_and_reports_two_fixed_heads():
    dimension, classes, feature_dim = 24, 6, 9
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["num_classes"] = classes
    config["ranpac"].update(
        feature_dimension=feature_dim, encode_batch_size=8,
        evaluation_batch_size=7,
    )
    projection = torch.randn(
        feature_dim, dimension, generator=torch.Generator().manual_seed(1310)
    )
    sources = _small_sources(
        dimension=dimension, classes=classes,
        projection_bytes=projection.numel() * projection.element_size(),
    )
    generator = torch.Generator().manual_seed(1311)
    features = torch.randn(72, feature_dim, generator=generator)
    labels = torch.arange(72) % classes
    training_parts = [torch.arange(0, 24), torch.arange(24, 48)]
    validation_parts = [torch.arange(48, 60), torch.arange(60, 72)]
    result = m13._run_unit(
        config=config, sources=sources, width=dimension, budget="p2b_int8",
        projection=projection, train={"features": features, "labels": labels},
        training_parts=training_parts, validation_parts=validation_parts,
        device=torch.device("cpu"),
    )
    assert result["rank_contract"]["derived_max_rank"] == 5
    assert result["final_total_persistent_bytes"] <= result["rank_contract"][
        "target_total_persistent_bytes"
    ]
    assert result["rank_contract"]["final_budget_underfill_bytes"] == 3
    assert set(result["validation_aia_percent"]) == {
        "official_ridge0", "matched_m6_ridge",
    }
    assert result["maximum_solver_relative_residual"] < 1e-5


def test_m13_notebook_is_source_locked_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "--extract-train-only" in code
    assert "test.pt').exists()" in code
    assert "--source-m6-artifact" in code
    assert "--source-m11-artifact" in code
    assert "--require-clean-git" in code
    assert "m13_results.json" in code
    assert "srq_generalization_m13_loranpac_train_only.zip" in code
    locked_paths = (
        "configs/srq_generalization_m13_loranpac_train_only.json",
        "tools/srq_generalization_m13.py",
        "methods/frontends/loranpac.py",
        "methods/frontends/ranpac.py",
        "tools/srq_generalization_m6.py",
        "tools/srq_generalization_m5.py",
        "tools/srq_generalization_m4.py",
        "tools/experiment_runner.py",
        "models/backbone.py",
        "utils/data_utils.py",
        "utils/train_utils.py",
    )
    for relative in locked_paths:
        digest = hashlib.sha256(
            (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert f"'{relative}':'{digest}'" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), str(NOTEBOOK), "exec")


def test_m13_kaggle_notebook_preserves_zip_bytes_and_train_only_boundary():
    notebook = json.loads(KAGGLE_NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "name+'.bin'" in code
    assert "--extract-train-only" in code
    assert "not (cache/'test.pt').exists()" in code
    assert "--source-m6-artifact" in code and "--source-m11-artifact" in code
    assert "'/kaggle/working/srq_generalization_m13_loranpac_train_only.zip'" in code
    assert "M13_SOURCE_COMMIT" not in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), str(KAGGLE_NOTEBOOK), "exec")
