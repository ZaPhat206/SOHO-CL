"""M6 gates for the train-only SRQ width sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest
import torch

from methods.analytic_ridge import persistent_tensor_bytes
from tools import srq_generalization_m6 as m6


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "srq_generalization_m6_width_sweep_train_only.json"
NOTEBOOK = ROOT / "notebooks" / "srq_generalization_m6_width_sweep_colab.ipynb"


def test_m6_config_locks_train_only_nonselective_width_sweep():
    config = m6._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["widths"] == [2000, 4000, 6000, 8000, 10000, 15000, 20000]
    assert config["ridge_selection"]["policy"] == (
        "per_width_train_only_calibration_shared_across_backends"
    )


def test_m6_rejects_test_use_accuracy_selection_and_bad_widths(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for field in ("uses_test_set", "accuracy_based_selection"):
        broken = dict(config)
        broken[field] = True
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="train-only|select widths"):
            m6._read_config(path)
    broken = json.loads(CONFIG.read_text(encoding="utf-8"))
    broken["widths"] = [2000, 1000, 2000]
    broken["ranpac"]["maximum_expand_dimension"] = 2000
    path = tmp_path / "widths.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="widths"):
        m6._read_config(path)


@pytest.mark.parametrize("width", [17, 31])
def test_m6_symbolic_state_matches_real_backend_layout(width):
    config = m6._read_config(CONFIG)
    config["num_classes"] = 5
    config["ranpac"]["feature_dimension"] = 7
    lock = m6._state_lock(config, width)
    backends = m6._backend_group(config, width, 100.0, torch.device("cpu"))
    codes = torch.randn(25, width, generator=torch.Generator().manual_seed(width))
    labels = torch.arange(25) % 5
    projection = torch.randn(7, width)
    for name, backend in backends.items():
        backend.update(codes, labels)
        tensors = {"projection": projection}
        tensors.update(backend.persistent_tensors())
        assert persistent_tensor_bytes(tensors) == lock["total_persistent_bytes"][name]


def test_m6_summary_exposes_scaling_and_locked_gates():
    config = m6._read_config(CONFIG)
    results = []
    for width in config["widths"]:
        results.append(
            {
                "width": width,
                "validation_aia_percent": {
                    "exact": 90.0,
                    "fp16_square_root": 89.99,
                    "p2b_int8": 89.90,
                },
                "final_total_persistent_bytes": {
                    "exact": width * width * 4,
                    "fp16_square_root": width * width * 2,
                    "p2b_int8": width * width,
                },
                "fp16_validation_aia_loss_pp": 0.01,
                "p2b_validation_aia_loss_pp": 0.10,
                "maximum_solver_relative_residual": 1e-6,
            }
        )
    summary = m6._summarize(results, config)
    assert all(summary["gates"].values())
    for slope in summary["total_state_log_log_slope"].values():
        assert slope == pytest.approx(2.0)


def test_m6_merges_sequential_method_records_without_changing_semantics():
    names = ("exact", "fp16_square_root", "p2b_int8")
    groups = {}
    for offset, name in enumerate(names):
        groups[name] = {
            "records": [
                {
                    "task": task,
                    "accuracy_percent": {name: 90.0 + offset + task},
                    "state": {
                        name: {
                            "total_persistent_bytes": 100 + offset + task,
                            "solver_relative_residual": 1e-6,
                        }
                    },
                }
                for task in (1, 2)
            ],
            "analytic_update_seconds": {name: 1.0 + offset},
            "encoding_seconds": 0.1 + offset,
        }
    state_lock = {
        "width": 10,
        "projection_bytes": 40,
        "quadratic_or_factor_bytes": {name: 50 + i for i, name in enumerate(names)},
    }
    summary = m6._width_summary(
        groups, {"selected_ridge_lambda": 100.0}, state_lock
    )
    assert summary["records"][1]["accuracy_percent"] == {
        "exact": 92.0,
        "fp16_square_root": 93.0,
        "p2b_int8": 94.0,
    }
    assert summary["analytic_update_seconds"]["p2b_int8"] == 3.0
    assert summary["representation_encoding_seconds"]["exact"] == 0.1


def test_m6_notebook_is_source_locked_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "--extract-train-only" in code
    assert "test.pt').exists()" in code
    assert "--require-clean-git" in code
    assert "m6_results.json" in code
    assert "width_sweep.csv" in code
    assert "width_sweep_accuracy_state.svg" in code
    assert "width_sweep_update_time.svg" in code
    assert "files.download(archive)" in code
    locked = dict(re.findall(r"'([^']+)':'([0-9a-f]{64})'", code))
    assert "tools/srq_generalization_m6.py" in locked
    for relative_path, expected in locked.items():
        canonical = (ROOT / relative_path).read_text(encoding="utf-8")
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert actual == expected, relative_path
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
