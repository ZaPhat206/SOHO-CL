import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import torch

from tools import srq_generalization_m9 as m9


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m9_repeated_systems_train_only.json"
NOTEBOOK = ROOT / "notebooks/srq_generalization_m9_repeated_systems_colab.ipynb"


def test_m9_locked_config_is_balanced_and_uses_priority5():
    config, source_path, source = m9._read_config(CONFIG)
    assert config["repetitions"] == 4
    assert source_path == (
        ROOT / "configs/srq_fly_priority5_cifar100_whole_process_memory.json"
    ).resolve()
    assert source["dataset"] == "CIFAR-100"
    first = [order[0] for order in config["method_orders"]]
    assert first.count("exact_fly_10000") == 2
    assert first.count("srq_fly_p2b_10000") == 2


def test_m9_rejects_unbalanced_order(tmp_path):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["method_orders"] = [
        ["exact_fly_10000", "srq_fly_p2b_10000"] for _ in range(4)
    ]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="balanced"):
        m9._read_config(path)


def test_checkpoint_measurement_records_real_torch_save_and_removes_file(tmp_path):
    class Learner:
        def state_dict(self):
            return {"weights": torch.arange(32, dtype=torch.float32)}

    path = tmp_path / "checkpoint.pt"
    size = m9._serialize_checkpoint(torch, Learner(), path)
    assert size > 32 * 4
    assert not path.exists()


def test_sample_stats_uses_sample_standard_deviation():
    row = m9._sample_stats([1.0, 2.0, 3.0, 4.0])
    assert row["mean"] == 2.5
    assert row["sample_std"] == pytest.approx(1.2909944487358056)
    assert row["minimum"] == 1.0
    assert row["maximum"] == 4.0


def test_method_metrics_separates_state_allocator_nvml_and_time():
    result = {
        "persistent_state_bytes": 100,
        "serialized_checkpoint_bytes": 120,
        "torch_cuda_stages": {
            "backbone_load": {"seconds": 1.0},
            "feature_extraction": {"seconds": 2.0},
            "analytic_update": {
                "seconds": 3.0,
                "peak_allocated_bytes": 400,
                "peak_reserved_bytes": 500,
            },
            "final_probe": {"seconds": 4.0},
            "checkpoint_serialization": {"seconds": 5.0},
        },
    }
    monitor = {
        "peak_worker_process_bytes": 700,
        "stage_peaks": {"analytic_update": {"process_bytes": 600}},
    }
    metrics = m9._method_metrics(result, monitor)
    assert metrics["persistent_state_bytes"] == 100
    assert metrics["serialized_checkpoint_bytes"] == 120
    assert metrics["torch_analytic_peak_allocated_bytes"] == 400
    assert metrics["nvml_analytic_peak_process_bytes"] == 600
    assert metrics["nvml_whole_process_peak_process_bytes"] == 700
    assert metrics["total_measured_stage_seconds"] == 15.0


def test_svg_writer_creates_valid_vector_figure(tmp_path):
    values = {
        method: {"mean": mean, "sample_std": 1.0, "minimum": 1.0, "maximum": 3.0}
        for method, mean in zip(m9.METHODS, (10.0, 5.0))
    }
    path = tmp_path / "figure.svg"
    m9._svg_bars(path, "Title", [("Metric", values)], "MiB")
    root = ET.parse(path).getroot()
    assert root.tag.endswith("svg")


def test_m9_source_and_runbook_enforce_train_only_scope():
    source = (ROOT / "tools/srq_generalization_m9.py").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/research/SRQ_GENERALIZATION_M9_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    assert "test_features_materialized" in source
    assert '"uses_test_set": False' in source
    assert "train=False" not in source
    assert "No gate constrains the variance" in runbook
    assert "temporary serialized checkpoint bytes" in runbook


def test_m9_notebook_is_source_locked_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "train=False" not in code
    assert "uses_test_set':False" in code
    assert "PASS_M9_REPEATED_SYSTEMS_TRAIN_ONLY" in code
    assert "35-50 minutes" in code
    compile(code, str(NOTEBOOK), "exec")

    expected = {
        "configs/srq_generalization_m9_repeated_systems_train_only.json":
            "8d49d927e9adf532f1c2536fef1a2ab6b24f04febdd3da10c1ed736c648a0787",
        "tools/srq_generalization_m9.py":
            "94e190cf664f716b7d5952044a2782a1548acd26b6d104c81b134a08aef48482",
        "tools/srq_fly_priority5_memory.py":
            "7ae9397d3e26d8eeec03ad13b76adcbb3f778797d6d587cf64b4f8c8fdea2c94",
        "models/backbone.py":
            "90f70c9a2b16e4435e6e348ba701083a17de830d9a3d9ba080695e23d333f58b",
        "utils/data_utils.py":
            "cad262c013dbbd85c6bcd790b9276882ba1ff0a15b2bae57bff5a20293e9e5d8",
    }
    for path, digest in expected.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
        assert f"'{path}':'{digest}'" in code
