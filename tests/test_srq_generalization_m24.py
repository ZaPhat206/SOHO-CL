from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools import srq_generalization_m24 as m24


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "srq_generalization_m24_equal_memory_controls_train_only.json"
NOTEBOOK = ROOT / "notebooks" / "srq_generalization_m24_equal_memory_controls_colab.ipynb"


def test_m24_config_is_train_only_and_accuracy_is_not_a_gate():
    config = m24._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["gates"]["accuracy_is_not_a_gate"] is True
    gate_names = " ".join(config["gates"]).lower()
    assert "aia" not in gate_names
    assert "accuracy_retention" not in gate_names
    source = (ROOT / "tools" / "srq_generalization_m24.py").read_text(
        encoding="utf-8"
    )
    assert "load_test=False" in source
    assert "refuses a visible test.pt" in source


def test_m24_cli_supports_per_stream_handoffs():
    common = [
        "--config",
        str(CONFIG),
        "--source-m23-artifact",
        "source.zip",
        "--output-dir",
        "output",
    ]
    parsed = m24.parse_args(
        [
            "run-stream",
            *common,
            "--feature-cache-dir",
            "cache",
            "--stream-id",
            "s2025",
        ]
    )
    assert parsed.stream_id == "s2025"
    summary = m24.parse_args(["summarize", *common])
    assert summary.feature_cache_dir is None
    with pytest.raises(SystemExit):
        m24.parse_args(["run-stream", *common, "--feature-cache-dir", "cache"])


def test_m24_colab_is_pinned_train_only_and_resume_safe():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "REPO_COMMIT='b25e751aa79377ada68c66dc9a144226089f02e3'" in code
    assert "--extract-train-only" in code
    assert "--extract-test" not in code
    assert "load_test=True" not in code
    assert "test.pt').exists()" in code
    assert code.count("run_and_download_stream(") == 4  # definition + 3 streams
    assert code.index("'s2025',2025") < code.index("'s2026',2026")
    assert code.index("'s2026',2026") < code.index("'s2027',2027")
    assert code.index("'s2027',2027") < code.index("'summarize'")
    assert "srq_generalization_m24_equal_memory_controls_train_only.zip" in code
    for relative in (
        "configs/srq_generalization_m24_equal_memory_controls_train_only.json",
        "tools/srq_generalization_m24.py",
        "methods/analytic_ridge/equal_memory_controls.py",
        "tools/experiment_runner.py",
    ):
        digest = hashlib.sha256(
            (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert f"'{relative}':'{digest}'" in code


def test_m24_byte_lock_is_exact_and_maximal():
    config = m24._read_config(CONFIG)
    source = {
        "budget_lock": {"target_p2b_total_persistent_bytes": 91_880_088}
    }
    lock = m24.derive_budget_lock(config, source)
    assert lock["packed_exact_dimension"] == 5878
    assert lock["packed_exact_total_persistent_bytes"] == 91_873_540
    assert lock["packed_exact_next_dimension_bytes"] > 91_880_088
    assert lock["frequent_directions_rank"] == 1400
    assert lock["frequent_directions_total_persistent_bytes"] == 91_840_400
    assert lock["frequent_directions_next_rank_bytes"] > 91_880_088
    assert lock["locked_before_representation_encoding_or_accuracy"] is True


def _source_stream(stream_id: str, seed: int) -> dict:
    return {
        "status": "PASS_M5_EQUAL_BUDGET_TRAIN_ONLY",
        "uses_test_set": False,
        "stream_id": stream_id,
        "provenance": {"stream_seed": seed},
        "gates": {"valid": True},
    }


def test_m24_source_reader_requires_matching_archive_duplicates(tmp_path):
    config = m24._read_config(CONFIG)
    streams = {
        item["stream_id"]: _source_stream(item["stream_id"], item["seed"])
        for item in m24.STREAMS
    }
    aggregate = {
        "study_id": config["source_m23"]["study_id"],
        "status": config["source_m23"]["required_status"],
        "uses_test_set": False,
        "streams": [item["stream_id"] for item in m24.STREAMS],
        "gates": {"valid": True},
        "per_stream": streams,
    }
    artifact = tmp_path / config["source_m23"]["filename"]
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("config.json", json.dumps({"uses_test_set": False}))
        archive.writestr("m23_results.json", json.dumps(aggregate))
        for item in m24.STREAMS:
            archive.writestr(
                f"stream_{item['seed']}_results.json",
                json.dumps(streams[item["stream_id"]]),
            )
    local = copy.deepcopy(config)
    local["source_m23"]["sha256"] = m24._sha256_file(artifact)
    loaded = m24._read_source_m23(artifact, local)
    assert loaded["_artifact_sha256"] == local["source_m23"]["sha256"]
    assert set(loaded["_archived_streams"]) == {"s2025", "s2026", "s2027"}

    broken = copy.deepcopy(local)
    broken["source_m23"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        m24._read_source_m23(artifact, broken)


def test_m24_aggregate_reports_paired_results_without_accuracy_gate():
    config = m24._read_config(CONFIG)
    stream_payloads = {}
    packed_aia = (92.0, 92.1, 92.2)
    fd_aia = (91.0, 91.2, 91.4)
    srq_aia = (92.5, 92.6, 92.7)
    lock = {"target_total_persistent_bytes": 100}
    for index, stream in enumerate(m24.STREAMS):
        stream_payloads[stream["stream_id"]] = {
            "status": m24.PASS_STREAM,
            "uses_test_set": False,
            "gates": {"structural": True},
            "budget_lock": lock,
            "source_reference": {
                "p2b_int8": {"validation_aia_percent": srq_aia[index]}
            },
            "methods": {
                "packed_exact": {
                    "validation_aia_percent": packed_aia[index],
                    "final_validation_accuracy_percent": 88.0 + index,
                    "final_total_persistent_bytes": 99,
                    "analytic_update_seconds": 2.0 + index,
                    "maximum_update_peak_allocated_bytes": 200 + index,
                },
                "frequent_directions_ridge": {
                    "validation_aia_percent": fd_aia[index],
                    "final_validation_accuracy_percent": 87.0 + index,
                    "final_total_persistent_bytes": 98,
                    "analytic_update_seconds": 3.0 + index,
                    "maximum_update_peak_allocated_bytes": 300 + index,
                },
            },
        }
    source = {
        "status": config["source_m23"]["required_status"],
        "_artifact_sha256": config["source_m23"]["sha256"],
        "aggregate": {
            name: {"aia_mean": 1.0}
            for name in (
                "p2b_int8",
                "byte_matched_exact",
                "countsketch_exact",
                "full_width_exact",
            )
        },
    }
    result = m24.summarize_streams(
        config=config, source_m23=source, stream_payloads=stream_payloads
    )
    assert result["status"] == m24.PASS_STUDY
    assert result["aggregate"]["packed_exact"]["aia_mean"] == pytest.approx(92.1)
    assert result["aggregate"]["packed_exact"]["aia_std"] == pytest.approx(0.1)
    assert result["paired_comparisons"]["packed_exact"][
        "srq_aia_advantage_over_control_pp_mean"
    ] == pytest.approx(0.5)
    assert result["gates"]["accuracy_not_used_as_gate"] is True
