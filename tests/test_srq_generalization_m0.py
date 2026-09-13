"""Integrity checks for the SRQ paper evidence lock and staged protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper" / "EXPERIMENTS_MANIFEST.json"
PROTOCOL = ROOT / "docs" / "research" / "SRQ_GENERALIZATION_PROTOCOL.md"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT
    )


def test_manifest_is_machine_readable_and_complete():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["manifest_id"] == "srq-fly-paper-evidence-lock-m0-v1"
    assert len(payload["repository"]["baseline_head_full"]) == 40

    evidence = payload["evidence"]
    assert len(evidence) == 19
    assert len({entry["id"] for entry in evidence}) == len(evidence)
    for entry in evidence:
        assert len(entry["artifact_sha256"]) == 64
        assert entry["test_tuning_allowed"] is False
        assert entry["paper_roles"]
        config = ROOT / entry["config"]
        assert config.is_file()
        assert _sha256(config.read_bytes()) == entry["repository_config_sha256"]
        if entry["uses_test_set"]:
            assert entry.get("caveat")


def test_frozen_method_hashes_resolve_from_baseline_commit():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    commit = payload["repository"]["baseline_head_full"]
    for path, expected in payload["repository"]["method_files"].items():
        assert _sha256(_git_blob(commit, path)) == expected


def test_state_matched_recovery_caveat_cannot_be_silently_dropped():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in payload["evidence"]
        if item["id"] == "p2b_state_matched_three_dataset"
    )
    assert entry["uses_test_set"] is True
    assert "adapter" in entry["caveat"].lower()
    assert "recovery" in entry["caveat"].lower()


def test_precision_followup_outcomes_cannot_be_silently_relabelled():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in payload["evidence"]}
    assert (
        by_id["adaptive_precision_train_only"]["status"]
        == "PASS_M11_ADAPTIVE_PRECISION_TRAIN_ONLY"
    )
    refined = by_id["same_byte_scale_refinement_train_only"]
    assert refined["status"] == "FAIL_M11B_SCALE_REFINED_INT8_TRAIN_ONLY"
    assert "0.000115" in refined["caveat"]
    assert "not rounded or relaxed" in refined["caveat"]


def test_loranpac_failure_and_closure_cannot_be_silently_relabelled():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in payload["evidence"]}
    assert (
        by_id["multiseed_equal_byte_loranpac_train_only"]["status"]
        == "FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY"
    )
    assert "No seed is excluded" in by_id[
        "multiseed_equal_byte_loranpac_train_only"
    ]["caveat"]
    assert (
        by_id["loranpac_task1_numerical_closure"]["status"]
        == "PASS_M15_LORANPAC_TASK1_CLOSURE"
    )
    assert "nor retroactively passes M14" in by_id[
        "loranpac_task1_numerical_closure"
    ]["caveat"]


def test_protocol_is_fail_closed_about_scope_and_test_use():
    text = PROTOCOL.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "does not authorize a held-out" in text
    assert "The term `universal plug-in` is prohibited" in text
    assert "cannot be overridden using test accuracy" in normalized
    assert "low-rank/streaming Ridge sketch" in text
