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
    assert len(evidence) == 9
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


def test_protocol_is_fail_closed_about_scope_and_test_use():
    text = PROTOCOL.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "does not authorize a held-out" in text
    assert "The term `universal plug-in` is prohibited" in text
    assert "cannot be overridden using test accuracy" in normalized
    assert "low-rank/streaming Ridge sketch" in text
