import json
from pathlib import Path

import pytest

from tools import crt_gate_runner, schur_locked_eval
from tests.test_crt_gate_runner import gate_args


def authorized_artifact(tmp_path):
    args = gate_args(tmp_path)
    crt_gate_runner.prepare_cache(args)
    report = crt_gate_runner.run_gates(args)
    assert report["held_out_test_authorized"] is True
    path = Path(args.output_dir) / "gate_results.json"
    return args, path, schur_locked_eval.sha256(path)


def test_authorization_locks_exact_bytes_and_selected_candidates(tmp_path):
    _, path, digest = authorized_artifact(tmp_path)
    report, lock = schur_locked_eval.authorize_gate_report(path, digest)

    assert lock["gate_results_sha256"] == digest
    assert lock["selected_proposal"] == report["selected_proposal"]
    assert lock["selected_proposal"]["method"] == "schur_residual"
    assert all(lock["authorization_checks"].values())


def test_hash_mismatch_fails_before_any_cache_is_opened(tmp_path, monkeypatch):
    args, path, _ = authorized_artifact(tmp_path)
    opened = []

    def forbidden(*call_args, **call_kwargs):
        opened.append(call_kwargs.get("load_test"))
        raise AssertionError("feature cache must not be opened")

    monkeypatch.setattr(schur_locked_eval, "validate_cache", forbidden)
    args.gate_results = str(path)
    args.gate_results_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        schur_locked_eval.run(args)
    assert opened == []


def test_failed_gate_fails_before_any_cache_is_opened(tmp_path, monkeypatch):
    args, path, _ = authorized_artifact(tmp_path)
    payload = json.loads(path.read_text())
    payload["gates"]["gate3_proposal_beats_controls"]["pass"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    opened = []
    monkeypatch.setattr(
        schur_locked_eval,
        "validate_cache",
        lambda *call_args, **call_kwargs: opened.append(call_kwargs.get("load_test")),
    )
    args.gate_results = str(path)
    args.gate_results_sha256 = schur_locked_eval.sha256(path)
    with pytest.raises(ValueError, match="all four declared validation gates"):
        schur_locked_eval.run(args)
    assert opened == []


def test_runtime_mismatch_never_opens_test_cache(tmp_path, monkeypatch):
    args, path, digest = authorized_artifact(tmp_path)
    original = schur_locked_eval.validate_cache
    opened = []

    def tracking(*call_args, **call_kwargs):
        opened.append(bool(call_kwargs.get("load_test", True)))
        return original(*call_args, **call_kwargs)

    monkeypatch.setattr(schur_locked_eval, "validate_cache", tracking)
    args.gate_results = str(path)
    args.gate_results_sha256 = digest
    args.anchor_dim += 1
    with pytest.raises(ValueError, match="gate cache mismatch|runtime configuration mismatch"):
        schur_locked_eval.run(args)
    assert True not in opened


def test_source_provenance_commit_does_not_change_feature_identity(tmp_path):
    args, path, digest = authorized_artifact(tmp_path)
    metadata_path = Path(args.feature_cache_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["git_commit"] = "a-new-code-commit-with-identical-frozen-features"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    args.gate_results = str(path)
    args.gate_results_sha256 = digest
    args.output_dir = str(tmp_path / "heldout-after-code-change")

    output = schur_locked_eval.run(args)

    assert output["test_cache_opened"] is True


def test_authorized_toy_run_uses_locked_methods_and_full_training_stream(tmp_path):
    args, path, digest = authorized_artifact(tmp_path)
    args.gate_results = str(path)
    args.gate_results_sha256 = digest
    args.output_dir = str(tmp_path / "heldout")
    output = schur_locked_eval.run(args)

    methods = {result["method"] for result in output["results"]}
    assert methods == {
        "raw_ridge", "anchor_only", "full_raw_residual", "schur_residual",
        "random_residual", "fisher_residual", "confusion_residual",
        "shuffled_confusion_residual", "confusion_no_residualization",
    }
    assert output["test_cache_opened"] is True
    assert output["hyperparameter_search_performed"] is False
    assert output["full_training_total_count"] == 90
    assert output["full_training_counts_by_task"] == [30, 30, 30]
    assert output["class_order"] == output["lock"]["source_gate_cache"]["class_order"]
    assert (Path(args.output_dir) / "locked_manifest.json").is_file()
    assert (Path(args.output_dir) / "heldout_results.json").is_file()
    bundled = Path(args.output_dir) / "authorized_gate_results.json"
    assert bundled.is_file()
    assert schur_locked_eval.sha256(bundled) == digest
