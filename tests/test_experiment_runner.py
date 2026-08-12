import json
from types import SimpleNamespace

from tools import experiment_runner


def _args(tmp_path):
    return SimpleNamespace(method="spectral_confusion_code", rank=2, ridge_lambda=.5, seed=1993,
        feature_cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "out"), dataset="CIFAR-100",
        model_name="vit_base_patch16_224", num_classes=100, num_tasks=10, device="cpu", resume=False)


def test_tiny_cache_runner_emits_required_artifacts(tmp_path):
    args = _args(tmp_path); experiment_runner.tiny(args)
    required = {"config.json", "environment.json", "metrics.json", "task_accuracies.csv", "accuracy_matrix.csv", "state_bytes.csv", "timing.csv", "code_diagnostics.json", "run.log"}
    assert required <= {path.name for path in (tmp_path / "out").iterdir()}
    assert "persistent_state_bytes" in json.loads((tmp_path / "out" / "metrics.json").read_text())


def test_cache_rejects_metadata_mismatch(tmp_path):
    args = _args(tmp_path); experiment_runner.tiny(args); args.dataset = "other"
    try: experiment_runner.validate_cache(args.feature_cache_dir, args)
    except ValueError as error: assert "metadata mismatch" in str(error)
    else: raise AssertionError("mismatched cache was accepted")


def test_resume_skips_completed_run(tmp_path, capsys):
    args = _args(tmp_path); experiment_runner.tiny(args)
    args.resume = True
    experiment_runner.run(args)
    assert "completed run already present" in capsys.readouterr().out
