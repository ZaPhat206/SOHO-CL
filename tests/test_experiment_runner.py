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


def test_selection_uses_cached_training_features_not_test_set(tmp_path):
    args = _args(tmp_path); experiment_runner.tiny(args)
    args.search_methods = "raw_ridge,spectral_confusion_code"
    args.search_ranks, args.search_lambdas = "1,2", "0.1,1.0"
    args.validation_fraction, args.selection_output = .2, str(tmp_path / "selection.json")
    experiment_runner.select_config(args)
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["selection_protocol"].endswith("only")
    assert selection["best"]["uses_test_set"] is False


def test_selection_does_not_open_test_cache(tmp_path):
    args = _args(tmp_path)
    experiment_runner.tiny(args)
    test_path = tmp_path / "cache" / "test.pt"
    test_path.rename(tmp_path / "cache" / "test.hidden")
    args.search_methods = "sft_raw_ridge"
    args.search_ranks, args.search_lambdas = "1", "0.1"
    args.validation_fraction, args.selection_output = .2, str(tmp_path / "selection-no-test.json")
    experiment_runner.select_config(args)
    assert (tmp_path / "selection-no-test.json").is_file()


def test_sft_cache_runner_and_train_only_selection(tmp_path):
    args = _args(tmp_path)
    args.method = "confusion_fisher_soft"
    args.fisher_kappa, args.fisher_delta, args.fisher_scatter_epsilon = .3, .1, 1e-4
    experiment_runner.tiny(args)
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["persistent_state_bytes"] > 0
    diagnostics = json.loads((tmp_path / "out" / "code_diagnostics.json").read_text())
    assert diagnostics[-1]["effective_rank"] == 8

    args.output_dir = str(tmp_path / "selection-out")
    args.search_methods = "sft_raw_ridge,fisher_hard,confusion_fisher_soft,shuffled_confusion_fisher_soft"
    args.search_ranks, args.search_lambdas = "1,2", "0.1"
    args.search_kappas, args.search_deltas = "0.1", "0.1,0.5"
    args.validation_fraction, args.selection_output = .2, str(tmp_path / "sft-selection.json")
    experiment_runner.select_config(args)
    selection = json.loads((tmp_path / "sft-selection.json").read_text())
    assert selection["best"]["uses_test_set"] is False
    assert {candidate["method"] for candidate in selection["candidates"]} == set(args.search_methods.split(","))
    assert all(isinstance(candidate["validation_average_accuracy"], float) for candidate in selection["candidates"])


def test_crt_cache_runner_and_train_only_selection(tmp_path):
    args = _args(tmp_path)
    args.method = "crt_confusion_residual"
    args.rank = 2
    args.crt_anchor_dim, args.crt_synaptic_degree, args.crt_coding_level = 12, 3, .25
    args.crt_residual_ridge, args.crt_complement_ridge = .4, .3
    args.crt_confusion_temperature, args.crt_scatter_epsilon = .7, 1e-5
    args.crt_statistics_dtype = "float64"
    experiment_runner.tiny(args)
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["exemplar_free"] is True
    diagnostics = json.loads((tmp_path / "out" / "code_diagnostics.json").read_text())
    assert diagnostics[-1]["geometry"] == "confusion_fisher"
    assert diagnostics[-1]["residualized"] is True

    args.output_dir = str(tmp_path / "crt-selection-out")
    args.search_methods = "crt_anchor_only,crt_random_residual,crt_confusion_residual"
    args.search_ranks, args.search_lambdas = "1,2", "0.2"
    args.search_crt_residual_ridges = "0.4"
    args.search_crt_complement_ridges = "0.3"
    args.search_crt_temperatures = "0.7"
    args.validation_fraction = .2
    args.selection_output = str(tmp_path / "crt-selection.json")
    experiment_runner.select_config(args)
    selection = json.loads((tmp_path / "crt-selection.json").read_text())
    assert selection["best"]["uses_test_set"] is False
    assert {candidate["method"] for candidate in selection["candidates"]} == set(args.search_methods.split(","))
    assert all("crt_residual_ridge" in candidate for candidate in selection["candidates"])


def test_forgetting_uses_maximum_over_later_stages():
    # Task 0 forgets 25 (95 -> 70), task 1 forgets 5 (90 -> 85), so the
    # protocol mean is 15 rather than 10 from only introduction-stage scores.
    matrix = [[80.0], [95.0, 90.0], [70.0, 85.0, 92.0]]
    assert experiment_runner.forgetting_from_matrix(matrix) == 15.0
