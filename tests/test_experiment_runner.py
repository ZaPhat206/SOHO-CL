import json
from types import SimpleNamespace

import pytest
import torch

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


def test_train_only_feature_cache_contains_no_heldout_tensor(tmp_path):
    args = _args(tmp_path)
    args.data_augmentation = "vit"
    cache = tmp_path / "train-only-cache"
    features = torch.randn(12, 8)
    labels = torch.arange(3).repeat_interleave(4)

    experiment_runner.save_train_cache(cache, features, labels, args, 7, "checkpoint-hash")
    train, test, metadata = experiment_runner.validate_cache(cache, args, load_test=False)

    assert test is None
    assert train["features"].shape == (12, 8)
    assert not (cache / "test.pt").exists()
    assert metadata["test_features_materialized"] is False
    assert metadata["split_sizes"] == {"train": 12, "test": 7}


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


def test_pps_cache_runner_and_train_only_selection(tmp_path):
    args = _args(tmp_path)
    args.method = "pps_class_protected"
    args.rank = 4
    args.pps_anchor_dim, args.pps_synaptic_degree, args.pps_coding_level = 12, 3, .25
    args.pps_gamma, args.pps_statistics_dtype = .7, "float64"
    experiment_runner.tiny(args)
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    diagnostics = json.loads((tmp_path / "out" / "code_diagnostics.json").read_text())
    assert metrics["exemplar_free"] is True
    assert diagnostics[-1]["geometry"] == "class_protected"
    assert diagnostics[-1]["sketch_size"] == 4
    assert diagnostics[-1]["gamma"] == .7

    # The selection path must continue to work after test.pt is hidden.
    (tmp_path / "cache" / "test.pt").rename(tmp_path / "cache" / "test.hidden")
    args.output_dir = str(tmp_path / "selection-out")
    args.search_methods = "pps_standard_fd,pps_class_protected"
    args.search_ranks, args.search_lambdas = "3,4", "0.1"
    args.search_pps_gammas = "0.5,1.0"
    args.validation_fraction = .2
    args.selection_output = str(tmp_path / "pps-selection.json")
    experiment_runner.select_config(args)
    selection = json.loads((tmp_path / "pps-selection.json").read_text())
    assert selection["best"]["uses_test_set"] is False
    assert {candidate["method"] for candidate in selection["candidates"]} == {
        "pps_standard_fd", "pps_class_protected",
    }
    assert len(selection["candidates"]) == 6
    assert all(candidate["pps_gamma"] in {None, .5, 1.0} for candidate in selection["candidates"])
    provenance = selection["run_provenance"]
    assert provenance["class_order"] == [1, 2, 0]
    assert len(provenance["class_order_sha256"]) == 64
    assert len(provenance["training_indices_sha256"]) == 64
    assert len(provenance["validation_indices_sha256"]) == 64
    assert provenance["python"]
    assert provenance["torch"]


def test_full_pps_sketch_matches_exact_cached_fly_with_identical_wta_map(tmp_path):
    args = _args(tmp_path)
    args.fly_expand_dim = args.pps_anchor_dim = 12
    args.fly_synaptic_degree = args.pps_synaptic_degree = 3
    args.fly_coding_level = args.pps_coding_level = .25
    args.pps_statistics_dtype = "float32"
    exact = experiment_runner.create_cached_learner(
        args, "cached_flycl", 6, ridge_lambda=.4, rank=0
    )
    protected = experiment_runner.create_cached_learner(
        args, "pps_class_protected", 6, ridge_lambda=.4, rank=12, pps_gamma=1.0
    )
    torch.testing.assert_close(
        exact.flyhash.projection_matrix.to_dense(),
        protected.anchor.projection_matrix.to_dense(),
        atol=0,
        rtol=0,
    )
    features = torch.randn(36, 6, generator=torch.Generator().manual_seed(71))
    labels = torch.tensor([5, 1, 8] * 12)
    exact.update(features, labels)
    protected.update(features, labels)
    torch.testing.assert_close(
        protected.predict_logits(features[:9]),
        exact.predict_logits(features[:9]),
        atol=2e-4,
        rtol=2e-4,
        check_dtype=False,
    )


def test_forgetting_uses_maximum_over_later_stages():
    # Task 0 forgets 25 (95 -> 70), task 1 forgets 5 (90 -> 85), so the
    # protocol mean is 15 rather than 10 from only introduction-stage scores.
    matrix = [[80.0], [95.0, 90.0], [70.0, 85.0, 92.0]]
    assert experiment_runner.forgetting_from_matrix(matrix) == 15.0


def test_fidelity_baselines_have_explicit_dispatch_and_internal_gcv(tmp_path):
    args = _args(tmp_path)
    args.fly_expand_dim = 16
    args.fly_synaptic_degree = 3
    args.fly_coding_level = .25
    args.fly_ridge_lower = -1
    args.fly_ridge_upper = 2
    fly = experiment_runner.create_cached_learner(
        args, "cached_flycl_fidelity", 5, ridge_lambda=999.0, rank=0
    )
    assert fly.diagnostics["ridge_policy"] == "original_current_task_gcv"
    assert fly.is_exemplar_free is True

    args.soho_expand_dim = 16
    args.soho_density = .4
    args.soho_olda_dim = 5
    args.soho_coding_level = .25
    args.soho_ridge_lower = -1
    args.soho_ridge_upper = 2
    args.soho_replay_chunk_size = 8
    args.soho_gcv_sample_size = 12
    soho = experiment_runner.create_cached_learner(
        args, "cached_soho_replay_fidelity", 5, ridge_lambda=999.0, rank=0
    )
    assert soho.diagnostics["ridge_policy"] == "current_soho_replay_sample_gcv"
    assert soho.is_exemplar_free is False


def test_fidelity_baselines_cannot_be_tested_in_external_search_grid(tmp_path):
    args = _args(tmp_path)
    experiment_runner.tiny(args)
    args.output_dir = str(tmp_path / "selection")
    args.search_methods = "cached_flycl_fidelity"
    args.search_ranks = "1"
    args.search_lambdas = "0.1"
    args.validation_fraction = .2
    args.selection_output = str(tmp_path / "forbidden.json")

    with pytest.raises(ValueError, match="locked internal GCV policy"):
        experiment_runner.select_config(args)


def test_fidelity_baselines_complete_cache_runner_and_disclose_state(tmp_path):
    for method in ("cached_flycl_fidelity", "cached_soho_replay_fidelity"):
        args = _args(tmp_path / method)
        args.method = method
        args.fly_expand_dim = args.soho_expand_dim = 16
        args.fly_synaptic_degree = 3
        args.fly_coding_level = args.soho_coding_level = .25
        args.fly_ridge_lower = args.soho_ridge_lower = -1
        args.fly_ridge_upper = args.soho_ridge_upper = 2
        args.soho_density = .4
        args.soho_olda_dim = 8
        args.soho_replay_chunk_size = 8
        args.soho_gcv_sample_size = 12

        experiment_runner.tiny(args)

        metrics = json.loads((tmp_path / method / "out" / "metrics.json").read_text())
        diagnostics = json.loads(
            (tmp_path / method / "out" / "code_diagnostics.json").read_text()
        )
        assert metrics["exemplar_free"] is (method == "cached_flycl_fidelity")
        assert all(item["selected_ridge"] in (0.1, 1.0, 10.0) for item in diagnostics)
        if method == "cached_soho_replay_fidelity":
            assert diagnostics[-1]["retained_sample_count"] == 21
            assert diagnostics[-1]["replay_required"] is True
