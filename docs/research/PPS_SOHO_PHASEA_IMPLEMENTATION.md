# PPS-SOHO Phase A implementation

Status: implemented locally; CIFAR-100 train-only Colab pilot pending.

The new method lives under `methods/pps_soho/` and is dispatched only by
`pps_class_protected` or its `pps_standard_fd` control. Existing baseline
implementations were not modified.

## Local gate

Exact command:

```text
python -m pytest -q tests/test_pps_soho_math.py tests/test_pps_soho_learner.py tests/test_experiment_runner.py
python -m pytest -q
```

Current focused result: 27 passed. Full repository result: 110 passed. PyTorch emitted its existing sparse-CSC beta warning
and environment-level deprecation warnings; no PPS numerical warning occurred.

The first Colab execution correctly failed the numerical gate: the original
float32 Woodbury expression suffered cancellation at `lambda=0.1`, producing
relative residuals above `4.9e3`. Those validation accuracies are invalid and
must not be interpreted as a method result. The solver now uses the equivalent
orthonormal compact-subspace system and includes a float32 WTA-scale regression
test. The train-only pilot must be rerun from a fresh selection output.

Synthetic tests use explicit `torch.float64`, fixed seeds recorded in each
fixture, and `torch.testing.assert_close` tolerances from `1e-10` to `1e-12`
for exact identities. The covariance and perturbation inequalities allow
`1e-9` numerical slack.

## Colab gate

Use `notebooks/pps_soho_cifar100_train_only_colab.ipynb`. It restores or
extracts the frozen ViT feature cache, runs the local tests, executes the
committed 1024D train-only search, displays one compact progress line per
candidate, and downloads the selection artifact. It must not evaluate
`test.pt` or be reported as held-out accuracy.
