# CertiFLY Q0/Q1 protocol

## Q0: mathematical and state gate

Q0 uses synthetic data only. Required tests cover deterministic symmetric
quantization, exact diagonal preservation, streaming cumulative-error bounds,
SPD and solver residual checks, classifier/logit perturbation, certified argmax,
checkpoint continuation, forbidden sample-state detection, and analytical
state-size projection at the FLY `m=10000, C=200` setting.

Q0 passes only if all targeted tests and the full repository suite pass. No
dataset result or accuracy claim is authorized by Q0.

## Q1: train-only cached-feature feasibility

Q1 uses the verified ImageNet-R training feature cache and the exact same
frozen ViT checkpoint, preprocessing, class order, task split, seed `2025`,
sparse projection, projection dimension, Top-K ratio and current-task GCV Ridge
policy as the matched exact-FLY control. The held-out `test.pt` must be renamed
out of reach before the runner starts.

Methods:

1. matched exact FLY;
2. raw-feature Ridge;
3. fixed int8 CertiFLY;
4. certified adaptive int8/int16 CertiFLY.

The bit assignment is determined only by the mathematical error budget. It is
not a validation hyperparameter. Q1 may compare two predeclared certificate
fractions (`0.05` and `0.10`) but must not expand the grid after observing the
held-out test.

Q1 gates:

- held-out test remained inaccessible;
- maximum classifier solve residual `<=1e-5`;
- cumulative Gram error bound `< selected lambda` at every task;
- selected CertiFLY validation AA is within `0.50` point of exact FLY;
- final persistent state is `<=25%` of exact FLY;
- no checkpoint contains sample-level features or WTA codes.

Failure closes this formulation on the development protocol. It does not
authorize changing seed or dataset to conceal a negative result. A held-out
evaluation requires a separate review after Q1 passes.

