# CARS-FLY Phase A: train-only feasibility protocol

Status: preregistered implementation target. Held-out test evaluation is not
authorized by this document.

## Purpose

Phase A asks whether adaptive conditional Schur rank gives a reproducible
accuracy-memory improvement over fixed-rank and geometry controls. It uses only
a stratified validation subset of the training split. The held-out feature file
must be physically hidden while selection runs.

The first research run must use a dataset whose held-out split has not been used
to tune CARS-FLY. CIFAR-100 may be used for synthetic/integration debugging only
because its held-out results have already informed earlier branches. The planned
new-study seed is `2025`.

## Shared contract

All rows use the same frozen checkpoint, preprocessing, feature cache identity,
class order, task split, seed, compact FlyHash projection, Top-K semantics, and
global seen-class evaluation. Hyperparameters are selected from training data
only and are locked before any held-out evaluation.

## Required controls

1. raw-feature streaming Ridge;
2. compact anchor-only Ridge;
3. full raw-residual block Ridge (headroom oracle within the joint view);
4. fixed-rank Schur residual;
5. adaptive-rank CARS-FLY;
6. random residual at the selected rank;
7. standard-Fisher residual at the selected rank;
8. confusion and shuffled-confusion residual controls;
9. matched current FLY-CL;
10. a truncated-SVD/LoRanPAC-style control when its implementation and protocol
    are available.

## Phase A gates

Correctness gates, all mandatory:

- synthetic streaming and batch statistics agree;
- reconstructed backfill and materialized-row oracle agree to `1e-5` or better;
- full effective rank matches full raw-residual logits to `1e-5` or better;
- maximum linear-system relative residual is at most `1e-5`;
- learner checkpoint/state audit finds no sample-level tensor;
- held-out test remains physically unavailable during selection.

Research gates:

- full joint residual improves compact anchor validation AA by at least `0.10`
  percentage point;
- CARS-FLY improves the strongest random/Fisher/confusion/shuffled control by at
  least `0.20` point at a matched rank/state budget;
- CARS-FLY either lies within `1.0` point of matched FLY while using at most
  `10%` of its persistent state, or establishes another preregistered Pareto
  point before held-out evaluation;
- selected rank is not equal to the maximum at every task. Otherwise the
  adaptive-budget claim is unsupported.

Failure of any correctness gate blocks all experiments. Failure of a research
gate stops CARS-FLY as the primary proposal; it does not authorize post-hoc test
tuning.

## Required artifacts

The runner must emit exact config/command, git commit and dirty status,
environment, checkpoint/cache hashes, class-order and split hashes, every
candidate and per-task diagnostic, solver residuals, selected ranks, captured
and tail energies, persistent state bytes, runtime, gate decision, and an
explicit `uses_test_set=false` field.
