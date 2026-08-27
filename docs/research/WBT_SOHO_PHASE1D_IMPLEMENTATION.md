# WBT-SOHO Phase 1D implementation

## Scope

Phase 1D is a train-only feasibility gate for **WTA-aware Boundary Transport
SOHO (WBT-SOHO)**. It is isolated in `methods/wbt_soho/` and
`tools/wbt_soho_phase1d.py`. Existing SOHO, FLY, SRQ and MARS implementations
are unchanged. The phase does not implement a paper-scale experiment and must
not open a test cache.

The locked predecessor is the negative MARS Phase-1C artifact with SHA-256
`602922b5860f64b89e7ba00c0ec95679d2fae735502b1c32a3695082cc22a970`.
Phase 1C selected tangent rank 16 but left a 1.257 pp gap to empirical replay
and reduced hard-WTA statistic error by only 2.37%. Phase 1D tests whether the
missing information is non-Gaussian residual shape plus low-margin WTA
boundary coverage.

## Method

For a current class `j`, current-task samples are normalized and mapped to the
tangent space at their class direction:

\[
r_{j,i}=\log_{u_j}(x_{j,i})-\bar r_j.
\]

For an old class `c`, the residual is whitened in the current class sketch and
colored by the stored old-class sketch:

\[
\widehat r_{c\leftarrow j,i}
=B_c\Lambda_c^{1/2}(\Lambda_j+\delta I)^{-1/2}B_j^\top r_{j,i}
+D_c^{1/2}D_j^{-1/2}r_{j,i}^{\perp}.
\]

The pseudo feature is `Exp_uc(r_hat)`. The source class is the nearest current
class in the current hard-WTA code geometry. For a locked fraction of rows,
the runner searches a fixed geodesic alpha grid toward that enemy and selects
the old-dominant candidate with the smallest gap between WTA order statistics
`k` and `k+1`. Alpha zero is included, so boundary movement cannot increase
the selected row's Top-K gap when a feasible candidate exists.

Pseudo features exist only during one update. Their weighted codes rebuild the
old contribution to `G` and `Q`; they are then discarded. The arriving task is
encoded exactly and discarded after its aggregate statistics are updated.

## Persistent learner state

The new learner stores only:

- sparse random projection and active SOHO rotation;
- dense analytic `G`, `Q` and classifier;
- streaming class counts, sums, squared sums and pooled scatter;
- per-class direction, tangent mean, rank-16 basis/eigenvalues and exact
  diagonal residual.

No raw image, historical feature, historical label, current residual, pseudo
feature or WTA code is persistent. Tensor leading dimensions depend only on
feature dimension, expansion dimension, rank or seen-class count.

The dense Phase-1D Gram is deliberate: SRQ is excluded until the accuracy and
fidelity gate passes. Persistent tensor bytes are not peak runtime memory.

## Mathematical invariants

1. A full-rank whitening/coloring map matches the declared target covariance
   up to floating-point tolerance.
2. If `gap_k(a) > 2 ||delta_a||_inf`, the Top-K support is unchanged.
3. Boundary candidates remain closer to the old prototype code than to the
   selected current enemy prototype code.
4. All analytic classifiers use Cholesky solve; no explicit inverse is formed.
5. Inference has no `task_id` argument and scores all seen classes.
6. The exact oracle is computed once per split/seed. Its per-task `G,Q` tensors
   are experiment cache, not learner state, and are reused by all candidates.
7. Selection and outer validation refuse a visible `test.pt`.

## Falsifiable hypotheses and gates

The proposed method is compared with exact feature replay, tangent Gaussian,
mean-shift empirical residuals, covariance-colored residuals and a
shuffled-enemy boundary control. Hyperparameters are selected only on inner
training validation; outer training validation is a gate, not a second tuning
set.

Phase 1D passes only if all conditions hold:

- WBT closes at least 50% of the tangent-to-oracle AIA gap;
- WBT reduces combined `G,Q` relative error by at least 10% versus tangent
  Gaussian;
- WBT exceeds shuffled-enemy transport by at least 0.20 pp AIA;
- its remaining oracle gap is at most 0.75 pp;
- its AIA gain over tangent Gaussian is positive in every development
  replicate;
- old-prototype dominance is at least 99%, solver residual is at most `1e-4`,
  state is sample-free and test remains hidden.

Failure means the empirical-residual/boundary hypothesis is rejected on this
protocol. Do not tune the gate, open test data, add SRQ, or move to CUB or
ImageNet-R after failure.

## Local correctness command

```text
python -m pytest -q tests/test_wbt_soho_math.py tests/test_wbt_soho_learner.py tests/test_wbt_soho_phase1d.py
```

At implementation time this command reports `11 passed` on synthetic data.

The complete repository suite was also run:

```text
python -m pytest -q
```

Result: `351 passed, 20 warnings in 49.61s`. The warnings are existing PyTorch
JIT deprecation and sparse CSC/invariant notices; no WBT-SOHO test failed.
