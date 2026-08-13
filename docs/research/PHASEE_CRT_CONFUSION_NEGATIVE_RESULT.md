# Phase E — CRT confusion validation result

Status: Numerical Gate 0 and information/compression Gates 1–2 **PASS**;
confusion-specific Gate 3 **FAIL**. Held-out CIFAR-100 test is not authorized.

This record transcribes the train-validation output returned from the committed
Colab workflow at commit `24292f1`. No held-out test metric was used.

## Gate results

| Gate | Observed result | Decision |
|---|---:|---|
| maximum relative solver residual | `5.246588479529226e-06` | PASS (`<=1e-4`) |
| full residual gain over anchor | `+1.2282305303073997` pp | PASS (`>=0.10`) |
| selected low-rank gap from full residual | `0.05673045582241798` pp | PASS (`<=0.50`) |
| confusion gain over strongest control | `0.0` pp | **FAIL** (`<0.10`) |

Selected validation AA values were `90.719857` for the anchor,
`91.948088` for full raw residual, and `91.891357` for requested-rank-128
confusion residual. Standard Fisher obtained exactly `91.891357`; shuffled
confusion and no-residualization also obtained `91.891357` at their best
reported rank-128 configurations. Best random residual obtained `90.920111`.

At requested rank 64, confusion and standard Fisher both obtained `91.694190`;
shuffled confusion obtained `91.689746`. At requested rank 32, confusion and
standard Fisher both obtained `91.258349`, while shuffled confusion obtained
`91.259214`. Temperature did not create a consistent semantic advantage.

## Correct rank interpretation

The table emitted by the original runner displayed requested rank. The learner
caps Fisher-family rank at `min(requested_rank, D, C_seen-1)`. Therefore the
requested rank 128 is effective rank 99 at the final 100-class stage, and less
at all earlier stages. It spans the entire class-mean contrast space rather
than providing strict `r<C_seen-1` compression at the final stage.

For a connected positive class graph, standard and reweighted graph between
scatter generally share the same class-mean contrast span once all `C-1`
directions are retained. Euclidean basis changes inside that shared subspace
cannot create a distinct isotropic-Ridge predictor. The equality at rank 128
is therefore consistent with the algebra rather than evidence for confusion
semantics. Equality or near-equality at ranks 32 and 64 further falsifies the
claim on this protocol.

## What survived falsification

The residual architecture itself remains supported on train validation:

- adding the full raw residual recovered `1.228` pp over the nonlinear anchor;
- requested rank 64 was only about `0.254` pp below full residual;
- its total state was `15,003,032` bytes versus `20,330,904` for full residual;
- relative to the `14,518,680`-byte anchor, rank 64 used about 91.7% less
  additional state than the full residual branch.

These observations do not establish a held-out improvement, do not establish
superiority to raw Ridge, and do not validate confusion-aware novelty. Anchor
Ridge values `0.01`, `0.1`, and `1.0` produced identical displayed accuracy,
so feature/statistic scale sensitivity must remain visible in diagnostics.

## Decision

Do not tune confusion temperature further and do not run held-out test. The
next hypothesis is a Schur-targeted reduced-rank correction: choose the
residual coefficient subspace directly from the label signal left unexplained
after analytically eliminating the anchor. Raw Ridge must be evaluated on the
same train-validation split before this new hypothesis can pass.
