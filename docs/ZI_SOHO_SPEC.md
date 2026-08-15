# ZI-SOHO specification

Status: implementation contract for a train-only feasibility pilot. It does not
authorize held-out evaluation or an accuracy claim.

## Scope

ZI-SOHO keeps a frozen backbone and one fixed FlyHash/Top-K representation for
the entire class-incremental stream. Adaptation occurs only in a global
class-conditional analytic head. The learner receives no Task-ID at inference
and retains no image, sample index, historical feature, historical sparse code,
or replay label.

The working hypothesis is that a fixed WTA code is better modeled as a
zero-inflated class-conditional random vector than by retaining its full
`H x H` Ridge Gram matrix. This is a falsifiable hypothesis, not an assumption
that WTA coordinates are actually independent.

## Fixed representation

For a frozen backbone feature `x in R^D`,

```text
v = S x                    S: (H,D), fixed sparse random projection
I = topk(v, k)             k = floor(q H), fixed q
z_j = v_j if j in I else 0
a_j = 1[j in I]
```

`S`, `H`, `k`, backbone, preprocessing and seed never change after learner
creation. Dynamic OLDA, ETF transport and re-projection are outside ZI-SOHO.

## Streaming sufficient statistics

For each seen class `c` and WTA coordinate `j`, retain only:

```text
n_c       : scalar class count
A_jc      : sum_i 1[z_ij != 0]
S_jc      : sum_i 1[z_ij != 0] z_ij
U_jc      : sum_i 1[z_ij != 0] z_ij^2
```

Shapes for `C` seen classes are `n:(C,)` and `A,S,U:(H,C)`. Updates are
elementwise additions, so batch and streaming results must agree and the final
state must be invariant to task partition/order for the same labeled samples.

With fixed Jeffreys support smoothing `alpha=1/2`,

```text
p_jc = (A_jc + alpha) / (n_c + 2 alpha).
```

For active amplitudes,

```text
mu_jc    = S_jc / max(A_jc,1)
var_jc   = (U_jc - S_jc^2/max(A_jc,1)) / max(A_jc-1,1)
rho_jc   = A_jc / (A_jc + kappa)
var~_jc  = rho_jc var_jc + (1-rho_jc) var_pool_j + epsilon.
```

`var_pool` is reconstructed from pooled `A,S,U`; `kappa>0` is the only proposed
method hyperparameter searched in Phase A. `epsilon` and `alpha` are fixed
before the pilot.

## Scorers

The proposed `hurdle` scorer uses

```text
logit_c(z) = sum_{j:a_j=0} log(1-p_jc)
           + sum_{j:a_j=1} [log p_jc
             - 1/2 (log(2 pi var~_jc) + (z_j-mu_jc)^2/var~_jc)].
```

Uniform class priors are used in the balanced benchmark protocol. The
implementation precomputes the all-zero base term and evaluates only active
coordinate corrections in bounded chunks.

Required controls share the identical fixed WTA map:

- `wta_ncm`: nearest class mean in WTA space;
- `support_only`: Bernoulli support likelihood;
- `active_gaussian`: active-amplitude diagonal Gaussian without support odds;
- `hurdle`: proposed combined support/amplitude score.

Raw Ridge, matched FLY Ridge and replay SOHO remain external controls.

## Learner state and complexity

Permitted checkpoint tensors are the fixed sparse projection, `n,A,S,U`, and
bounded configuration/class metadata. Scorer parameters may be reconstructed
from these statistics and are not duplicated in persistent state.

- retained sample-level bytes: exactly zero;
- statistic state: `O(H C)` rather than the FLY Ridge Gram's `O(H^2+H C)`;
- update after WTA encoding: `O(N k)` scatter accumulation;
- inference: `O(B C k)` with coordinate chunking;
- output: logits over every seen class, with no Task-ID argument.

Experiment feature/code caches are disk infrastructure only. They are forbidden
from learner checkpoints and cannot be required to resume a saved learner.

## Claims that may be proved

1. Streaming statistics equal batch statistics for a fixed WTA map.
2. The final statistics are invariant to sample/task partition and order up to
   floating-point reduction error.
3. Checkpoint state is independent of the historical sample count `N` and has
   `O(HC)` statistic storage.
4. The scorer is the Bayes rule under the stated conditionally independent
   hurdle model.
5. Save/load preserves logits without replay.

Bayes optimality is conditional. Exact Top-K creates dependence because every
sample activates exactly `k` coordinates; Gaussian active amplitudes are also
subject to selection bias. These are explicit limitations and empirical
falsifiers.

## Phase A train-only gate

The first real-data run uses one deterministic stratified training-validation
split and never opens held-out features. It uses the locked `H=10000`, degree
`300`, coding level `0.3` fixed map for both ZI-SOHO and the matched FLY
fidelity control. FLY retains its declared current-task GCV policy over
exponents `[6,10)`; only `kappa` may be searched for the proposed amplitude
models. Raw-Ridge lambda, smoothing and representation parameters are locked
before the run. It must report exact runner/config/cache identity.

ZI-SOHO advances only if all hold:

1. finite logits and exact aggregate-state audit;
2. proposed hurdle validation AA exceeds raw Ridge by at least `0.20` pp;
3. it is within `0.50` pp of matched FLY, or exceeds FLY;
4. it exceeds both support-only and active-Gaussian controls by `0.10` pp;
5. persistent state is at most `15%` of matched FLY state;
6. held-out test remains physically unavailable to the selection runner.

A failed gate stops this accuracy track. CIFAR-100 results are exploratory
because its held-out split has already informed earlier project decisions; any
confirmatory paper claim requires a separately preregistered untouched split.
