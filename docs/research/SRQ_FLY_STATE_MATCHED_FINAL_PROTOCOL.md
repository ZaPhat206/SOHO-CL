# SRQ-FLY Phase 1: final state-matched control

## Question

Does P2B's accuracy at width 10,000 come only from spending more learner-state
bytes than a smaller Exact-FLY model, or does structure-preserving compression
retain useful expansion width at the same deployed tensor budget?

This is a secondary confirmation study. CIFAR-100, CUB-200-2011 and the legacy
processed ImageNet-R test splits were already consumed by earlier runs. No
result from this phase may be described as fresh held-out evidence.

## State matching before accuracy

For feature dimension `d`, Exact-FLY width `m`, synaptic degree `s` and final
class count `C`, the repository counts:

\[
S_{\rm exact}(m)=12ms+8(d+1)+4m^2+8mC+4C.
\]

The terms are sparse-CSC projection values and indices, dense float32 Gram,
float32 cross statistic, float32 classifier and class counts. For each dataset
the selected control width is the largest integer `m < 10000` satisfying

\[
S_{\rm exact}(m)\leq S_{\rm P2B}.
\]

No accuracy enters this choice. The locked results are:

| Dataset | P2B target bytes | Exact-FLY width | Exact-FLY bytes | Relative gap |
|---|---:|---:|---:|---:|
| CIFAR-100 | 97,166,236 | 4,409 | 97,163,276 | 0.00305% |
| CUB-200-2011 | 105,166,636 | 4,518 | 105,149,848 | 0.01596% |
| ImageNet-R | 105,166,636 | 4,518 | 105,149,848 | 0.01596% |

The next integer width exceeds the respective P2B budget. Runtime tensor
inventory must equal the analytic Exact-FLY byte count.

## Train-only selection

The lower-width control gets its own Ridge selection. For every dataset:

- official train only is split class-stratified into 80% development and 20%
  outer validation;
- development is again split into 80% inner fit and 20% inner validation;
- the split seed is 2025;
- three development replicate pairs are `(2025,4201)`, `(2026,4202)` and
  `(2027,4203)` for class order and projection;
- the fixed grid is `1e-3` through `1e8` in decade steps;
- selection maximizes mean inner-validation AIA, breaking ties toward the
  larger lambda;
- a grid-endpoint winner stops before test rather than silently extending the
  grid after seeing test information;
- outer validation confirms but does not reselect the candidate.

The selection runner refuses a visible `test.pt`.

## Confirmation

After all three selections and the immutable P2B final artifact are hashed into
an authorization record, Exact FLY at the byte-matched width is evaluated on
the six existing final replicate pairs `(3031,5031)` through `(3036,5036)`.
Each pair uses the same class order and projection seed as its P2B reference.

The primary new statistic is paired P2B-minus-state-matched-FLY AIA with a 95%
t interval. The already locked paired P2B-minus-Exact-FLY-10000 interval is
reported alongside it. There is no accuracy gate or accuracy-based early stop.

The combined table contains:

1. same-width Exact FLY at 10,000 from the immutable P2B artifact;
2. P2B at 10,000 from that artifact;
3. newly evaluated state-matched Exact FLY;
4. Raw Ridge from the immutable artifact.

## Interpretation

A positive paired P2B-minus-state-matched interval supports the claim that
factor compression preserves expansion capacity better than reducing width at
the same learner-state budget. It does not show that P2B is more accurate than
same-width Exact FLY. An interval containing zero is inconclusive. A negative
interval falsifies the claimed state-matched accuracy advantage.

ImageNet-R remains a legacy split with 19 cross-split duplicate contents, 18
under conflicting labels, and must not be described as content-disjoint.
