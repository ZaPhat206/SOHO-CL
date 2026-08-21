# SRQ-FLY paper-readiness boundary

Status: manuscript drafting is authorized; definitive paper claims and
held-out evaluation are not yet authorized.

## Mainline decision

The main method remains the existing groupwise-int8 square-root SRQ-FLY. Error
feedback is deliberately deferred to future work or a separately named
ablation. No learner, hyperparameter, state definition, or historical result
is changed by this decision.

## Evidence already available

1. Synthetic correctness establishes deterministic streaming updates,
   structurally positive-definite reconstructed Gram matrices, checkpoint
   continuation, task-ID-free inference, sample-free persistent state, and
   measured state-byte accounting.
2. ImageNet-R train-validation D2/D2.1 shows SRQ-FLY-10000 outperforming exact
   FLY-4518 at nearly identical state: `+0.9201` point in average incremental
   accuracy and `+1.1112` points at task 20 after the matched-width Ridge
   control is independently checked.
3. CUB D3 reports the same direction on seed 2025: `+0.2417` average and
   `+0.5890` final points over state-matched FLY.
4. CUB D4 repeats the state-matched comparison over five fresh seeds. Mean
   average and final gains are `+0.5153` and `+0.6051` points, with four of
   five average-accuracy wins and 23.27% of exact FLY-10000 state.

Historical statuses remain unchanged: D1 and D3 were formal numerical stops,
and D4 is a formal fidelity stop. The manuscript must report those facts
rather than merging substantive positive signals into a retrospective pass.

## Claims that may be drafted now

- SRQ-FLY is an exemplar-free, task-ID-free analytic continual learner whose
  persistent state contains a sparse fixed projection, a compressed
  square-root sufficient statistic, class cross-statistics/counts, and the
  global classifier.
- Storing a factor and reconstructing `R^T R` preserves positive definiteness
  by construction when the stored diagonal is positive.
- On the observed ImageNet-R and CUB train-validation protocols, SRQ provides
  a favorable empirical accuracy/state tradeoff against an exact FLY width
  selected by state accounting alone.
- Quantization changes the predictor: it is not an orthogonal change of basis,
  not exactly equivalent to full FLY, and has an observed late-stream fidelity
  limitation.

## Claims that remain prohibited

- state-of-the-art, statistically significant, or held-out generalization;
- universal superiority over FLY, SOHO, or raw Ridge;
- strict reproduction of the original FLY paper environment;
- exact prediction preservation under quantization;
- exemplar-free deployment if the shipped checkpoint includes feature or WTA
  caches;
- any benefit from error feedback, 4-bit storage, or backbone quantization.

## Draft structure

1. **Problem and audit.** Explain why feature replay violates deployable
   exemplar-free state and why direct Gram quantization can lose positive
   definiteness.
2. **Method.** Define streaming WTA statistics, square-root update,
   groupwise-int8 strict-triangle storage, exact diagonal, global Ridge solve,
   checkpoint state, and task-ID-free inference.
3. **Theory.** State SPD-by-construction, sufficient-statistic/exemplar-free
   invariants, state complexity, and a Ridge perturbation bound. Do not reuse
   Shampoo convergence results, which concern a different optimization
   process.
4. **Protocol.** Separate experiment caches, runtime memory, and persistent
   learner state; document nested selection and immutable stop gates.
5. **Results.** Present exact FLY-10000, state-matched FLY-4518, raw Ridge, and
   SRQ on ImageNet-R and CUB, including every negative result and confidence
   interval.
6. **Limitations.** Cover frozen-backbone scope, train-validation evidence,
   quantization drift, environment mismatch, and sample-level experiment
   infrastructure.

## Next evidence gate without error feedback

The immediate next activity is writing and independent protocol review, not
another adaptive search. D4 seeds `2026-2030` are now observed and cannot be
called fresh confirmation again. CUB test remains closed.

Before a held-out run, a protocol must be committed that fixes:

- the unchanged SRQ-int8 implementation and D3/D4 hyperparameters;
- datasets, class orders, task counts, seeds, backbone, checkpoint,
  preprocessing, and hardware reporting;
- exact FLY-10000, state-matched FLY-4518, raw Ridge, and existing FLY/SOHO
  controls where their state semantics are comparable;
- primary endpoints: final accuracy, average incremental accuracy, persistent
  state bytes, and the paired gain over state-matched FLY;
- prediction agreement as a reported fidelity diagnostic, preserving D4's
  formal failure rather than retroactively redefining it;
- a single-use held-out rule and a stopping rule if artifact identity,
  numerical stability, or sample-free checkpoint audits fail.

The strongest next confirmation is a locked held-out evaluation on both CUB
and ImageNet-R, followed by a second backbone or train-from-scratch setting if
the frozen-backbone claim survives. That evaluation requires separate review;
this document does not authorize it.
