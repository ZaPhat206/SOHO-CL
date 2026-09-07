# SRQ-FLY manuscript workspace

This directory contains a Markdown-first research draft and an LNCS-formatted
projection of the same study. The Markdown files remain the evidence and claim
ledgers; the LaTeX file controls presentation only.

- `SRQ_FLY_DRAFT.md`: working manuscript;
- `SRQ_FLY_LNCS_DRAFT.tex`: Springer LNCS-formatted working manuscript with
  placeholder author metadata; compile it only with the official `llncs.cls`
  and `splncs04.bst` supplied by Springer;
- `RESULTS_LEDGER.md`: immutable source for every reported experimental
  number and status;
- `THEORY_CHECKLIST.md`: statements, proof obligations, and prohibited
  extrapolations;
- `PROOFS.md`: full assumptions and algebraic proofs used by the draft;
- `references.bib`: verified bibliography metadata;
- `RELATED_WORK_LEDGER.md`: primary-source and novelty-boundary audit;
- `VALIDATION.md`: exact validation commands and results for this draft.

Current status is **drafting only**. Locked CIFAR-100/CUB test results and a
disclosed legacy-split ImageNet-R confirmation are available, but they are not
fresh untouched evidence and the manuscript must not be presented as
submission-ready. The bibliography is not claimed exhaustive. New literature
statements must be added to both `references.bib` and the source ledger before
submission.

The historical proposed held-out design remains recorded in
`docs/research/SRQ_FLY_HELDOUT_PROTOCOL_DRAFT.md`. That draft itself did not
authorize test extraction; later source-locked protocols and artifacts record
the evaluations that were actually authorized and completed.

Experiment caches are never paper artifacts or learner checkpoints. The paper
must keep frozen-feature/WTA caches, runtime memory, and persistent learner
state separate.
