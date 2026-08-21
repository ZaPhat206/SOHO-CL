# SRQ-FLY manuscript workspace

This directory contains a venue-neutral research draft. It is intentionally
Markdown-first because no submission venue or official LaTeX template has
been selected.

- `SRQ_FLY_DRAFT.md`: working manuscript;
- `RESULTS_LEDGER.md`: immutable source for every reported experimental
  number and status;
- `THEORY_CHECKLIST.md`: statements, proof obligations, and prohibited
  extrapolations;
- `PROOFS.md`: full assumptions and algebraic proofs used by the draft;
- `references.bib`: verified bibliography metadata;
- `RELATED_WORK_LEDGER.md`: primary-source and novelty-boundary audit;
- `VALIDATION.md`: exact validation commands and results for this draft.

Current status is **drafting only**. CIFAR-100/CUB held-out and legacy-split
ImageNet-R results are
not available, and the manuscript must not be presented as submission-ready.
The bibliography is not claimed exhaustive. New literature statements must be
added to both `references.bib` and the source ledger before conversion to a
venue template.

The proposed held-out design is recorded in
`docs/research/SRQ_FLY_HELDOUT_PROTOCOL_DRAFT.md`. It is explicitly blocked and
does not authorize CIFAR-100, CUB or ImageNet-R test extraction.

Experiment caches are never paper artifacts or learner checkpoints. The paper
must keep frozen-feature/WTA caches, runtime memory, and persistent learner
state separate.
