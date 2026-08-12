# Repository rules

- Preserve the current SOHO and FlyCL implementations. Change them only when a change is strictly necessary and its compatibility impact is documented.
- Select every new method through its own explicit configuration; do not silently reuse SOHO/FLY defaults.
- A learner state must not retain historical samples, images, per-example embeddings, labels, or any sample-level replay tensor.
- Do not tune hyperparameters on a test set. Use a declared train-only/validation/GCV policy.
- Every comparison must use the same frozen backbone, preprocessing, class order, task split, seed, and evaluation protocol.
- At the end of every implementation or experiment phase, run its prescribed tests and report the exact commands and results.
- Do not call a method exemplar-free when its checkpoint contains sample-level data, including embeddings or labels.
- Do not start a large experiment until the preceding phase gate in `docs/EXPERIMENT_PROTOCOL.md` has passed.

See `docs/T_SOHO_SPEC.md` for the method contract and `docs/EXPERIMENT_PROTOCOL.md` for phase gates.
