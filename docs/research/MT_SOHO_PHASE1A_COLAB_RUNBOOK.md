# MT-SOHO Phase 1A Colab runbook

Use `notebooks/mt_soho_phase1a_cifar100_colab.ipynb` on a Colab T4 GPU.

The notebook first runs synthetic correctness tests. It then restores a
training-only CIFAR feature cache when available; otherwise it downloads the
official locked ViT checkpoint and the configured CIFAR-100 Kaggle artifact
and extracts **training features only**. The locked runner performs nested
train-only selection and writes one small JSON after every candidate/replicate.

Edit only the first configuration cell. Run cells from top to bottom. Progress
lines have these meanings:

- `ANCHOR START/DONE`: fixed-WTA Ridge selection;
- `START/DONE candidate=i/16`: proposed MT grid;
- `TASK ... stage=t/10`: one continual-learning stage completed;
- `RESTORE`: an already completed unit was loaded;
- final `phase1a_pass` or `phase1a_fail`: train-only gate decision.

If Drive has insufficient space, keep `SAVE_PROGRESS_TO_DRIVE=False`; download
the final ZIP before the runtime ends. If interruption recovery matters, set it
to `True`: only small JSON outputs go to Drive, not the feature cache.

Return `mt_soho_phase1a_train_only.zip` for audit. Do not expose CIFAR test
features and do not modify the grid after seeing results.

Locked identities:

- config SHA-256: `395cccdca828ae462c58bab5371b08ea959da0df37f0b3e8b5f5fd0e0200bfbc`;
- runner SHA-256: `353b59935a8ece2b5694237f662c1e12ea8b64352522c0f093fb28a233e2c8ab`;
- protocol seed: `2025`;
- feature width: `1,000` for Phase 1A only.
