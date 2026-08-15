# PPS-SOHO Phase A2: train-only rank scaling

Status: implementation ready; execution pending. Local focused and full-suite
gates pass (27 and 110 tests, respectively).

Phase A established that the numerically valid class-protected sketch is much
better than standard Frequent Directions at comparable state size, but remains
5.40 percentage points behind the matched cached FLY control. Phase A2 is one
bounded diagnostic for whether that gap is primarily caused by the original
rank-16/32 bottleneck. It is exploratory and must not be presented as the
preregistered Phase A result.

## Locked protocol

- Dataset/cache: the same CIFAR-100 frozen ViT cache and checkpoint identity as
  Phase A.
- Split: the same deterministic 10% stratified subset of training data.
- Methods: cached FLY, sufficient-statistic raw Ridge, standard FD, and
  class-protected PPS.
- PPS ranks: 64, 128, and 256.
- Ridge lambda: 1.0 only.
- Protection gamma: 1.0 only.
- Held-out `test.pt`: physically renamed before the runner starts.
- Candidate count: 8 (one FLY, one raw Ridge, and three ranks for each PPS
  variant).

The output records runner commit/dirty status, environment versions, class
order and hash, train/validation index hashes, and config hash. It stores no
sample-level data.

## Decision rule

- Gap to matched FLY at rank 256 at most 0.50 pp: return for explicit held-out
  authorization; do not open test automatically.
- Gap in (0.50, 1.00] pp: inconclusive; stop and review without test access.
- Gap above 1.00 pp: stop PPS-SOHO as the primary accuracy method.

Raw Ridge remains a mandatory control. Even if the FLY gate passes, a large gap
to raw Ridge prevents an accuracy-superiority claim.

## Colab

Run `notebooks/pps_soho_cifar100_rank_scaling_colab.ipynb` from top to bottom.
The notebook restores the cache from Drive when possible, prints one `UPDATE`
line per continual task, and downloads
`pps_soho_phasea2_rank_scaling.zip`. Do not run a held-out evaluation afterward.

Local verification commands:

```text
python -m json.tool notebooks/pps_soho_cifar100_rank_scaling_colab.ipynb
python -m json.tool configs/pps_soho_cifar100_rank_scaling.json
python -m pytest -q tests/test_pps_soho_math.py tests/test_pps_soho_learner.py tests/test_experiment_runner.py
python -m pytest -q
```
