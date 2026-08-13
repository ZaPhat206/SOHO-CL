# Phase B SFT-CL negative result

Status: hypothesis falsified on the locked CIFAR-100 protocol; retained as a
documented negative result and control.

The frozen ViT cache used checkpoint SHA-256
`32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`,
seed `1993`, ten tasks, and train-only hyperparameter selection. The selected
soft configuration was `lambda=0.01`, `kappa=0.01`, `delta=0.01`; hard Fisher
selected rank `128` and `lambda=0.01`.

| Method | Final accuracy | Average incremental accuracy | Forgetting |
|---|---:|---:|---:|
| raw Ridge | 87.15 | 92.255 | 5.578 |
| Fisher soft | 87.15 | 92.255 | 5.578 |
| confusion Fisher soft | 87.15 | 92.255 | 5.578 |
| shuffled-confusion Fisher soft | 87.15 | 92.255 | 5.578 |
| Fisher hard | 87.14 | 92.252 | 5.578 |

All rows retained `5,948,192` learner-state bytes and were marked
exemplar-free. The disk feature cache was `184,804,220` bytes and is experiment
infrastructure, not learner state.

At the final task, standard Fisher had 99 eigenvalues above `1e-12`, as
expected from the `C-1` rank bound for 100 classes. Hard rank 128 therefore
retained every informative Fisher direction plus numerical-null directions.
Confusion Fisher's maximum eigenvalue was approximately `6.34e-4`, while its
median soft gain was the floor `0.1`. Solver residuals were `1e-11` to `1e-10`,
so the negative result is not evidence of solver failure.

Conclusion: full-rank Fisher rescaling under weak Ridge did not produce a
useful predictor change, and real confusion weights did not beat their shuffled
control. SFT-CL must not be claimed to improve raw Ridge. This motivates
CRT-SOHO's fixed nonlinear anchor plus complementary residual augmentation.
