# SRQ generalization M17 — conditioning audit of the Ridge perturbation bound

Status: frozen contract, not yet executed. M17 is prediction-free and
train-only. It cannot change the recorded status of M6 or M7.

## 1. Why this milestone exists

M8 derives a Ridge-solution bound that is *conditional*:

\[
\epsilon_t=\|A_t^{-1}\Delta_t\|_2<1
\quad\Longrightarrow\quad
\frac{\|\widehat W_t-W_t\|_F}{\|W_t\|_F}\le\frac{\epsilon_t}{1-\epsilon_t}.
\]

M7 never measured \(\epsilon_t\). It measured a 16-vector randomized *action*
of \(\Delta_t\), and `SRQ_GENERALIZATION_M8_THEORY.md` says so explicitly:
*"M7 measures a randomized action of \(\Delta_t\), not \(\epsilon_t\), so it
cannot be used as a numerical upper bound without an estimate of
\(\|A_t^{-1}\|_2\)."*

The manuscript therefore currently states a bound it never evaluates, and its
claim that downstream error is amplified "through \(A_t^{-1}\)" rests on
correlation rather than measurement. A reviewer is entitled to call the
analysis decorative. M17 closes that gap by measuring the quantity the bound is
actually about.

This also connects the result to published work rather than leaving it
isolated: LoRanPAC reports that lifted RanPAC features become highly
ill-conditioned as tasks accumulate. If \(\kappa(A_t)\) grows over the stream
here too, the amplification has a named, independently reported mechanism.

## 2. What is measured

Per task \(t\), at widths 10,000 and 20,000, on the locked M6/M7 stream:

- \(\lambda_{\max}(A_t)\) by power iteration on \(A_t\);
- \(\lambda_{\min}(A_t)\) by power iteration on \(A_t^{-1}\), then inverted;
- \(\kappa_2(A_t)=\lambda_{\max}/\lambda_{\min}\);
- \(\epsilon_t=\|A_t^{-1}\Delta_t\|_2\) by power iteration on the normal
  operator \(M^\top M\) with \(M=A_t^{-1}\Delta_t\);
- the bound \(\epsilon_t/(1-\epsilon_t)\);
- the ratio of that bound to the relative weight error **already recorded by
  M7** for the same width and task.

No accuracy, no prediction, no test split.

## 3. Cost control

\(\Delta_t\) is never materialised. Both \(A_t\) and \(\Delta_t\) are
symmetric, so with \(\Delta_t v=\widehat R_t^\top(\widehat R_t v)-A_t v\),

\[
M^\top M v=\Delta_t\!\left(A_t^{-1}\!\left(A_t^{-1}\!\left(\Delta_t v\right)\right)\right),
\]

which costs two \(\Delta\)-products and two triangular solve pairs per step.
\(A_t^{-1}\) is applied through one Cholesky factor \(L\) recomputed per task.
\(A_t\) itself is formed only long enough to be factored and is then released:
every later product is taken as \(A_tv=L(L^\top v)\). The run therefore holds
\(L\) and the decoded compressed factor, and never the dense exact system or the
reconstructed quantized system. At width 20,000 in float64 the peak is about
9.6 GB, which is what makes the audit feasible on a 16 GB Tesla T4.

Reduced-precision matmul paths are disabled before any computation: on Ampere
and later, PyTorch may route float32 matmuls through TF32 (10-bit mantissa),
which would perturb \(A_t\) by order \(10^{-3}\) against a quantization signal
of order \(10^{-2}\). Turing has no TF32 path, so this changes nothing on a T4;
it is what keeps the result valid if the audit is repeated on newer hardware.
The locked settings and the device name are recorded in the artifact.

## 4. The noise floor — a precondition, not an afterthought

\(\Delta_t\) is formed as \(\widehat R_t^\top\widehat R_t-A_t\), so its own
rounding error is of order \(\varepsilon_{\text{mach}}\|A_t\|\). After applying
\(A_t^{-1}\) this leaves a floor of roughly
\(\varepsilon_{\text{mach}}\,\kappa(A_t)\) on \(\epsilon_t\). Below that floor
the power iteration is chasing arithmetic noise and will not converge.

M17 therefore reports `epsilon_noise_floor` and `epsilon_over_noise_floor` for
every record, and **gates on the measurement sitting above its own floor**. In
float32 with \(\kappa\sim10^{3}\) the floor is about \(10^{-4}\), comfortably
below the expected \(\epsilon_t\approx5\times10^{-2}\); if any width reports a
ratio near one, rerun that width with `--audit-dtype float64`.

This behaviour is covered by
`tests/test_srq_generalization_m17.py::test_audit_task_reports_consistent_condition_number`
and `::test_audit_resolves_a_real_perturbation_above_the_noise_floor`.

## 5. Expected magnitude, declared before the run

M7 records a final relative weight error of `0.048829` at width 20,000. Through
the bound, a realised relative error of \(r\) implies
\(\epsilon_t\ge r/(1+r)\), so \(\epsilon_t\gtrsim0.047\). The bound's
precondition \(\epsilon_t<1\) is therefore expected to hold with a wide margin.
Recording this expectation here, before execution, is what makes
`epsilon_below_one` a real gate rather than a formality.

## 6. Gates

| Gate | Meaning |
|---|---|
| `source_m6_identity` | M6 ZIP SHA-256 and status match the record |
| `source_m7_identity` | M7 ZIP SHA-256 and status match the record |
| `all_widths_and_tasks_complete` | 2 widths x 10 tasks, none dropped |
| `power_iteration_convergence` | every estimate met its tolerance |
| `spd_exact_system` | \(\lambda_{\min}(A_t)>0\) at every task |
| `cholesky_residual` | max relative solve residual \(\le10^{-5}\) |
| `epsilon_below_one` | the bound's precondition actually holds |
| `epsilon_resolved_above_noise_floor` | the estimate is signal, not rounding |
| `bound_dominates_measured_weight_error` | **decisive**: \(\epsilon_t/(1-\epsilon_t)\ge\) M7's recorded relative weight error, at every width and task |
| `replay_matches_m7` | this run's recomputed local factor error matches M7's recorded value to \(10^{-6}\) relative, proving the same stream was regenerated — without it, comparing a bound computed here with a weight error recorded there would be meaningless, on any hardware |
| `finite_diagnostics` | no NaN or infinity anywhere |

`bound_slack_tolerance` is `0.0` and is not to be raised. A violation of the
decisive gate would indicate an error in the derivation or in the estimator —
not a property of the method — and must be reported, not tuned away.

## 7. What a PASS licenses, and what it does not

A PASS licenses exactly three sentences in the manuscript:

1. the bound's precondition holds on the measured stream;
2. the derived bound dominates the observed classifier perturbation at every
   measured point, so the analysis is predictive rather than merely consistent;
3. \(\kappa(A_t)\) is reported over the stream, which situates the observed
   amplification relative to the ill-conditioning LoRanPAC reports.

It does **not** license: a claim that quantization error *causes* the accuracy
gap, a claim about widths or datasets not measured, any change to M6's failed
retention gate, or any re-selection of method, width or precision policy.

## 8. Execution

```bash
python tools/srq_generalization_m17.py \
  --config configs/srq_generalization_m17_conditioning_audit.json \
  --feature-cache-dir <cache without test.pt> \
  --output-dir <artifact dir> \
  --m6-artifact srq_generalization_m6_width_sweep_train_only.zip \
  --m7-artifact srq_generalization_m7_error_trajectory_train_only.zip \
  --audit-dtype float64 \
  --require-clean-git
```

Required tests before any run:

```bash
python -m pytest -q tests/test_srq_generalization_m17.py
```

Recorded at contract time: `19 passed`, and the full repository suite
`659 passed`.

The notebook is `notebooks/srq_generalization_m17_conditioning_colab.ipynb`.

## 9. Abort condition

M17 exists to strengthen the submission, never to endanger it. If the run has
not completed by **2026-09-18**, abandon it and carry the gap into the
rebuttal window instead: the manuscript already states plainly that M7
estimates an action rather than a norm, and that statement remains honest
without M17.
