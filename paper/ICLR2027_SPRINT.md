# ICLR 2027 sprint tracker

Abstract deadline **2026-09-18 AoE** (hard gate) · Paper deadline **2026-09-25 AoE**
Reviews 2026-11-05 · Discussion to 11-18 · Decision 12-16
Fallback: CVPR 2027 abstract 11-07, paper 11-12 (withdraw from ICLR first).

Target file: `paper/SRQ_ICLR2027.tex` (9 pages main + unlimited appendix).
`paper/SRQ_FLY_LNCS_DRAFT.tex` is untouched and remains the CVPR fallback.

---

## Status

### D1 — done
- [x] `SRQ_ICLR2027.tex` created; 9-page budget written into the file header.
- [x] Title reframed: *Error Accumulation Limits the Compression of Analytic
      Continual Learners*.
- [x] Abstract rewritten around the accumulating-vs-EMA distinction.
- [x] Intro rewritten: C1 finding / C2 analysis / C3 resolution, with the
      "why this is not optimizer-state quantization" paragraph on page 1.
- [x] **R1 patched**: new Related Work paragraph on square-root factorizations
      with 10 citations; Theorem 1 demoted to `Lemma 1 (standard)` with an
      explicit no-novelty statement.
- [x] LoRanPAC cited (3 places) and connected to Eq. ridge-perturbation.
- [x] `references_iclr2027.bib`: 11 -> 27 entries. 27/27 keys resolve, no orphans.
- [x] Figures 2 and 4 generated: `paper/fig/make_figures.py`.

### D1 — style pass (done)
- [x] `ridge` capitalisation normalised (was mixed Ridge/ridge).
- [x] Removed filler openers ("Figure X reports the result", "Note that").
- [x] Results paragraphs reordered to lead with the claim, not with a table
      reference: 5.1 now opens on the gate failure, 5.3 on the degradation
      being removed, 5.4 on the SPD failure.
- [x] Limitations split into four named `\paragraph` blocks so each limitation
      is findable — and so each maps to a deliverable in the rebuttal window.
- [x] Section title 5.5 made claim-bearing.
- [x] LaTeX validated: environments balanced, braces 0, math even, no dangling
      `\ref`.

### Build — working

`latexmk -pdf SRQ_ICLR2027.tex` runs clean from scratch: exit 0, 12 PDF pages,
zero undefined citations or references. The IDE build button works now.

What was wrong and what fixed it:

1. `iclr2027_conference.sty` was missing. Downloaded
   `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip` and
   extracted `iclr2027_conference.sty`, `iclr2027_conference.bst`,
   `natbib.sty`, `fancyhdr.sty` into `paper/`. The guessed filename was right,
   so no `\usepackage` change was needed.
2. MiKTeX's on-the-fly installer was hanging on a dialog, which is why the
   build appeared to freeze with a 0-byte log. Installed the missing packages
   explicitly instead: `eso-pic` (required by the ICLR sty) and `psnfss`
   (provides `times.sty`), plus booktabs, multirow, microtype, hyperref,
   xcolor, caption. If a future build hangs again, run
   `pdflatex --disable-installer` to get the real error immediately, then
   `mpm --require=<package>`.
3. The `\author` block was wrong. `iclr2027_conference.sty` substitutes
   "Anonymous authors / Paper under double-blind review" automatically while
   `\iclrfinalcopy` is commented out, so the real author details belong in
   `\author{}` and must not be replaced by a fake anonymous block. Fixed, and
   verified on the rendered page 1.
4. The official template ships its own `\documentclass`, which can confuse
   root-document detection. Moved to `paper/iclr2027_template_reference/`.

Local `natbib.sty` / `fancyhdr.sty` come from the ICLR pack and take precedence
over MiKTeX's copies. That is intended — do not delete them.

### Figures — 2 and 3 now generated from the verified artifacts

`paper/fig/make_figures.py` reads the CSVs straight out of the experiment ZIPs
and **verifies each SHA-256 against the protocol before plotting**, refusing to
run on a mismatch. No number is transcribed by hand any more, so a figure
cannot drift from the evidence. The recomputed width-20,000 gap is `0.255845`
and the state reduction `82.71%`, matching the recorded values exactly — an
independent check that the earlier hand-entered table was right.

Figure 4 still carries inline M5 values; that ZIP is not in the checkout.

Figure 3 turned out to carry a stronger result than the width comparison. At a
fixed width of 20,000, across ten tasks: local factor error `x1.10`, classifier
error `x4.3`, logit error `x5.4`, margin-certified fraction `97.2% -> 75.4%`.
Because that is measured *within one run*, it isolates accumulation with no
confound from changing the problem. FP16 shows the same compounding *shape* two
orders of magnitude lower, which matches the theory: accumulation is a property
of the recursion; precision only sets whether it matters. This is now the second
paragraph of 5.2 and the caption of Figure 3.

### LENGTH — resolved, main text fits exactly

Measured on the real build: **main text is 9 pages with 0 lines of overflow**,
Reproducibility and AI-use statements (which do not count) begin on page 10.
Build is clean: 0 overfull boxes, 0 undefined citations or references.

Cuts applied to get there, after Figure 3 pushed it 15 lines over:
Conclusion tightened; `What it costs` de-duplicated; GACL block compressed;
Related Work `Alternatives at equal memory` tightened; Method 3.1 `admissible
family` tightened; Protocol paragraph tightened; LoRanPAC sentence in 5.6
shortened; Table 3 row labels shortened to `frontend / data / backbone`, which
also fixed the one overfull box.

**There is now no headroom.** Figure 1 (teaser) is still unplaced and will need
roughly another 12 lines cut from the list below.

### LENGTH — earlier estimate, superseded

Estimated main text **~9.5 pages against a 9.0 limit**, and Figure 1 (teaser)
and Figure 3 (error trajectory) are not yet placed; they add roughly 0.6 more.
Plan for **~1.1 pages of cuts** at D10, once the real style file allows a true
compile. Ranked cut list, least damaging first:

1. `\paragraph{What it costs}` in 5.6 — keep three sentences and the headline
   ratios, move the per-metric breakdown to the appendix. (~0.25 p)
2. GACL block in 5.5 — compress to three sentences; the adapter-validation
   numbers can live in the appendix. (~0.20 p)
3. Related Work "Alternatives at equal memory" — tighten. (~0.15 p)
4. Conclusion — currently restates the whole results section; cut to five
   sentences. (~0.15 p)
5. Method 3.1 "admissible family" paragraph — tighten, it partly repeats the
   Backend Scope argument. (~0.10 p)
6. Table 3 (portability) — move the GACL block to the appendix, leaving RanPAC
   and Cars in the main table. (~0.20 p)

Do NOT cut: the direct-quantization ablation, the M6/M7 failure narrative, or
any preregistration disclosure. Those are the paper.

### M17 — conditioning audit (built, not yet run)

The gap it closes: M8 derives a bound conditional on
$\epsilon_t=\|A_t^{-1}\Delta_t\|_2<1$, and M7 never measured $\epsilon_t$ — it
measured a randomized *action* of $\Delta_t$ and says so. The manuscript
currently states a bound it never evaluates, which a reviewer can fairly call
decorative. M17 measures $\lambda_{\max}$, $\lambda_{\min}$, $\kappa(A_t)$ and
$\epsilon_t$ per task, then checks $\epsilon_t/(1-\epsilon_t)$ against the
relative weight error M7 already recorded.

Files: `configs/srq_generalization_m17_conditioning_audit.json`,
`tools/srq_generalization_m17.py`, `tests/test_srq_generalization_m17.py`,
`docs/research/SRQ_GENERALIZATION_M17_PROTOCOL.md`,
`notebooks/srq_generalization_m17_conditioning_colab.ipynb`.

Validation done locally: **17/17 M17 tests pass, 657/657 repository tests pass.**
The estimator is checked against dense ground truth — the power iteration
reproduces `torch.linalg.matrix_norm(A^{-1} Delta, ord=2)` to 1e-6 relative, and
the matrix-free $\Delta$ product matches the materialised one. Artifact
verification was exercised against the real M6/M7 ZIPs.

Two findings worth keeping:

1. **$\Delta_t$ is applied matrix-free** as $\widehat R^\top(\widehat R v)-A_tv$,
   so the quantized system is never materialised. That saves 3.2 GB at width
   20,000 and is what makes a float64 audit feasible at all.
2. **$\epsilon_t$ has an arithmetic noise floor** of about
   $\varepsilon_{\text{mach}}\kappa(A_t)$, because $\Delta_t$ is a difference of
   two nearly equal matrices. In float32 with $\kappa\sim10^3$ that floor is
   $\sim10^{-4}$; the expected signal is $\sim5\times10^{-2}$, so float32 would
   work but float64 is the safe default. The floor is reported per record and
   gated on, so the audit cannot silently measure rounding error.

Before running, pin `REPO_COMMIT` in notebook cell 1 to the commit containing
these files — the cell asserts on the placeholder.

**Abort condition: if it has not run by 2026-09-18, drop it and carry the gap
into rebuttal.** The submission must not wait on this.

### Next (D2-D3)
- [ ] Port Method/Analysis proofs into Appendix C **from `M8_THEORY.md`, not
      `PROOFS.md`** (see issue 1).
- [ ] Port byte accounting (PROOFS.md §H) and state invariant (§I) to appendix.
- [ ] Restructure results text to the 5.1->5.6 narrative (skeleton in place).
- [ ] Appendix A manifest table from `EXPERIMENTS_MANIFEST.json`.

---

## Blocked — needs action

| # | Blocker | Needed for | Who |
|---|---|---|---|
| B1 | **Register abstract before 2026-09-18 AoE** | everything | **you** |
| ~~B2~~ | ~~Style files~~ — **RESOLVED**, see "Build" below | compile | done |
| B3 | Read ICLR 2027 **reciprocal reviewing** policy — may gate eligibility | submission | you |
| ~~B4~~ | ~~M7 artifact~~ — **RESOLVED**, SHA-256 verified, Figure 3 built from it | — | done |
| ~~B5~~ | ~~M6 artifact~~ — **RESOLVED**, SHA-256 verified, Figure 2 now reads from it | — | done |
| **B7** | Condition number of $A_t$ — not derivable from the artifacts (they store metrics only). **M17 built and validated; awaiting a Colab run.** | converts C1 from correlational to mechanistic | **you: run the notebook, or abandon on 09-18** |
| B6 | Verify authors for 3 bib entries still marked `(verify authors)`: `kim2026qgpm`, `positcl2024`, `streamingridge2020` | camera-ready correctness | you or me |

Only final-task M7 values are recorded in the docs, so Fig 3 and the condition
number cannot be produced without B4/B5. Everything else proceeds meanwhile.

---

## Consistency issues found while porting

### 1. `PROOFS.md` is behind `M8_THEORY.md` — use M8
`PROOFS.md` §E contains only the **local** factor-to-Gram relation
$\Delta = R^\top E + E^\top R + E^\top E$, and states outright: *"a stream-level
result also needs a recurrence... the current manuscript does not claim
non-accumulation."* The cumulative recurrence $\Delta_t=\Delta_{t-1}+D_t$ that
the paper's central claim rests on lives in
`docs/research/SRQ_GENERALIZATION_M8_THEORY.md` (2026-09-08), which supersedes
it. The appendix must be built from M8. Verified independently: the recurrence
is correct, since $\bar R_t^\top\bar R_t=\widehat A_{t-1}+\Phi_t^\top\Phi_t$
gives $\Delta_t=\Delta_{t-1}+D_t$ exactly.

### 2. Notation collision between `PROOFS.md` and the paper
`PROOFS.md` uses $B_t$ for the **intermediate Gram**
($B_t=\widehat R_{t-1}^\top\widehat R_{t-1}+Z_t^\top Z_t$) and $Q_t$ for the
cross statistic. The paper and `M8_THEORY.md` use $A_t$ for the system and
$B_t$ for the **cross statistic**. Porting PROOFS.md text verbatim into the
appendix would define $B_t$ twice with different meanings. Renotate to
$A_t/B_t/\Phi_t$ throughout when porting.

### 3. Two different Ridge-perturbation bounds are in circulation
`PROOFS.md` Thm 3 gives the $\lambda_{\min}$ form; `M8_THEORY.md` and the paper
give the relative form $\epsilon_t/(1-\epsilon_t)$ with
$\epsilon_t=\|A_t^{-1}\Delta_t\|_2$. Both are correct and the derivation of the
second was checked. Use the M8 form in the main text; the appendix may give
both, but must not present them as the same statement.

### 4. THREE DISTINCT STREAMS — never merge on one accuracy axis
Full-width Exact scores **92.6222** AIA on the M4/M5 stream but **92.4428** at
width 10,000 on the M6/M7/M11/M12 stream; M14 is a third (six paired seeds).
These are not interchangeable. Figure 4 therefore uses M5 points only. Any
future plot combining adaptive (M11/M12) or LoRanPAC (M14) points with M5
points would be a genuine methodological error, and a reviewer would catch it.
This constraint is repeated at the top of `paper/fig/make_figures.py`.

### 5. Ledger still stops at M9
`RESULTS_LEDGER.md` and `VALIDATION.md` cover through M9 only; the paper cites
M10-M16. Appendix A cannot be completed until these are backfilled. Not
blocking the main text.

---

## Deferred to the rebuttal window (2026-11-05 .. 11-18)

Write the Limitations section so each of these is an announced gap, then
deliver it during discussion. Prepare the scripts before 09-25 so the window is
execution only.

| Item | Prepared? |
|---|---|
| Frequent-Directions equal-budget baseline | not started |
| M14 rerun with scale-aware gates (already authorized by M13-N) | not started |
| Multi-seed M5 | not started |
| Second GPU for the systems measurement | not started |
