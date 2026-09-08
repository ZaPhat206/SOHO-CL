from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_m8_derivation_states_required_scope_and_bounds():
    text = (ROOT / "docs/research/SRQ_GENERALIZATION_M8_THEORY.md").read_text(
        encoding="utf-8"
    )
    required = (
        "\\Delta_t=\\Delta_{t-1}+D_t",
        "2\\|\\bar R_t\\|_2\\|E_t\\|_2+\\|E_t\\|_2^2",
        "\\epsilon_t=\\|A_t^{-1}\\Delta_t\\|_2<1",
        "\\frac{\\epsilon_t}{1-\\epsilon_t}",
        "2\\|\\widehat\\ell-\\ell\\|_\\infty<\\gamma(x)",
        "sufficient, not necessary",
        "not transfer the Shampoo theorem",
        "M6 remains a formal gate failure",
    )
    for fragment in required:
        assert fragment in text


def test_manuscript_contains_m8_theory_and_m7_claim_boundary():
    text = (ROOT / "paper/SRQ_FLY_LNCS_DRAFT.tex").read_text(encoding="utf-8")
    required = (
        r"\subsection{Cumulative Perturbation and Prediction Stability}",
        r"\label{eq:error-recurrence}",
        r"\label{eq:cumulative-bound}",
        r"\label{eq:ridge-perturbation}",
        r"\label{eq:margin-certificate}",
        r"\subsection{Task-Wise Error Trajectory}",
        "causal proof or evidence of ill-conditioning.",
        "remains a formal failure.",
    )
    for fragment in required:
        assert fragment in text
