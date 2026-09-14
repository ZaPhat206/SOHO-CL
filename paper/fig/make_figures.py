"""Generate figures for SRQ_ICLR2027.tex directly from the archived artifacts.

Usage:  python paper/fig/make_figures.py
Output: paper/fig/fig2_width_scaling.pdf
        paper/fig/fig3_error_trajectory.pdf
        paper/fig/fig4_pareto.pdf

PROVENANCE
----------
Figures 2 and 3 are read straight out of the experiment ZIPs in paper/, and the
SHA-256 of each ZIP is verified against the value recorded in
docs/research/SRQ_GENERALIZATION_PROTOCOL.md before anything is plotted. No
number is transcribed by hand, so the figures cannot drift from the evidence.

Figure 4 still carries inline values: the M5 equal-budget ZIP is not present in
this checkout. Replace INLINE_M5 with a ZIP read as soon as that artifact is
available.

!! DO NOT MERGE STREAMS !!
M4/M5, M6/M7/M11/M12 and M14 are THREE DIFFERENT train-only streams. Full-width
Exact scores 92.6222 AIA on the M5 stream but 92.4428 at width 10,000 on the M6
stream. Plotting points from different streams on one accuracy axis would be a
methodological error. Figure 4 uses M5 points only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import pathlib
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).parent
PAPER = OUT.parent

ARTIFACTS = {
    "m6": (
        "srq_generalization_m6_width_sweep_train_only.zip",
        "b2739b9da023ebd2eedb6fdfe01c394e94f252773e847533b35350021c3d239e",
    ),
    "m7": (
        "srq_generalization_m7_error_trajectory_train_only.zip",
        "df92adadce046c53efa5b9fcf01435d1fab2a4c71a690b10c3d7c221205acf36",
    ),
}


def read_artifact_csv(key: str, member: str) -> list[dict]:
    """Verify the artifact hash, then return one CSV member as dict rows."""
    name, expected = ARTIFACTS[key]
    path = PAPER / name
    if not path.exists():
        sys.exit(f"missing artifact: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        sys.exit(
            f"SHA-256 MISMATCH for {name}\n  expected {expected}\n  found    {digest}\n"
            "Refusing to plot from an artifact that is not the recorded one."
        )
    print(f"  verified {name}  sha256 ok")
    with zipfile.ZipFile(path) as z:
        text = z.read(member).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


C_EXACT = "#4C4C4C"
C_FP16 = "#0072B2"
C_P2B = "#D55E00"
C_ALT = "#009E73"
C_GATE = "#CC3311"
C_LOCAL = "#7A5195"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

GATE = 0.25  # preregistered P2B accuracy-retention bound, in AIA points


# ---------------------------------------------------------------------------
# Figure 2: width scaling (M6)
# ---------------------------------------------------------------------------
def figure2() -> None:
    rows = read_artifact_csv("m6", "width_sweep.csv")
    by = {(int(r["width"]), r["method"]): r for r in rows}
    widths = sorted({int(r["width"]) for r in rows})

    def aia(w, m):
        return float(by[(w, m)]["validation_aia_percent"])

    def state(w, m):
        return float(by[(w, m)]["final_total_persistent_bytes"])

    gap_p2b = [aia(w, "exact") - aia(w, "p2b_int8") for w in widths]
    gap_fp16 = [aia(w, "exact") - aia(w, "fp16_square_root") for w in widths]
    reduction = [100.0 * (1.0 - state(w, "p2b_int8") / state(w, "exact")) for w in widths]

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.axhline(GATE, color=C_GATE, ls="--", lw=1.0, zorder=1)
    ax.text(
        2_300,
        GATE - 0.028,
        f"preregistered retention bound ({GATE:.2f} pt)",
        color=C_GATE,
        fontsize=6.5,
        va="top",
    )
    ax.plot(widths, gap_p2b, "o-", color=C_P2B, lw=1.4, ms=4, label="fixed INT8 (P2B)", zorder=3)
    ax.plot(widths, gap_fp16, "s-", color=C_FP16, lw=1.2, ms=3.2, label="FP16 square root", zorder=3)
    ax.axhline(0.0, color="#BBBBBB", lw=0.6, zorder=0)

    ax.plot([widths[-1]], [gap_p2b[-1]], "o", ms=9, mfc="none", mec=C_GATE, mew=1.6, zorder=4)
    ax.annotate(
        f"gate fails\nby {gap_p2b[-1] - GATE:.4f} pt",
        xy=(widths[-1], gap_p2b[-1]),
        xytext=(15_600, 0.30),
        fontsize=6.5,
        color=C_GATE,
        ha="center",
        arrowprops=dict(arrowstyle="-", color=C_GATE, lw=0.7),
    )

    ax.set_xlabel("representation width $m$")
    ax.set_ylabel("AIA gap to Exact (points)")
    ax.set_xticks(widths)
    ax.set_xticklabels([f"{w // 1000}k" for w in widths])
    ax.set_ylim(-0.05, 0.35)
    ax.legend(loc="upper left", frameon=False)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(widths, reduction, "^:", color=C_ALT, lw=1.1, ms=3.5, label="P2B state reduction")
    ax2.set_ylabel("state reduction vs Exact (%)", color=C_ALT)
    ax2.tick_params(axis="y", colors=C_ALT)
    ax2.set_ylim(50, 90)
    ax2.legend(loc="lower right", frameon=False)

    fig.savefig(OUT / "fig2_width_scaling.pdf")
    plt.close(fig)
    print(f"  wrote fig2_width_scaling.pdf  (20k gap {gap_p2b[-1]:.6f}, reduction {reduction[-1]:.2f}%)")


# ---------------------------------------------------------------------------
# Figure 3: task-wise error trajectory (M7)
# ---------------------------------------------------------------------------
def figure3() -> None:
    rows = read_artifact_csv("m7", "error_trajectory.csv")

    def series(method: str, width: int, field: str):
        rs = [r for r in rows if r["method"] == method and int(r["width"]) == width]
        rs.sort(key=lambda r: int(r["task"]))
        return [int(r["task"]) for r in rs], [float(r[field]) for r in rs if r[field] != ""]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # (a) error magnitudes at the failing width
    W = 20_000
    curves = [
        ("relative_local_factor_quantization_error", "local factor $\\|E_t\\|$", C_LOCAL, "o"),
        ("relative_system_action_error", "system $\\Delta_t$ (action)", C_ALT, "s"),
        ("relative_weight_error", "classifier $W_t$", C_P2B, "^"),
        ("relative_logit_error", "logits", "#B4004E", "D"),
    ]
    for field, label, colour, marker in curves:
        t, v = series("p2b_int8", W, field)
        axa.plot(t, v, marker=marker, color=colour, lw=1.3, ms=3.2, label=label)

    t, v = series("fp16_square_root", W, "relative_logit_error")
    axa.plot(t, v, ls=":", color=C_FP16, lw=1.2, label="FP16 logits")

    axa.set_yscale("log")
    axa.set_xlabel("task")
    axa.set_ylabel("relative error")
    axa.set_xticks(range(1, 11))
    axa.set_xlim(0.4, 11.4)
    # Headroom at the top so the legend never sits on a curve.
    axa.set_ylim(2e-3, 8.0)
    axa.set_title(f"(a) width {W:,}, fixed INT8", fontsize=7.5, loc="left")
    axa.legend(frameon=False, fontsize=6.0, loc="upper left", ncol=2,
               handlelength=1.5, columnspacing=0.9, handletextpad=0.4)

    # The contrast that carries the argument: local flat, downstream compounding.
    tl, vl = series("p2b_int8", W, "relative_local_factor_quantization_error")
    tg, vg = series("p2b_int8", W, "relative_logit_error")
    axa.annotate(
        f"$\\times${vl[-1] / vl[0]:.2f}",
        xy=(10, vl[-1]),
        xytext=(6, -1),
        textcoords="offset points",
        fontsize=6.4,
        color=C_LOCAL,
        va="center",
    )
    axa.annotate(
        f"$\\times${vg[-1] / vg[0]:.1f}",
        xy=(10, vg[-1]),
        xytext=(6, -1),
        textcoords="offset points",
        fontsize=6.4,
        color="#B4004E",
        va="center",
    )

    # (b) decision-level consequence
    for width, style in ((10_000, "--"), (20_000, "-")):
        t, v = series("p2b_int8", width, "prediction_agreement")
        axb.plot(t, [100 * x for x in v], style, color=C_P2B, lw=1.3,
                 label=f"agreement, $m$={width // 1000}k")
        t, v = series("p2b_int8", width, "margin_certified_fraction")
        axb.plot(t, [100 * x for x in v], style, color=C_ALT, lw=1.3,
                 label=f"margin certified, $m$={width // 1000}k")

    axb.set_xlabel("task")
    axb.set_ylabel("percent of validation codes")
    axb.set_xticks(range(1, 11))
    axb.set_title("(b) effect on decisions", fontsize=7.5, loc="left")
    axb.legend(frameon=False, fontsize=6.1, loc="lower left")

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig3_error_trajectory.pdf")
    plt.close(fig)
    print(f"  wrote fig3_error_trajectory.pdf  (local x{vl[-1] / vl[0]:.2f} vs logits x{vg[-1] / vg[0]:.1f})")


# ---------------------------------------------------------------------------
# Figure 4: equal-budget frontier (M5) -- inline until the ZIP is available
# ---------------------------------------------------------------------------
MB_TO_MIB = 1e6 / 1048576.0
INLINE_M5 = [
    ("Exact (full width)", 92.6222, 438.720, 4.3156, C_EXACT, "o"),
    ("FP16 square root", 92.6215, 138.750, 8.0715, C_FP16, "s"),
    ("SRQ INT8/FP32 (P2B)", 92.4608, 91.880, 8.3869, C_P2B, "D"),
    ("Exact, byte-matched\n($m$=4,333)", 91.9770, 91.877, 0.5498, C_ALT, "v"),
    ("CountSketch Exact\n($d$=3,809)", 91.8889, 91.852, 0.4363, C_ALT, "^"),
    ("Raw-feature Ridge", 91.1154, 2.974, 0.0528, "#999999", "x"),
]


def figure4() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    place = {
        "Exact (full width)": (0, 13, "center"),
        "FP16 square root": (10, -10, "left"),
        "SRQ INT8/FP32 (P2B)": (-11, 7, "right"),
        "Exact, byte-matched\n($m$=4,333)": (13, 12, "left"),
        "CountSketch Exact\n($d$=3,809)": (13, -14, "left"),
        "Raw-feature Ridge": (11, 0, "left"),
    }
    ax.axvspan(84.0, 91.5, color="#000000", alpha=0.055, zorder=0, lw=0)

    for label, aia, mb, secs, colour, marker in INLINE_M5:
        mib = mb * MB_TO_MIB
        edge = {} if marker == "x" else dict(edgecolors="white", linewidths=0.6)
        ax.scatter(mib, aia, s=52, c=colour, marker=marker, zorder=4, **edge)
        dx, dy, ha = place[label]
        ax.annotate(
            f"{label}\n{secs:.2f} s update",
            xy=(mib, aia),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=6.3,
            color="#333333",
            linespacing=1.25,
        )

    x_gap = 80.0
    y_lo, y_hi = 91.9770, 92.4608
    ax.annotate("", xy=(x_gap, y_hi), xytext=(x_gap, y_lo),
                arrowprops=dict(arrowstyle="<->", color=C_P2B, lw=1.1))
    ax.annotate(
        f"+{y_hi - y_lo:.3f} pt\nat equal bytes",
        xy=(x_gap, (y_lo + y_hi) / 2),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=6.5,
        color=C_P2B,
        linespacing=1.25,
    )
    ax.annotate("equal byte budget ($\\approx$87.6 MiB)", xy=(87.6, 90.97),
                fontsize=6.3, color="#777777", ha="center")

    ax.set_xscale("log")
    ax.set_xlim(1.9, 1100)
    ax.set_ylim(90.9, 92.95)
    ax.set_xlabel("total persistent learner state (MiB, log scale)")
    ax.set_ylabel("validation AIA (%)")

    fig.savefig(OUT / "fig4_pareto.pdf")
    plt.close(fig)
    print("  wrote fig4_pareto.pdf  (inline M5 values -- ZIP not in checkout)")


if __name__ == "__main__":
    print("figures:")
    figure2()
    figure3()
    figure4()
