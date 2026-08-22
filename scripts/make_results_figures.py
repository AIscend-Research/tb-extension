"""The two results figures derivable from RESULTS_RUN1.md's measured tables.

Every number here is transcribed from docs/RESULTS_RUN1.md (run 1, DenseNet121,
20 epochs, LOCO). The three curve figures (reliability, risk-coverage, per-clinic
heatmap) need per-sample predictions and are deliberately NOT drawn here.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, PAPER, PANEL = "#1b1b1f", "#8a8f98", "#fbfbfd", "#eef0f4"
C_MONT, C_SHEN = "#2778c4", "#b06e08"  # validated: CVD dE 23.8, contrast >= 3:1

SEV = [0.00, 0.25, 0.50, 0.75, 1.00]
ACC_MONT = [0.717, 0.674, 0.681, 0.652, 0.645]   # Montgomery held out, n=138
ACC_SHEN = [0.625, 0.675, 0.644, 0.600, 0.583]   # Shenzhen held out, n=662

VAL = {"Montgomery": 0.919, "Shenzhen": 0.952}
HELD = {"Montgomery": 0.717, "Shenzhen": 0.625}
N = {"Montgomery": 138, "Shenzhen": 662}


def _style(ax):
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=PANEL, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def accuracy_vs_severity(path):
    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=PAPER)
    _style(ax)
    for acc, col, name, n in ((ACC_MONT, C_MONT, "Montgomery", 138),
                              (ACC_SHEN, C_SHEN, "Shenzhen", 662)):
        ax.plot(SEV, acc, color=col, lw=2, marker="o", ms=5,
                markerfacecolor=col, markeredgecolor=PAPER, markeredgewidth=1.2,
                label=f"{name} held out (n={n})", zorder=3)
        ax.annotate(f"{name}\nn={n}", (SEV[-1], acc[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=8.6, color=INK,
                    va="center", weight="bold")
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.005, 0.507, "chance (two-class)", fontsize=7.6, color=MUTED)
    # the counter-intuitive point worth a reader's attention
    ax.annotate("pristine input is OOD relative to the\nseverity-randomised training mean",
                xy=(0.25, 0.675), xytext=(0.30, 0.76), fontsize=7.6, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(-0.03, 1.22)
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(SEV)
    ax.set_xlabel("degradation severity", fontsize=9.5, color=INK)
    ax.set_ylabel("accuracy on the held-out clinic", fontsize=9.5, color=INK)
    ax.set_title("Accuracy under synthetic smartphone degradation (E3)",
                 fontsize=11, color=INK, loc="left", pad=10)
    leg = ax.legend(loc="upper right", fontsize=8.2, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def generalization_gap(path):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), facecolor=PAPER)
    _style(ax)
    folds = ["Montgomery", "Shenzhen"]
    x = np.arange(len(folds), dtype=float)
    w = 0.34
    for i, fold in enumerate(folds):
        col = C_MONT if fold == "Montgomery" else C_SHEN
        ax.bar(x[i] - w / 2 - 0.01, VAL[fold], w, color=col, alpha=0.35, zorder=2)
        ax.bar(x[i] + w / 2 + 0.01, HELD[fold], w, color=col, zorder=2)
        ax.text(x[i] - w / 2 - 0.01, VAL[fold] + 0.012,
                f"validation (seen)\n{VAL[fold]:.3f}",
                ha="center", fontsize=8.2, color=INK, linespacing=1.3)
        ax.text(x[i] + w / 2 + 0.01, HELD[fold] + 0.012,
                f"held-out\n{HELD[fold]:.3f}",
                ha="center", fontsize=8.2, color=INK, weight="bold", linespacing=1.3)
        drop = (HELD[fold] - VAL[fold]) * 100
        ax.annotate("", xy=(x[i] + w / 2 + 0.01, HELD[fold] + 0.085),
                    xytext=(x[i] + w / 2 + 0.01, VAL[fold]),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
        ax.text(x[i] + w / 2 + 0.09, (VAL[fold] + HELD[fold]) / 2 + 0.03,
                f"{drop:+.0f} pts", fontsize=9, color=INK, weight="bold", va="center")
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(1.44, 0.507, "chance", fontsize=7.6, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} held out\n(n={N[f]})" for f in folds],
                       fontsize=9.2, color=INK)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy", fontsize=9.5, color=INK)
    ax.set_title("Cross-site generalisation gap, leave-one-clinic-out (E2)",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.set_ylim(0.0, 1.12)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    out = sys.argv[1]
    accuracy_vs_severity(f"{out}/e3_accuracy_vs_severity.png")
    generalization_gap(f"{out}/e2_generalization_gap.png")
    print("done")
