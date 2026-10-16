"""Figure 2.2: PRISMA 2020 flow, identification of studies via other methods (16 October 2026)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "figures")
out.mkdir(parents=True, exist_ok=True)
INK, EDGE, FILL, SIDE = "#0b0b0b", "#52514e", "#eef4fc", "#fdf1ec"
plt.rcParams.update({"font.family": "DejaVu Serif"})
fig, ax = plt.subplots(figsize=(6.6, 7.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fill=FILL, size=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fill, ec=EDGE, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", wrap=True, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.9))


box(2, 93, 96, 5, "Identification of studies via other methods (PRISMA 2020), search date 16 October 2026",
    fill="#ffffff", size=7.2, bold=True)
for (y, lab) in ((72, "Identification"), (47, "Screening"), (14, "Included")):
    ax.text(1, y, lab, rotation=90, ha="center", va="center", fontsize=8, color=EDGE, fontweight="bold")
box(5, 62, 58, 28,
    "Records identified from:\n"
    "Forward citation searching (Semantic Scholar):\n"
    "Silva et al. (2022), n = 111; Saito et al. (2021), n = 130;\n"
    "Li et al. (2011), n = 220 (partial, API truncation);\n"
    "Silva et al. (2022, iRec), n = 12; Pires et al. (2025), n = 3\n"
    "Backward citation searching:\n"
    "Pires et al. (2025), n = 60; Andrade et al. (2023), n = 21\n"
    "Targeted web searches (3 query strings)\n"
    "Researcher's collection (January 2026), n = 8")
box(68, 70, 30, 14,
    "Titles screened against the\ninclusion criteria (bandit\nmethod, evaluation focus);\nSemantic Scholar search\nendpoint unavailable", fill=SIDE, size=6.6)
arrow(63, 77, 68, 77)
box(5, 40, 58, 14, "Records assessed for eligibility\nn = 33\n(forward 17, backward 7, targeted 2,\nresearcher's collection 8; one record from two routes)")
arrow(34, 62, 34, 54.8)
box(68, 40, 30, 14, "Records excluded, n = 2\nDeep-RL library, not bandit\nevaluation (n = 1)\nEarlier version of OBP,\nduplicate (n = 1)", fill=SIDE, size=6.6)
arrow(63, 47, 68, 47)
box(5, 4, 58, 28,
    "New studies included in the review, n = 31\n"
    "evaluation frameworks, benchmarks or protocols: 6\n"
    "evidence on evaluation practice: 14\n"
    "datasets or simulators: 5\n"
    "surveys or reviews: 6\n\n"
    "First pass: Reviewer 1;\nall 33 decisions verified by the author")
arrow(34, 40, 34, 32.8)
box(68, 4, 30, 28, "Depth of reading at\nthe time of writing:\nfull text 5; repository\nREADME 1; abstract 4;\ncited via another\nincluded paper 5;\nalready cited in thesis 1;\ntitle or metadata only 15\n\nOnly full-text records\nsupport claims in Ch. 2", fill=SIDE, size=6.6)
arrow(63, 18, 68, 18)
p = out / "fig2_2_prisma_other_methods.png"
fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white"); print("written", p)
