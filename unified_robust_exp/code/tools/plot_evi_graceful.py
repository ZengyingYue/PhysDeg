"""Vector-PDF figure: the evidential Dempster readout degrades more gracefully
than the fused-softmax readout under optical degradation (EarthMiss, pixel acc.).

Data = the EarthMiss calibration sweep (tools/eval_calib.py on runs/em_r1): for
each condition we read both the softmax-fused prediction's accuracy and the
evidential Dempster (alpha/S) prediction's accuracy from the SAME forward pass.
Honest nuance kept: evidential wins big under haze/noise but loses under extreme
blur (blur@5) — shaded green where evidential helps, red where it hurts.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# severity index 0=clean,1,2,3 -> labels; values = pixel accuracy
XL = ["clean", "1", "3", "5"]
X = [0, 1, 2, 3]
DATA = {
    "Blur":            {"soft": [0.6395, 0.6314, 0.5855, 0.5017],
                        "evi":  [0.6734, 0.6718, 0.5942, 0.4106]},
    "Gaussian noise":  {"soft": [0.6395, 0.6414, 0.6388, 0.5177],
                        "evi":  [0.6734, 0.6782, 0.6653, 0.5883]},
    "Haze / cloud":    {"soft": [0.6395, 0.4626, 0.3994, 0.3778],
                        "evi":  [0.6734, 0.6289, 0.4942, 0.4471]},
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.linewidth": 0.8, "pdf.fonttype": 42,
})
C_SOFT, C_EVI = "#5B6C8F", "#C0392B"

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
for ax, (name, d) in zip(axes, DATA.items()):
    s, e = d["soft"], d["evi"]
    ax.fill_between(X, s, e, where=[ev >= so for ev, so in zip(e, s)],
                    interpolate=True, color="#2E8B57", alpha=0.16, zorder=0)
    ax.fill_between(X, s, e, where=[ev < so for ev, so in zip(e, s)],
                    interpolate=True, color="#C0392B", alpha=0.13, zorder=0)
    ax.plot(X, s, "--o", color=C_SOFT, lw=1.6, ms=4.5, mfc="white",
            mew=1.3, label="Softmax fusion", zorder=3)
    ax.plot(X, e, "-s", color=C_EVI, lw=1.9, ms=4.5, label="Evidential (Dempster)", zorder=4)
    ax.set_title(name)
    ax.set_xticks(X); ax.set_xticklabels(XL)
    ax.set_xlabel("degradation severity")
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.margins(x=0.05)
axes[0].set_ylabel("Pixel accuracy")
axes[0].set_ylim(0.38, 0.72)

handles = [Line2D([0], [0], color=C_SOFT, ls="--", marker="o", mfc="white", mew=1.3, ms=4.5, lw=1.6),
           Line2D([0], [0], color=C_EVI, ls="-", marker="s", ms=4.5, lw=1.9),
           plt.Rectangle((0, 0), 1, 1, fc="#2E8B57", alpha=0.16),
           plt.Rectangle((0, 0), 1, 1, fc="#C0392B", alpha=0.13)]
labels = ["Softmax fusion", "Evidential (Dempster)", "evidential advantage", "evidential deficit"]
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

import sys
no_title = "--no-title" in sys.argv
if not no_title:
    fig.suptitle("Evidential reliability fusion degrades more gracefully than softmax "
                 "fusion under optical degradation (EarthMiss)", y=1.02, fontsize=10)
fig.tight_layout(rect=[0, 0.03, 1, 1])

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = args[0] if args else "evi_graceful.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}{' (no title)' if no_title else ''}")
