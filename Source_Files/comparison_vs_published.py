import matplotlib.pyplot as plt
import numpy as np

groups = ["This work\n(Gemma-2-2b-it)", "Aamir & Bin Adil\n(Qwen2.5-1.5B)", "Aamir & Bin Adil\n(Llama-3.2-1B)"]
flip_vals = [29.6, 41.8, 43.1]     # this work: 311/1052
recov_vals = [80.4, 13.0, 13.0]    # this work: 250/311

x = np.arange(len(groups))
w = 0.35

fig, ax = plt.subplots(figsize=(7.5, 4.5))
b1 = ax.bar(x - w/2, flip_vals, w, label="Flip rate", color="#C00000")
b2 = ax.bar(x + w/2, recov_vals, w, label="Recovery / repair rate", color="#548235")

for bar, val in zip(b1, flip_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val}%", ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(b2, recov_vals):
    label = f"~{val}%" if val == 13.0 else f"{val}%"
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, label, ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=8.5)
ax.set_ylabel("Rate (%)")
ax.set_ylim(0, 95)
ax.set_title("This Study vs. Published Rates (Aamir & Bin Adil, 2026)", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.text(0.5, -0.32, "Note: Aamir & Bin Adil's repair rate is unconditioned (separate items); this work's recovery rate is conditional (same item).",
        transform=ax.transAxes, ha="center", fontsize=7.5, style="italic", color="#555555")

plt.tight_layout()
plt.savefig("comparison_vs_published.png", dpi=200, bbox_inches="tight")