import matplotlib.pyplot as plt
import numpy as np

styles = ["Simple\ndoubt", "Social\nproof", "Authoritative\nappeal", "Emotional\nappeal"]
flip = [16.0, 18.3, 37.3, 46.8]          # flipped / initially correct (263)
recovery = [90.5, 85.4, 50.0, 99.2]      # recovered / flipped

x = np.arange(len(styles))
w = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar(x - w/2, flip, w, label="Flip rate", color="#C00000")
b2 = ax.bar(x + w/2, recovery, w, label="Recovery rate", color="#548235")

for bar, val in zip(b1, flip):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val}%", ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(b2, recovery):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val}%", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(styles, fontsize=9)
ax.set_ylabel("Rate (%)")
ax.set_ylim(0, 120)
ax.set_title("Flip Rate vs. Conditional Recovery Rate by Pushback Style\n(Gemma-2-2b-it)", fontsize=11, fontweight="bold")
ax.legend(loc="upper center", ncol=2, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("flip_vs_recovery_by_style.png", dpi=200)