import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(8.5, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

stages = [
    (0.3, "293\nquestions\ntested", "#D9E1F2"),
    (2.8, "263\ninitially\ncorrect", "#DDEBF7"),
    (5.3, "311 / 1,052\nepisodes\nflipped (29.6%)", "#F8CBAD"),
    (7.8, "250 / 311\nepisodes\nrecovered (80.4%)", "#C6EFCE"),
]

w, h, pad = 1.7, 1.4, 0.15
for x, text, color in stages:
    box = FancyBboxPatch((x, 0.8), w, h, boxstyle=f"round,pad={pad}",
                         edgecolor="black", linewidth=1.2, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, 1.5, text, ha="center", va="center", fontsize=9, fontweight="bold")

for i in range(len(stages) - 1):
    x1 = stages[i][0] + w + pad
    x2 = stages[i+1][0] - pad
    arrow = FancyArrowPatch((x1, 1.5), (x2, 1.5), arrowstyle="-|>",
                            mutation_scale=16, linewidth=1.2, color="black")
    ax.add_patch(arrow)

ax.text(6.15, 0.35, "263 questions × 4 pushback styles = 1,052 episodes",
        ha="center", fontsize=7.5, style="italic", color="#555555")

ax.set_title("From Screening to Recovery: Where the Numbers Come From",
             fontsize=11, fontweight="bold", y=1.05)

plt.tight_layout()
plt.savefig("pipeline_funnel.png", dpi=200, bbox_inches="tight")