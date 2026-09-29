"""
Make graphs from the results of every model you have run.

It looks inside outputs/<model>/transcripts/ for:
  - baseline.jsonl  (main run)   -> graph 1: flip rate per pushback type
  - screen.jsonl    (screening)  -> graph 2: how many questions each model got right

Run from the project folder:
    python src/plot_results.py

Graphs are saved in:  figures/
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

TYPES = ["simple", "social", "authoritative", "emotional"]

# Nice names for the graph legend
NICE_NAMES = {
    "gemma2b": "Gemma-2-2B",
    "llama1b": "Llama-3.2-1B",
    "qwen1_5b": "Qwen2.5-1.5B",
}

# Numbers from the paper's table (Qwen2.5-1.5B, n=159 per type)
PAPER = {
    "name": "Qwen2.5-1.5B (paper)",
    "n": 159,
    "flips": {"simple": 79, "social": 67, "authoritative": 60, "emotional": 60},
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson_interval(k, n, z=1.96):
    """95% error bar for a percentage (how sure we are about it)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


# ---------------------------------------------------------------------
# Collect results from every model folder
# ---------------------------------------------------------------------

models = []  # list of dicts: name, n, flips{type: k}

for baseline in sorted(ROOT.glob("outputs/*/transcripts/baseline.jsonl")):
    slug = baseline.parent.parent.name
    rows = read_jsonl(baseline)

    flips = {t: 0 for t in TYPES}
    totals = {t: 0 for t in TYPES}

    for r in rows:
        t = r.get("pushback_type")
        if t not in TYPES or not r.get("initially_correct"):
            continue
        totals[t] += 1
        flips[t] += int(bool(r.get("capitulated")))

    n = max(totals.values()) if totals else 0
    if n == 0:
        continue

    models.append({"name": NICE_NAMES.get(slug, slug), "n": n, "flips": flips})
    print(f"[plot] {slug}: n={n}, flips={flips}")

models.append(PAPER)

# ---------------------------------------------------------------------
# GRAPH 1: flip rate per pushback type, one bar per model
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

n_models = len(models)
width = 0.8 / n_models

for i, m in enumerate(models):
    rates, err_low, err_high = [], [], []
    for t in TYPES:
        k, n = m["flips"][t], m["n"]
        p = k / n
        lo, hi = wilson_interval(k, n)
        rates.append(p * 100)
        err_low.append((p - lo) * 100)
        err_high.append((hi - p) * 100)

    xs = [j + (i - (n_models - 1) / 2) * width for j in range(len(TYPES))]
    hatch = "//" if "paper" in m["name"] else None
    bars = ax.bar(
        xs, rates, width,
        yerr=[err_low, err_high], capsize=3,
        label=f"{m['name']} (n={m['n']})",
        hatch=hatch, edgecolor="black", linewidth=0.5,
    )
    for b, r, eh in zip(bars, rates, err_high):
        ax.text(b.get_x() + b.get_width() / 2, r + eh + 1, f"{r:.0f}%",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(range(len(TYPES)))
ax.set_xticklabels([t.capitalize() for t in TYPES])
ax.set_ylabel("Flip rate (%)\n(correct answer changed to wrong)")
ax.set_xlabel("Type of pushback")
ax.set_title("How often does the model give up a correct answer?")
ax.set_ylim(0, 100)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
path1 = OUT_DIR / "flip_rate_by_type.png"
plt.savefig(path1, dpi=200)
print(f"[plot] saved {path1}")
plt.close()

# ---------------------------------------------------------------------
# GRAPH 2: initial accuracy from screening
# ---------------------------------------------------------------------

names, correct, wrong = [], [], []

for screen in sorted(ROOT.glob("outputs/*/transcripts/screen.jsonl")):
    slug = screen.parent.parent.name
    rows = read_jsonl(screen)
    if len(rows) < 10:  # skip tiny test runs
        continue
    c = sum(bool(r.get("initially_correct")) for r in rows)
    names.append(NICE_NAMES.get(slug, slug))
    correct.append(c)
    wrong.append(len(rows) - c)

if names:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, correct, label="Correct", color="tab:green")
    ax.bar(names, wrong, bottom=correct, label="Wrong", color="tab:red", alpha=0.6)
    for i, (c, w) in enumerate(zip(correct, wrong)):
        total = c + w
        ax.text(i, c / 2, f"{c}\n({c / total:.0%})", ha="center", va="center",
                color="white", fontweight="bold")
    ax.set_ylabel("Number of questions")
    ax.set_title("Screening: first-answer accuracy (TriviaQA)")
    ax.legend()
    plt.tight_layout()
    path2 = OUT_DIR / "screening_accuracy.png"
    plt.savefig(path2, dpi=200)
    print(f"[plot] saved {path2}")
    plt.close()
else:
    print("[plot] no full screening results found, skipped graph 2")