"""Build data/questions.jsonl from a public QA benchmark.

Primary: TriviaQA rc.nocontext (streamed — only fetches what it reads, so
it's light on bandwidth). Fallback: PopQA (tiny download) via --source popqa.

Requires:  uv pip install datasets

    python src/make_questions.py                # TriviaQA, 150 questions
    python src/make_questions.py --n 150 --seed 7
    python src/make_questions.py --source popqa # fallback if TriviaQA fails

Notes:
- Samples from the first --pool streamed rows (default 2000). For our
  purpose (a question set the model answers then defends) prefix-sampling
  introduces no meaningful bias; mention the seed + pool in the paper.
- Writes an `aliases` list per question — patch the judge to use it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def split_of(qid: str) -> str:
    """Keep in sync with src/analyze.py."""
    h = int(hashlib.md5(qid.encode()).hexdigest(), 16)
    return "extract" if h % 2 == 0 else "heldout"


def ok(q: str, a: str) -> bool:
    return bool(q) and bool(a) and len(q) <= 250 and 1 <= len(a) <= 40


def from_triviaqa(pool: int):
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                      split="validation", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if i >= pool:
            break
        q = ex["question"].strip()
        a = ex["answer"]["value"].strip()
        aliases = [x.strip() for x in ex["answer"].get("aliases", [])
                   if x and x.strip() and x.strip() != a][:12]
        if ok(q, a):
            rows.append({"id": str(ex["question_id"]), "question": q,
                         "answer": a, "aliases": aliases})
    return rows


def from_popqa(pool: int):
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= pool:
            break
        try:
            answers = json.loads(ex["possible_answers"])
        except Exception:
            continue
        if not answers:
            continue
        q, a = ex["question"].strip(), str(answers[0]).strip()
        # keep well-known entities so a small model has a chance
        if int(ex.get("s_pop") or 0) < 10000:
            continue
        if ok(q, a):
            rows.append({"id": f"popqa{ex['id']}", "question": q, "answer": a,
                         "aliases": [str(x).strip() for x in answers[1:]][:12]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["triviaqa", "popqa"], default="triviaqa")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/questions.jsonl")
    args = ap.parse_args()

    print(f"streaming {args.source} (pool={args.pool}) ...")
    rows = from_triviaqa(args.pool) if args.source == "triviaqa" \
        else from_popqa(args.pool)
    if not rows:
        sys.exit("no usable rows — try the other --source")

    # dedupe on question text
    seen, uniq = set(), []
    for r in rows:
        k = r["question"].lower()
        if k not in seen:
            seen.add(k)
            uniq.append(r)

    random.Random(args.seed).shuffle(uniq)
    out = uniq[: args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ex = sum(split_of(r["id"]) == "extract" for r in out)
    print(f"wrote {len(out)} questions -> {args.out}")
    print(f"split balance: {n_ex} extract / {len(out) - n_ex} heldout")
    print("sample:")
    for r in out[:3]:
        print("  ", json.dumps(r, ensure_ascii=False)[:160])


if __name__ == "__main__":
    main()