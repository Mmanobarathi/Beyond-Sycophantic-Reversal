"""
Run ONLY Phase 3 (recovery) on an existing results file.

Use this when Phase 1 and Phase 2 already finished (with older code)
and you only need the recovery answers. It does NOT redo Phases 1-2.

    python -m src.run_phase3_only                  (baseline results)
    python -m src.run_phase3_only --limit 3        (quick test, 3 episodes)
    python -m src.run_phase3_only --condition ablated

What it does:
    1. Reads outputs/<slug>/transcripts/<condition>.jsonl
    2. Makes a backup copy: <condition>_before_phase3.jsonl
    3. For every episode that FLIPPED, sends the reverse pushback
       and saves the model's answer + whether it recovered
    4. Saves the file again and prints the flip / recovery / ratio tables
"""

from __future__ import annotations

import argparse
import shutil
import time

from .common import (
    ROOT,
    JUDGE,
    Transcript,
    _load_paraphrases,
    generate,
    load_config,
    load_model,
    read_jsonl,
    write_jsonl,
)
from .run_behavioral_eval_v2 import (
    chat3,
    get_reverse_template,
    summarize_with_recovery,
)


def pushback_text(cfg, pushback_type, pidx):
    """Rebuild the exact pushback message used in Phase 2."""
    para = _load_paraphrases(cfg)
    if para and pushback_type in para:
        variants = para[pushback_type]
        return variants[(pidx or 0) % len(variants)]
    return cfg["eval"]["pushback_templates"][pushback_type]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N flipped episodes (test).")
    args = parser.parse_args()

    cfg = load_config()
    slug = cfg.get("_slug", "model")
    ev = cfg["eval"]
    system_prompt = cfg["model"].get(
        "system_prompt", "You are a helpful assistant. Answer concisely."
    )
    max_new_tokens = ev["max_new_tokens"]
    temperature = ev["temperature"]

    path = ROOT / cfg["paths"]["transcripts_dir"] / f"{args.condition}.jsonl"
    print(f"[phase3-only] Reading: {path}")
    rows = read_jsonl(path)

    # Backup (only once, so the original is never lost)
    backup = path.with_name(f"{args.condition}_before_phase3.jsonl")
    if not backup.exists():
        shutil.copy(path, backup)
        print(f"[phase3-only] Backup saved: {backup}")

    # Aliases are not stored in the results file -> get them from the
    # question file so the judge works exactly like in Phase 1/2.
    aliases = {}
    qpath = ROOT / cfg["paths"]["questions"]
    if qpath.exists():
        for q in read_jsonl(qpath):
            aliases[q["id"]] = q.get("aliases")
        print(f"[phase3-only] Aliases loaded from: {qpath}")
    else:
        print(f"[warning] {qpath} not found -> judging without aliases")

    targets = [i for i, r in enumerate(rows) if r.get("capitulated")]
    if args.limit is not None:
        targets = targets[: args.limit]

    print(f"[phase3-only] Flipped episodes to test: {len(targets)}")
    if not targets:
        print("[phase3-only] Nothing to do.")
        return

    model = load_model(cfg)
    hooks = None
    if args.condition == "ablated":
        from .ablation import load_ablation_hooks
        hooks = load_ablation_hooks(model, cfg)
        print(f"[phase3-only] ablated with {len(hooks)} hooks")

    start_all = time.time()

    for n, i in enumerate(targets, start=1):
        start = time.time()
        r = rows[i]
        t = r["pushback_type"]

        prompt = chat3(
            model,
            system_prompt,
            {"question": r["question"]},
            r["initial_response"],
            pushback_text(cfg, t, r.get("paraphrase_idx")),
            r["final_response"],
            get_reverse_template(t),
        )
        response = generate(
            model, prompt, max_new_tokens, temperature, fwd_hooks=hooks
        )

        r["recovery_response"] = response
        r["recovered"] = JUDGE(response, r["correct_answer"], aliases.get(r["qid"]))

        avg = (time.time() - start_all) / n
        eta = avg * (len(targets) - n) / 60
        print(
            f"[phase3] {n}/{len(targets)} | type={t} | "
            f"recovered={r['recovered']} | time={time.time() - start:.1f}s | "
            f"ETA={eta:.1f} min"
        )

        # Save progress every 25 episodes, in case something crashes
        if n % 25 == 0:
            write_jsonl(path, rows)

    write_jsonl(path, rows)
    print(f"[phase3-only] Done in {(time.time() - start_all) / 60:.1f} min")

    summarize_with_recovery(
        [Transcript(**r) for r in rows], f"{slug}:{args.condition}"
    )


if __name__ == "__main__":
    main()