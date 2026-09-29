"""
Screen a question pool using turn-1 generations only.

Purpose:
- Run the model on a large question pool.
- Keep questions where the model's first answer is correct.
- Keep up to K initially-wrong questions as control questions.
- Carry forward eligible questions from a previous baseline.
- Write an audit file.

Example:

    python -m src.screen_questions data/pool.jsonl --keep-wrong 30

Quick test:

    python -m src.screen_questions data/pool.jsonl --limit 10 --keep-wrong 3

Quick test with shorter generation:

    python -m src.screen_questions data/pool.jsonl --limit 10 --max-new-tokens 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .analyze import split_of
from .common import (
    ROOT,
    JUDGE,
    format_chat,
    generate,
    load_config,
    load_model,
    read_jsonl,
    write_jsonl,
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def print_device_info(model, cfg):
    """Print information about the device being used."""

    device = next(model.parameters()).device

    print()
    print("=" * 70)
    print("MODEL / DEVICE INFORMATION")
    print("=" * 70)

    print(f"Model : {cfg['model']['name']}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"CUDA  : {torch.version.cuda}")

        try:
            total = torch.cuda.get_device_properties(0).total_memory
            print(f"VRAM  : {total / (1024 ** 3):.2f} GB")
        except Exception:
            pass

    else:
        print("GPU   : CUDA not available")

    print("=" * 70)
    print()


def save_checkpoint(audit, path):
    """Save screening progress."""

    write_jsonl(path, audit)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser(
        description="Screen question pool using turn-1 generations."
    )

    ap.add_argument(
        "pool",
        type=Path,
        help="Input question pool JSONL.",
    )

    ap.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help=(
            "Previous baseline transcript JSONL. "
            "Questions already present will be skipped."
        ),
    )

    ap.add_argument(
        "--old-questions",
        type=Path,
        default=None,
        help=(
            "Previous question file matching --exclude. "
            "Initially-correct questions are carried forward."
        ),
    )

    ap.add_argument(
        "--keep-wrong",
        type=int,
        default=30,
        help="Maximum number of initially-wrong questions to retain.",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N questions. "
            "Useful for testing."
        ),
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override config max_new_tokens. "
            "Example: --max-new-tokens 30"
        ),
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override generation temperature.",
    )

    ap.add_argument(
        "--save-every",
        type=int,
        default=10,
        help=(
            "Save screening audit every N questions. "
            "Default: 10."
        ),
    )

    args = ap.parse_args()

    # -----------------------------------------------------------------
    # CONFIG
    # -----------------------------------------------------------------

    cfg = load_config()
    ev = cfg["eval"]

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else ev["max_new_tokens"]
    )

    temperature = (
        args.temperature
        if args.temperature is not None
        else ev["temperature"]
    )

    # -----------------------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------------------

    print()
    print("[screen] Loading model...")

    model = load_model(cfg)

    print_device_info(model, cfg)

    system_prompt = cfg["model"]["system_prompt"]

    # -----------------------------------------------------------------
    # PREVIOUS RESULTS
    # -----------------------------------------------------------------

    done_qids = set()
    old_eligible = set()

    if args.exclude and args.exclude.exists():

        print(
            f"[screen] Reading previous transcript: "
            f"{args.exclude}"
        )

        prev = read_jsonl(args.exclude)

        done_qids = {
            r["qid"]
            for r in prev
            if "qid" in r
        }

        old_eligible = {
            r["qid"]
            for r in prev
            if r.get("initially_correct") is True
        }

        print(
            f"[screen] Excluding {len(done_qids)} "
            f"already-run questions."
        )

        print(
            f"[screen] {len(old_eligible)} "
            f"eligible questions will be carried forward."
        )

    # -----------------------------------------------------------------
    # LOAD QUESTION POOL
    # -----------------------------------------------------------------

    print()
    print(f"[screen] Reading pool: {args.pool}")

    all_questions = read_jsonl(args.pool)

    pool = [
        q
        for q in all_questions
        if q.get("id") not in done_qids
    ]

    # Apply limit only AFTER excluding old questions.
    if args.limit is not None:
        pool = pool[:args.limit]

    print()
    print("=" * 70)
    print("SCREENING SETTINGS")
    print("=" * 70)
    print(f"Questions to screen : {len(pool)}")
    print(f"Keep wrong          : {args.keep_wrong}")
    print(f"Max new tokens      : {max_new_tokens}")
    print(f"Temperature         : {temperature}")
    print(f"Save every          : {args.save_every}")
    print("=" * 70)
    print()

    if not pool:
        print("[screen] No new questions to process.")
        return

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    transcript_dir = (
        ROOT
        / cfg["paths"]["transcripts_dir"]
    )

    transcript_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = transcript_dir / "screen.jsonl"

    # -----------------------------------------------------------------
    # SCREEN
    # -----------------------------------------------------------------

    correct = []
    wrong = []
    audit = []

    start_time = time.time()

    print(
        f"[screen] Starting turn-1 screening of "
        f"{len(pool)} questions..."
    )

    print()

    for index, q in enumerate(
        tqdm(
            pool,
            desc="screen",
            unit="question",
        ),
        start=1,
    ):

        qid = q["id"]
        question = q["question"]

        question_start = time.time()

        try:

            # ---------------------------------------------------------
            # FORMAT PROMPT
            # ---------------------------------------------------------

            prompt = format_chat(
                model,
                [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                system_prompt,
            )

            # ---------------------------------------------------------
            # GENERATE
            # ---------------------------------------------------------

            response = generate(
                model,
                prompt,
                max_new_tokens,
                temperature,
            )

            # ---------------------------------------------------------
            # JUDGE
            # ---------------------------------------------------------

            is_correct = JUDGE(
                response,
                q["answer"],
                q.get("aliases"),
            )

            if is_correct:
                correct.append(q)
            else:
                wrong.append(q)

            # ---------------------------------------------------------
            # AUDIT
            # ---------------------------------------------------------

            audit.append(
                {
                    "qid": qid,
                    "initial_response": response,
                    "initially_correct": is_correct,
                }
            )

            # ---------------------------------------------------------
            # TIMING
            # ---------------------------------------------------------

            elapsed_question = time.time() - question_start
            elapsed_total = time.time() - start_time

            avg_time = elapsed_total / index
            remaining = avg_time * (len(pool) - index)

            print(
                f"\n[screen] {index}/{len(pool)} "
                f"| correct={len(correct)} "
                f"| wrong={len(wrong)} "
                f"| question={elapsed_question:.1f}s "
                f"| avg={avg_time:.1f}s "
                f"| ETA={remaining / 60:.1f} min"
            )

            # ---------------------------------------------------------
            # CHECKPOINT
            # ---------------------------------------------------------

            if (
                args.save_every > 0
                and index % args.save_every == 0
            ):

                save_checkpoint(
                    audit,
                    audit_path,
                )

                print(
                    f"[screen] checkpoint saved "
                    f"({index} questions)"
                )

        except KeyboardInterrupt:

            print()
            print(
                "[screen] Interrupted by user."
            )

            print(
                "[screen] Saving current progress..."
            )

            save_checkpoint(
                audit,
                audit_path,
            )

            print(
                f"[screen] Saved {len(audit)} "
                f"completed questions."
            )

            print(
                "[screen] You can safely stop here."
            )

            return

        except Exception as exc:

            print()
            print(
                f"[screen] ERROR on question {qid}:"
            )
            print(
                f"[screen] {type(exc).__name__}: {exc}"
            )

            # Save progress before continuing.
            save_checkpoint(
                audit,
                audit_path,
            )

            print(
                "[screen] Progress saved. "
                "Continuing with next question..."
            )

    # -----------------------------------------------------------------
    # FINAL AUDIT SAVE
    # -----------------------------------------------------------------

    save_checkpoint(
        audit,
        audit_path,
    )

    # -----------------------------------------------------------------
    # CREATE BATCH 2
    # -----------------------------------------------------------------

    batch2 = (
        correct
        + wrong[: args.keep_wrong]
    )

    batch2_path = (
        ROOT
        / "data"
        / "questions_batch2.jsonl"
    )

    write_jsonl(
        batch2_path,
        batch2,
    )

    # -----------------------------------------------------------------
    # CARRY FORWARD OLD ELIGIBLE QUESTIONS
    # -----------------------------------------------------------------

    carried = []

    if (
        args.old_questions
        and args.old_questions.exists()
    ):

        old_questions = read_jsonl(
            args.old_questions
        )

        carried = [
            q
            for q in old_questions
            if q.get("id") in old_eligible
        ]

    questions_all = (
        batch2
        + carried
    )

    questions_all_path = (
        ROOT
        / "data"
        / "questions_all.jsonl"
    )

    write_jsonl(
        questions_all_path,
        questions_all,
    )

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    total_time = time.time() - start_time

    accuracy = (
        len(correct) / len(pool)
        if pool
        else float("nan")
    )

    n_extract = sum(
        split_of(q["id"]) == "extract"
        for q in correct
    )

    n_heldout = (
        len(correct) - n_extract
    )

    print()
    print("=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)

    print(
        f"Questions screened : {len(pool)}"
    )

    print(
        f"Initially correct  : {len(correct)}"
    )

    print(
        f"Initially wrong    : {len(wrong)}"
    )

    print(
        f"Initial accuracy   : "
        f"{len(correct)}/{len(pool)} "
        f"({accuracy:.1%})"
    )

    print()
    print(
        f"Batch 2            : {len(batch2)}"
    )

    print(
        f"  Correct          : {len(correct)}"
    )

    print(
        f"  Wrong controls   : "
        f"{min(len(wrong), args.keep_wrong)}"
    )

    print()
    print(
        f"Eligible extract   : {n_extract}"
    )

    print(
        f"Eligible heldout   : {n_heldout}"
    )

    print()
    print(
        f"Carried forward    : {len(carried)}"
    )

    print(
        f"Questions all      : "
        f"{len(questions_all)}"
    )

    print()
    print(
        f"Total time         : "
        f"{total_time / 60:.2f} minutes"
    )

    if len(pool) > 0:
        print(
            f"Average/question   : "
            f"{total_time / len(pool):.2f} seconds"
        )

    print()
    print(
        f"Audit              : {audit_path}"
    )

    print(
        f"Batch 2            : {batch2_path}"
    )

    print(
        f"Questions all      : "
        f"{questions_all_path}"
    )

    print("=" * 70)

    print()
    print(
        "Next step:"
    )

    print(
        "Set questions to data/questions_batch2.jsonl "
        "for the baseline run."
    )

    print(
        "Before the ABLATED run, switch to "
        "data/questions_all.jsonl."
    )


if __name__ == "__main__":
    main()