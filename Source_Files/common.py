"""
Shared utilities, v2.

Compatible with:
- Qwen/Qwen2.5-7B-Instruct
- google/gemma-2-2b-it
- Other TransformerLens / Hugging Face chat models

Includes:
- Active-model overlay
- Frozen paraphrase assignment
- GPU/CPU device detection
- Chat-template compatibility for Gemma/Qwen
- Generation
- Residual activation extraction
- JSONL I/O
- Heuristic judging
"""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import torch
import yaml
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# CONFIG
# =====================================================================

def load_config(path: str | Path | None = None) -> dict:
    """Load the main configuration and apply active_model.yaml."""

    if path is None:
        for cand in (
            ROOT / "configs/config_v2.yaml",
            ROOT / "configs/config.yaml",
        ):
            if cand.exists():
                path = cand
                break

    if path is None:
        raise FileNotFoundError("No configuration file found.")

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # -------------------------------------------------------------
    # Active model overlay
    # -------------------------------------------------------------

    active = ROOT / "configs/active_model.yaml"

    if active.exists():

        with open(active, "r", encoding="utf-8") as f:
            a = yaml.safe_load(f) or {}

        cfg.setdefault("model", {})
        cfg.setdefault("paths", {})

        if "name" in a:
            cfg["model"]["name"] = a["name"]

        if "dtype" in a:
            cfg["model"]["dtype"] = a["dtype"]

        slug = a.get("slug")

        if slug:

            for key in (
                "transcripts_dir",
                "activations_dir",
                "directions_dir",
                "figures_dir",
            ):
                if key in cfg["paths"]:
                    cfg["paths"][key] = (
                        f"outputs/{slug}/"
                        f"{Path(cfg['paths'][key]).name}"
                    )

            if a.get("questions"):
                cfg["paths"]["questions"] = a["questions"]

            cfg["_slug"] = slug

    return cfg


# =====================================================================
# DEVICE
# =====================================================================

def resolve_device(cfg: dict) -> str:

    dev = cfg["model"].get("device", "auto")

    if dev != "auto":
        return dev

    if torch.cuda.is_available():
        return "cuda"

    if (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


# =====================================================================
# PARAPHRASES
# =====================================================================

def _load_paraphrases(cfg: dict) -> dict | None:

    tf = cfg.get("templates_file")

    if tf:
        path = ROOT / tf
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    return None


def paraphrase_index(qid: str, ptype: str, n: int = 5) -> int:

    return int(hashlib.md5(f"{qid}:{ptype}".encode()).hexdigest(), 16) % n


def get_template(cfg: dict, qid: str, ptype: str) -> tuple[str, int]:

    para = _load_paraphrases(cfg)

    if para and ptype in para:

        variants = para[ptype]

        forced = os.environ.get("V2_FORCE_PARAPHRASE")

        if forced is not None:
            idx = int(forced)
        else:
            idx = paraphrase_index(qid, ptype, len(variants))

        idx = idx % len(variants)

        return variants[idx], idx

    return cfg["eval"]["pushback_templates"][ptype], 0


def pushback_types(cfg: dict) -> list[str]:

    para = _load_paraphrases(cfg)

    if para:
        return sorted(para.keys())

    return sorted(cfg["eval"]["pushback_templates"].keys())


# =====================================================================
# MODEL
# =====================================================================

def load_model(cfg: dict | None = None):

    from transformer_lens import HookedTransformer

    cfg = cfg or load_config()

    device = resolve_device(cfg)

    dtype_name = cfg["model"].get("dtype", "float32")
    dtype = getattr(torch, dtype_name, torch.float32)

    # CPU safety: half precision is slow/unsupported on CPU
    if device == "cpu" and dtype != torch.float32:
        print("[common] CPU detected -> forcing float32")
        dtype = torch.float32

    print()
    print("=" * 70)
    print("MODEL / DEVICE INFORMATION")
    print("=" * 70)
    print(f"Model : {cfg['model']['name']}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        try:
            print(f"GPU   : {torch.cuda.get_device_name(0)}")
            print(f"CUDA  : {torch.version.cuda}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"VRAM  : {vram:.2f} GB")
        except Exception:
            pass

    print("=" * 70)
    print()

    model = HookedTransformer.from_pretrained(
        cfg["model"]["name"],
        device=device,
        dtype=dtype,
    )

    model.eval()

    print(
        f"[common] loaded {cfg['model']['name']} on {device} "
        f"({model.cfg.n_layers} layers, d_model={model.cfg.d_model})"
    )

    return model


# =====================================================================
# CHAT TEMPLATE
# =====================================================================

def format_chat(
    model_or_tokenizer,
    messages: list[dict],
    system_prompt: str,
) -> str:
    """
    Convert messages into the model's chat format.

    Qwen accepts a separate "system" message.
    Gemma does NOT, so for Gemma the system text is glued
    onto the start of the first user message instead.
    """

    tok = getattr(model_or_tokenizer, "tokenizer", model_or_tokenizer)

    # Copy so we never modify the caller's list
    msgs = [dict(m) for m in messages]

    # No system prompt -> just format the messages
    if not (system_prompt and system_prompt.strip()):
        return tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Try the normal way (works for Qwen)
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": system_prompt}] + msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Model has no system role (Gemma) -> merge into first user turn
    except Exception:
        if msgs and msgs[0]["role"] == "user":
            msgs[0]["content"] = (
                system_prompt.strip() + "\n\n" + msgs[0]["content"]
            )

        return tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )


# =====================================================================
# STOP STRINGS
# =====================================================================

STOP_STRINGS = (
    "<|im_end|>",       # Qwen
    "<|endoftext|>",    # Qwen / GPT-style
    "<|eot_id|>",       # Llama 3
    "<|end_of_text|>",  # Llama 3
    "<end_of_turn>",    # Gemma
    "<eos>",            # Gemma
)


# =====================================================================
# GENERATION
# =====================================================================

@torch.no_grad()
def generate(
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    fwd_hooks: list | None = None,
) -> str:

    # prepend_bos=False because the chat template already adds <bos>
    toks = model.to_tokens(prompt, prepend_bos=False)

    n_in = toks.shape[1]

    def _run():
        return model.generate(
            toks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            verbose=False,
        )

    if fwd_hooks:
        with model.hooks(fwd_hooks=fwd_hooks):
            out = _run()
    else:
        out = _run()

    text = model.tokenizer.decode(
        out[0, n_in:],
        skip_special_tokens=False,
    )

    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop)[0]

    return text.strip()


# =====================================================================
# LAST TOKEN RESIDUAL
# =====================================================================

@torch.no_grad()
def last_token_resid(model, prompt: str) -> torch.Tensor:

    toks = model.to_tokens(prompt, prepend_bos=False)

    names = lambda n: n.endswith("hook_resid_pre")

    _, cache = model.run_with_cache(toks, names_filter=names)

    acts = torch.stack(
        [
            cache[f"blocks.{l}.hook_resid_pre"][0, -1]
            for l in range(model.cfg.n_layers)
        ]
    )

    return acts.float().cpu()


# =====================================================================
# TRANSCRIPT
# =====================================================================

class Transcript(BaseModel):

    qid: str
    question: str
    correct_answer: str
    pushback_type: str
    condition: str
    initial_response: str
    final_response: str | None = None
    initially_correct: bool | None = None
    finally_correct: bool | None = None
    capitulated: bool | None = None
    judge: str = "heuristic"
    paraphrase_idx: int | None = None
    recovery_response: str | None = None   # Phase 3: model's third reply
    recovered: bool | None = None          # Phase 3: back to correct answer?


# =====================================================================
# JSONL I/O
# =====================================================================

def read_jsonl(path: str | Path) -> list[dict]:

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict | BaseModel]):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, BaseModel):
                r = r.model_dump()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[common] wrote {path}")


# =====================================================================
# JUDGE
# =====================================================================

def heuristic_judge(
    response: str,
    correct_answer: str,
    aliases: list[str] | None = None,
) -> bool:

    if response is None:
        return False

    r = response.lower()

    candidates = [correct_answer] + list(aliases or [])

    return any(
        c.lower().strip() in r
        for c in candidates
        if c and c.strip()
    )


JUDGE: Callable[..., bool] = heuristic_judge