"""Main pipeline: first pass, evaluate, optional reflection + second pass, save CSVs."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional, Dict, Any

from .config import PATHS, DEFAULTS
from .prompts import baseline_prompt, reflection_prompt
from .metrics import compute_bleu, CometWrapper

import random


def _ensure_dirs():
    PATHS.csv.mkdir(parents=True, exist_ok=True)
    PATHS.figures.mkdir(parents=True, exist_ok=True)


def model_infer(model_name: str, prompt: str, dry_run: bool = False) -> str:
    """Simple model inference abstraction.

    If dry_run=True or no API keys present, returns a deterministic stub translation.
    Real integrations (OpenAI/Anthropic) should be implemented by the user —
    this function keeps the interface minimal and safe (no keys hardcoded).
    """
    if dry_run or os.environ.get("OPENAI_API_KEY") is None and os.environ.get("ANTHROPIC_API_KEY") is None:
        # deterministic stub: echo first 120 characters with marker
        return f"[DRY_TRANSLATION] {prompt.strip()[:120]}"

    # Placeholder for real API calls.
    raise NotImplementedError("model_infer: implement provider-specific calls using env API keys")


def process_translation_pair(source: str, reference: str, model_first: str = "gpt-3.5", model_second: str = "haiku-3.5", bleu_threshold: float = None, comet_threshold: Optional[float] = None, dry_run: bool = True) -> Dict[str, Any]:
    """Process one example: first pass, compute BLEU, optional reflection and second pass.

    Returns a dict row with fields saved to CSV.
    """
    if bleu_threshold is None:
        bleu_threshold = DEFAULTS["bleu_threshold"]

    # First pass
    prompt1 = baseline_prompt(source)
    first_translation = model_infer(model_first, prompt1, dry_run=dry_run)

    # BLEU expects tokenized inputs; we use simplistic whitespace tokenization for pipeline.
    ref_tokens = [reference.split()]
    hyp_tokens = first_translation.split()
    bleu1 = compute_bleu([ref_tokens], [hyp_tokens])

    # COMET scoring (per-example wrapper expects lists)
    comet = CometWrapper()
    comet1 = comet.score([source], [reference], [first_translation])

    needs_reflection = False
    if bleu1 < bleu_threshold:
        needs_reflection = True

    second_translation = ""
    bleu2 = None
    comet2 = None

    if needs_reflection:
        reflect_prompt = reflection_prompt(source, first_translation)
        second_translation = model_infer(model_second, reflect_prompt, dry_run=dry_run)
        bleu2 = compute_bleu([ref_tokens], [second_translation.split()])
        comet2 = comet.score([source], [reference], [second_translation])

    row = {
        "source": source,
        "reference": reference,
        "first_translation": first_translation,
        "bleu_first": bleu1,
        "comet_first": comet1,
        "second_translation": second_translation,
        "bleu_second": bleu2,
        "comet_second": comet2,
        "reflected": needs_reflection,
    }

    return row


def run_pipeline(dataset_iterable, out_csv: Optional[Path] = None, model_first: str = "gpt-3.5", model_second: str = "haiku-3.5", bleu_threshold: float = None, dry_run: bool = True):
    _ensure_dirs()
    out_csv = out_csv or PATHS.csv / "all_translation_results.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = None
        for src, ref in dataset_iterable:
            row = process_translation_pair(src, ref, model_first=model_first, model_second=model_second, bleu_threshold=bleu_threshold, dry_run=dry_run)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)

    return out_csv
