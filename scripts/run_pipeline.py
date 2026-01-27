"""CLI to run the reflective translation pipeline.

This script supports a dry-run mode that does not require API keys and produces deterministic stub translations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.reflective_translation.pipeline import run_pipeline
from src.reflective_translation.config import PATHS


def tiny_dataset(n=10):
    # small synthetic dataset for dry runs
    for i in range(n):
        src = f"This is a test sentence number {i}."
        ref = f"Reference translation {i}."
        yield src, ref


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tiny")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model_first", default="gpt-3.5")
    parser.add_argument("--model_second", default="haiku-3.5")
    parser.add_argument("--bleu_threshold", type=float, default=None)
    parser.add_argument("--outdir", default=str(PATHS.csv))
    parser.add_argument("--dry_run", action="store_true", help="Run in dry-run mode without hitting APIs")

    args = parser.parse_args(argv)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if args.dataset == "tiny":
        dataset = tiny_dataset(args.n)
    else:
        # Placeholder: user should implement dataset loading from HuggingFace as desired
        print("Only 'tiny' dataset implemented in this CLI. For real datasets, implement HF loader.")
        dataset = tiny_dataset(args.n)

    out_csv = run_pipeline(dataset, out_csv=Path(args.outdir) / "all_translation_results.csv", model_first=args.model_first, model_second=args.model_second, bleu_threshold=args.bleu_threshold, dry_run=args.dry_run or True)
    print(f"Wrote results to {out_csv}")


if __name__ == "__main__":
    main()
