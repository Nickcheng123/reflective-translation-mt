"""Ablation utilities: threshold ablation and prompting ablation.

Reads CSV results (if present) or can recompute simple ablations from a provided dataset and the pipeline.
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .config import PATHS


def threshold_ablation(csv_path: Path = None, thresholds: Optional[List[float]] = None, out_csv: Path = None):
    csv_path = csv_path or PATHS.csv / "all_translation_results.csv"
    out_csv = out_csv or PATHS.csv / "threshold_ablation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    thresholds = thresholds or [0.0, 0.1, 0.2, 0.3, 0.4]
    rows = []
    for t in thresholds:
        # consider second attempt only when bleu_first < t
        filtered = df[df["bleu_first"] < t]
        mean_bleu = filtered["bleu_second"].mean()
        mean_comet = filtered["comet_second"].mean()
        rows.append({"threshold": t, "mean_bleu_second": mean_bleu, "mean_comet_second": mean_comet, "n": len(filtered)})

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def prompting_ablation(csv_paths: List[Path], labels: List[str], out_csv: Path = None):
    out_csv = out_csv or PATHS.csv / "prompting_ablation.csv"
    rows = []
    for p, label in zip(csv_paths, labels):
        if not p.exists():
            continue
        df = pd.read_csv(p)
        rows.append({"label": label, "mean_bleu_first": df["bleu_first"].mean(), "mean_bleu_second": df["bleu_second"].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out
