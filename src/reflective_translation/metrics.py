"""Metrics: BLEU wrapper and COMET wrapper (safe lazy loader).

COMET is supported if the `comet` package is installed and available; otherwise the comet_score function returns None.
"""
from typing import Optional, Sequence

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np


def compute_bleu(references: Sequence[Sequence[str]], hypotheses: Sequence[str]) -> float:
    """Compute corpus BLEU (smoothing) where references is list of list of tokens lists.

    references: [[ref1_tokens], [ref2_tokens], ...]
    hypotheses: [hyp_tokens, ...]
    """
    # references must be list of list of tokens
    smoothie = SmoothingFunction().method4
    # corpus_bleu expects references as list of list of reference-token-lists
    try:
        score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    except Exception:
        # safe fallback
        score = 0.0
    return float(score)


class CometWrapper:
    def __init__(self, model_name: str = "Unloaded"):
        self.model = None
        self.model_name = model_name
        try:
            # Local import to avoid hard dep for users who don't need COMET
            from comet import download_model, load_from_checkpoint

            # Try to load if available
            self.model = load_from_checkpoint(model_name)
        except Exception:
            self.model = None

    def score(self, sources, references, hypotheses) -> Optional[float]:
        """Return average COMET score across examples or None if COMET not available."""
        if self.model is None:
            return None
        try:
            data = [
                {"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)
            ]
            results = self.model.predict(data)
            # results is list of dicts with 'score'
            scores = [float(r.get("score", 0.0)) for r in results]
            return float(np.mean(scores)) if scores else None
        except Exception:
            return None
