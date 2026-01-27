"""Prompt templates for the reflective translation pipeline.

Important: prompts must not ask models to reveal their internal chain-of-thought or private reasoning.
Provide clear instructions for producing an improved translation or a short reflection (concise errors and suggestions).
"""
from typing import List


def baseline_prompt(source: str, src_lang: str = "en", tgt_lang: str = "zu") -> str:
    return (
        f"Translate the following text from {src_lang} to {tgt_lang}. "
        + f"Be concise and prefer natural, fluent phrasing.\n\nSOURCE:\n{source}\n\nTRANSLATION:"
    )


def reflection_prompt(source: str, first_translation: str, src_lang: str = "en", tgt_lang: str = "zu") -> str:
    """Ask model to identify errors or ways to improve its first translation and then produce a revised translation.

    This prompt asks for a short reflection (errors and suggestions) and an improved translation.
    It does NOT ask the model to reveal its chain-of-thought; instead it requests a short, structured note.
    """
    return (
        f"You are given a source sentence and a previous translation. "
        + f"First, in 1-2 short bullets, list specific issues with the previous translation (fluency, terminology, omissions). "
        + f"Second, provide an improved translation.\n\nSOURCE:\n{source}\n\nPREVIOUS TRANSLATION:\n{first_translation}\n\nOUTPUT (Bullets then TRANSLATION):"
    )


def few_shot_prompt(source: str, examples: List[dict], src_lang: str = "en", tgt_lang: str = "zu") -> str:
    s = "Use the examples as a guide. For the new source, produce a translation that follows the shown style.\n\n"
    for ex in examples:
        s += f"SOURCE: {ex['source']}\nTRANSLATION: {ex['translation']}\n---\n"
    s += f"NEW SOURCE: {source}\nTRANSLATION:"
    return s


def cot_prompt_placeholder(source: str) -> str:
    # We avoid requesting chain-of-thought. Provide a short step-based checklist instead.
    return (
        "Provide a short checklist of translation checks (e.g., terminology, verb agreement), then give the translation.\n\n"
        f"SOURCE:\n{source}\n\nCHECKLIST AND TRANSLATION:"
    )
