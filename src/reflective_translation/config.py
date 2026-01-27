from dataclasses import dataclass

@dataclass(frozen=True)
class Thresholds:
    bleu: float = 0.30
    comet: float = 0.70

@dataclass(frozen=True)
class Models:
    first: str = "gpt-3.5-turbo"
    second: str = "gpt-4o-mini"

LANG_MAP = {
    "zu": "isiZulu",
    "xh": "isiXhosa",
}