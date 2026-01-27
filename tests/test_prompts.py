from src.reflective_translation import prompts


def test_baseline_prompt_returns_string():
    s = prompts.baseline_prompt("Hello world.")
    assert isinstance(s, str)
    assert "TRANSLATION" in s


def test_reflection_prompt_structure():
    s = prompts.reflection_prompt("Hello", "Sawubona")
    assert isinstance(s, str)
    assert "OUTPUT" in s or "TRANSLATION" in s
