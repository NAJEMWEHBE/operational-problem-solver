"""Solution extraction from worker responses. Offline."""

from ops_solver.execution import extract_solution


def test_picks_largest_code_block():
    text = "```python\nx=1\n```\nmid\n```python\ndef big():\n    return 42\n```"
    code, lang, expl = extract_solution(text, "python")
    assert "def big()" in code
    assert lang == "python"
    assert "mid" in expl


def test_no_fence_returns_whole_text_as_code():
    code, lang, expl = extract_solution("just prose, no code", "go")
    assert code == "just prose, no code"
    assert lang == "go"
    assert expl == ""


def test_empty_fence_tag_uses_default_language():
    code, lang, _ = extract_solution("```\nhello\n```", "rust")
    assert lang == "rust"
    assert code == "hello"


def test_fence_tag_overrides_default_language():
    code, lang, _ = extract_solution("```javascript\nconsole.log(1)\n```", "python")
    assert lang == "javascript"
    assert "console.log" in code
