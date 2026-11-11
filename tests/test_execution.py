"""Solution extraction from worker responses. Offline."""

import pytest

from ops_solver.config import RunConfig
from ops_solver.execution import _filename, extract_solution, run_workers
from ops_solver.models import ContextManifest


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


@pytest.mark.parametrize(
    ("tag", "language"),
    [("py", "python"), ("PY", "python"), ("python3", "python"), ("js", "javascript"),
     ("ts", "typescript"), ("rs", "rust"), ("c++", "cpp"), ("c#", "csharp")],
)
def test_short_fence_tags_name_the_language(tag, language):
    _, lang, _ = extract_solution(f"```{tag}\nx\n```", "go")
    assert lang == language


@pytest.mark.parametrize("shell", ["sh", "zsh", "bash"])
def test_a_shell_dialect_keeps_its_name_and_is_a_sh_file(shell):
    """zsh is not bash: the judge and the report get the dialect that was asked
    for, as a fence tag or as the default; only the file name is shared."""
    assert extract_solution(f"```{shell}\necho hi\n```", "go")[1] == shell
    assert extract_solution("echo hi", shell)[1] == shell
    assert _filename(shell) == "solution.sh"


@pytest.mark.parametrize("text", ["```\ndef f(:\n    pass\n```", "def f(:\n    pass"])
def test_a_short_default_language_is_named_in_full(text):
    """`--lang python3` with an untagged fence, or no fence: the artifact is
    "python", the name the consensus gate's syntax check looks for."""
    _, lang, _ = extract_solution(text, "python3")
    assert lang == "python"


class _OneFenceLLM:
    """Stub LLM: every worker answers with the same ```py block (no API call)."""

    async def complete(self, *, model, system, user, max_tokens):
        return "```py\ndef test_ok():\n    assert True\n```\nA tiny test."


async def test_py_fence_is_saved_as_a_py_file(tmp_path):
    cfg = RunConfig(problem="p", exec_workers=1)
    arts = await run_workers(
        _OneFenceLLM(), cfg, ContextManifest(problem="p", language="auto"), tmp_path
    )
    assert [a.filename for a in arts] == ["solution.py"]
    assert arts[0].language == "python"
    assert (tmp_path / "Finch-01" / "solution.py").is_file()
