"""QA Layer 1 consensus gate + objective scoring. Offline; the --test-cmd tests
run tiny local Python commands."""

import shutil
import sys
import time

import pytest

from ops_solver import qa_consensus
from ops_solver.config import RunConfig
from ops_solver.models import EvalResult, WorkerArtifact
from ops_solver.qa_consensus import consensus, evaluate, objective_score


def _cfg(**kw):
    return RunConfig(problem="x", lint=False, **kw)


def test_consensus_empty():
    assert consensus([], 0.5) == ([], 0.0, False)


def test_consensus_threshold_met_and_missed():
    def ev(wid, stable):
        return EvalResult(worker_id=wid, stable=stable, syntax_ok=stable)

    # 1 of 2 stable -> 0.5, meets 0.5 threshold.
    ids, ratio, met = consensus([ev("a", True), ev("b", False)], 0.5)
    assert ids == ["a"] and ratio == 0.5 and met is True

    # 1 of 3 stable -> 0.33, misses.
    _, ratio2, met2 = consensus([ev("a", True), ev("b", False), ev("c", False)], 0.5)
    assert round(ratio2, 2) == 0.33 and met2 is False


def test_evaluate_valid_python_is_stable():
    art = WorkerArtifact(
        worker_id="Finch-01",
        strategy="simple",
        language="python",
        filename="solution.py",
        code="def add(a, b):\n    return a + b\n",
        workspace="",
    )
    res = evaluate(art, _cfg())
    assert res.syntax_ok is True
    assert res.stable is True
    assert res.tests_ran is False


def test_evaluate_broken_python_is_unstable():
    art = WorkerArtifact(
        worker_id="Finch-02",
        strategy="simple",
        language="python",
        filename="solution.py",
        code="def add(a, b)\n    return a + b\n",  # missing colon
        workspace="",
    )
    res = evaluate(art, _cfg())
    assert res.syntax_ok is False
    assert res.stable is False


def test_objective_score_rewards_passing_tests():
    passing = EvalResult(
        worker_id="a", stable=True, syntax_ok=True, tests_ran=True, tests_passed=True, lint_score=1.0
    )
    not_run = EvalResult(worker_id="b", stable=True, syntax_ok=True, lint_score=1.0)
    failing = EvalResult(
        worker_id="c", stable=False, syntax_ok=True, tests_ran=True, tests_passed=False, lint_score=1.0
    )
    assert objective_score(passing) > objective_score(not_run) > objective_score(failing)


# --- --test-cmd runs in the worker folder (executes code; local, tiny) -------

PY = f'"{sys.executable}"'


def _artifact(ws, code: str) -> WorkerArtifact:
    (ws / "solution.py").write_text(code, encoding="utf-8")
    return WorkerArtifact(
        worker_id="Finch-01", strategy="s", language="python",
        filename="solution.py", code=code, workspace=str(ws),
    )


def test_test_cmd_file_placeholder_points_pytest_at_the_solution(tmp_path):
    code = "def add(a, b):\n    return a + b\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    art = _artifact(tmp_path, code)
    res = evaluate(art, _cfg(test_cmd=f"{PY} -m pytest -q -p no:cacheprovider {{file}}"))
    assert res.tests_ran is True
    assert res.tests_passed is True, res.test_summary


def test_test_cmd_timeout_kills_the_whole_process_tree(tmp_path, monkeypatch):
    # The shell starts python, which starts a grandchild that holds the output
    # pipes for 20 s. A timeout must not wait for it.
    monkeypatch.setattr(qa_consensus, "_TEST_TIMEOUT_S", 2)
    grandchild = (
        "import subprocess, sys; "
        "subprocess.run([sys.executable, '-c', 'import time; time.sleep(20)'])"
    )
    art = _artifact(tmp_path, "x = 1\n")
    start = time.monotonic()
    res = evaluate(art, _cfg(test_cmd=f'{PY} -c "{grandchild}"'))
    elapsed = time.monotonic() - start
    assert res.tests_passed is False
    assert "timed out" in res.test_summary
    assert elapsed < 10, f"timeout took {elapsed:.1f}s: the child tree was not killed"


# --- lint score counts real issues (needs ruff on PATH, as in CI) -----------


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")
@pytest.mark.parametrize(("issues", "score"), [(1, 0.95), (2, 0.9), (3, 0.85)])
def test_lint_score_drops_per_issue_and_ignores_the_users_ruff_config(tmp_path, issues, score):
    # a project config above the runs dir that would silence F821
    (tmp_path / "pyproject.toml").write_text('[tool.ruff.lint]\nignore = ["F821"]\n', encoding="utf-8")
    ws = tmp_path / "runs" / "Finch-01"
    ws.mkdir(parents=True)
    code = "".join(f"a{i} = undefined_{i}\n" for i in range(issues))  # one F821 per line
    res = evaluate(_artifact(ws, code), RunConfig(problem="x", lint=True))
    assert res.lint_ran is True and res.lint_ok is False
    assert res.lint_score == score
