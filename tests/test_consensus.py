"""QA Layer 1 consensus gate + objective scoring. Offline (lint/tests disabled)."""

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
