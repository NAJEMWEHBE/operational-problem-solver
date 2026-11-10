"""End-to-end orchestrator loop with a mocked LLM. No API, no tokens."""

import sys
from pathlib import Path

import pytest

from ops_solver.config import RunConfig
from ops_solver.llm import TokenLedger
from ops_solver.models import JudgeVerdict, ManifestDraft
from ops_solver.orchestrator import solve


class MockLLM:
    """Stand-in for the real adapter. Deterministic outputs."""

    def __init__(self) -> None:
        self.ledger = TokenLedger()
        self._n = 0

    async def complete(self, *, model, system, user, max_tokens=4096, temperature=None, thinking=False):
        self._n += 1
        return f"```python\ndef solution():\n    return {self._n}\n```\nA clear approach."

    async def structured(
        self, *, model, system, user, schema, max_tokens=1500, temperature=None, thinking=False
    ):
        if schema is ManifestDraft:
            return ManifestDraft(constraints=["must be correct"], notes="from probe")
        if schema is JudgeVerdict:
            return JudgeVerdict(winner="A", reason="A is cleaner", a_score=8, b_score=6)
        return schema()

    async def judge(self, *, model, system, user, max_tokens=1500):
        # First solution (A) always wins -> deterministic leaderboard.
        return JudgeVerdict(winner="A", reason="A is cleaner", a_score=8, b_score=6)


async def test_full_solve_ships_a_winner(tmp_path: Path):
    cfg = RunConfig(
        problem="write a function that returns a number",
        cwd=tmp_path,
        runs_dir=tmp_path / "runs",
        intel_workers=2,
        exec_workers=3,
        lint=False,
    )
    report = await solve(cfg, llm=MockLLM())

    # Three workers produced artifacts.
    assert len(report.artifacts) == 3
    # All valid Python -> consensus fully met.
    assert report.consensus_met is True
    assert report.consensus_ratio == 1.0
    # A leaderboard with a winner that matches rank 1.
    assert report.leaderboard.entries
    assert report.winner_id == report.leaderboard.entries[0].worker_id
    # Winner was shipped to disk and the report persisted.
    assert report.winner_path and Path(report.winner_path).exists()
    assert (Path(report.winner_path) / "solution.py").exists()
    assert (cfg.runs_dir / report.run_id / "report.json").exists()


async def test_solve_handles_single_worker(tmp_path: Path):
    cfg = RunConfig(
        problem="trivial",
        cwd=tmp_path,
        runs_dir=tmp_path / "runs",
        intel_workers=1,
        exec_workers=1,
        lint=False,
    )
    report = await solve(cfg, llm=MockLLM())
    # One worker: no pairwise matches, but it still wins by default.
    assert len(report.artifacts) == 1
    assert report.winner_id == "Finch-01"


class FlakyLLM(MockLLM):
    """Broken Python on the first Phase-II attempt, valid on the second."""

    def __init__(self, workers: int) -> None:
        super().__init__()
        self._workers = workers

    async def complete(self, *, model, system, user, max_tokens=4096, temperature=None, thinking=False):
        self._n += 1
        if self._n <= self._workers:  # first `workers` complete() calls == attempt 0
            return "```python\ndef f(:\n    pass\n```"  # syntax error
        return "```python\ndef f():\n    return 1\n```"


class EmptyWorkerLLM(MockLLM):
    async def complete(self, *, model, system, user, max_tokens=4096, temperature=None, thinking=False):
        return ""  # every worker yields nothing


class DeadIntelLLM(MockLLM):
    async def structured(self, *, model, system, user, schema, max_tokens=1500, temperature=None, thinking=False):
        return None  # every Crow fails


async def test_consensus_reloop_then_succeeds(tmp_path: Path):
    cfg = RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs",
        intel_workers=1, exec_workers=2, max_reloops=1, lint=False,
    )
    report = await solve(cfg, llm=FlakyLLM(workers=2))
    assert report.attempts == 2          # re-looped after <50% stable on attempt 0
    assert report.consensus_met is True  # second attempt stabilized


async def test_all_workers_fail_raises_actionable_error(tmp_path: Path):
    cfg = RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs",
        intel_workers=1, exec_workers=2, max_reloops=0, lint=False,
    )
    with pytest.raises(RuntimeError, match="Execution phase produced 0"):
        await solve(cfg, llm=EmptyWorkerLLM())


async def test_all_intelligence_fail_raises_actionable_error(tmp_path: Path):
    cfg = RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs",
        intel_workers=2, exec_workers=2, max_reloops=0, lint=False,
    )
    with pytest.raises(RuntimeError, match="Intelligence phase produced 0"):
        await solve(cfg, llm=DeadIntelLLM())


class RaisingLLM(MockLLM):
    """Every call raises, the way a broken SDK or a bad key would."""

    async def structured(self, *, model, system, user, schema, max_tokens=1500, thinking=False):
        raise TypeError("got an unexpected keyword argument 'temperature'")

    async def complete(self, *, model, system, user, max_tokens=4096, thinking=False):
        raise TypeError("got an unexpected keyword argument 'temperature'")


async def test_intelligence_failure_names_the_real_cause(tmp_path: Path):
    cfg = RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs",
        intel_workers=2, exec_workers=2, max_reloops=0, lint=False,
    )
    with pytest.raises(RuntimeError, match="TypeError: got an unexpected keyword argument"):
        await solve(cfg, llm=RaisingLLM())


class RaisingWorkersLLM(MockLLM):
    async def complete(self, *, model, system, user, max_tokens=4096, thinking=False):
        raise PermissionError("403 model not allowed")


async def test_execution_failure_names_the_real_cause(tmp_path: Path):
    cfg = RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs",
        intel_workers=1, exec_workers=2, max_reloops=0, lint=False,
    )
    with pytest.raises(RuntimeError, match="PermissionError: 403 model not allowed"):
        await solve(cfg, llm=RaisingWorkersLLM())


class HalfDeafJudgeLLM(MockLLM):
    """Every other judge call returns no verdict (e.g. cut off at max_tokens)."""

    async def judge(self, *, model, system, user, max_tokens=1500):
        self._n += 1
        if self._n % 2:
            return None
        return JudgeVerdict(winner="A", reason="r", a_score=8, b_score=6)


async def test_partly_dropped_judge_matches_are_reported(tmp_path: Path):
    cfg = RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs",
        intel_workers=1, exec_workers=3, max_reloops=0, lint=False, judge_both_orders=False,
    )
    report = await solve(cfg, llm=HalfDeafJudgeLLM())
    assert 0 < len(report.leaderboard.matches) < 3
    assert any("judge matches dropped" in w for w in report.warnings)


async def test_runs_started_in_the_same_second_get_their_own_folders(tmp_path: Path):
    cfg = RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs",
        intel_workers=1, exec_workers=1, max_reloops=0, lint=False,
    )
    first = await solve(cfg, llm=MockLLM())
    second = await solve(cfg, llm=MockLLM())
    assert first.run_id != second.run_id
    assert len(list((tmp_path / "runs").iterdir())) == 2


class PythonThenJsLLM(MockLLM):
    """Attempt 0: broken Python plus notes. Attempt 1: JavaScript, no notes."""

    async def complete(self, *, model, system, user, max_tokens=4096, thinking=False):
        self._n += 1
        if self._n == 1:
            return "```python\ndef f(:\n```\nNotes from the first try."
        return "```javascript\nconsole.log(1);\n```"


async def test_a_reloop_leaves_no_stale_files_in_the_shipped_winner(tmp_path: Path):
    cfg = RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs",
        intel_workers=1, exec_workers=1, max_reloops=1, lint=False,
    )
    report = await solve(cfg, llm=PythonThenJsLLM())
    assert report.attempts == 2
    shipped = sorted(p.name for p in Path(report.winner_path).iterdir())
    assert shipped == ["solution.js"]


# --- --out writes exactly the path the operator named ------------------------


def _out_cfg(tmp_path: Path, out: Path, force: bool = False) -> RunConfig:
    return RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs",
        intel_workers=1, exec_workers=1, max_reloops=0, lint=False,
        out_path=out, force=force,
    )


def _link(link: Path, target: Path, is_dir: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        if is_dir and sys.platform == "win32":  # a junction needs no privilege
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
            return
        pytest.skip("this OS/account can't create symlinks")


async def test_out_writes_the_winner_and_respects_force(tmp_path: Path):
    out = tmp_path / "sub" / "solution.py"
    report = await solve(_out_cfg(tmp_path, out), llm=MockLLM())
    assert "def solution" in out.read_text(encoding="utf-8")
    assert not report.warnings

    out.write_text("mine\n", encoding="utf-8")
    report = await solve(_out_cfg(tmp_path, out), llm=MockLLM())
    assert out.read_text(encoding="utf-8") == "mine\n"
    assert any("not overwritten" in w for w in report.warnings)

    await solve(_out_cfg(tmp_path, out, force=True), llm=MockLLM())
    assert "def solution" in out.read_text(encoding="utf-8")


async def test_out_does_not_write_through_a_dangling_link(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    target = tmp_path / "elsewhere" / "planted.py"
    _link(proj / "out.py", target)
    report = await solve(_out_cfg(proj, proj / "out.py"), llm=MockLLM())
    assert not target.exists() and not target.parent.exists()
    assert (proj / "out.py").is_symlink()
    assert any("not overwritten" in w for w in report.warnings)


async def test_out_force_replaces_the_link_not_its_target(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    target = tmp_path / "keep.txt"
    target.write_text("keep\n", encoding="utf-8")
    _link(proj / "out.py", target)
    await solve(_out_cfg(proj, proj / "out.py", force=True), llm=MockLLM())
    assert target.read_text(encoding="utf-8") == "keep\n"
    assert not (proj / "out.py").is_symlink()
    assert "def solution" in (proj / "out.py").read_text(encoding="utf-8")


async def test_out_does_not_leave_the_working_directory_through_a_linked_folder(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _link(proj / "src", elsewhere, is_dir=True)
    report = await solve(_out_cfg(proj, proj / "src" / "x.py", force=True), llm=MockLLM())
    assert list(elsewhere.iterdir()) == []
    assert any("through a link" in w for w in report.warnings)


async def test_out_never_writes_into_a_folder(tmp_path: Path):
    (tmp_path / "taken").mkdir()
    report = await solve(_out_cfg(tmp_path, tmp_path / "taken", force=True), llm=MockLLM())
    assert list((tmp_path / "taken").iterdir()) == []
    assert any("not a regular file" in w for w in report.warnings)
