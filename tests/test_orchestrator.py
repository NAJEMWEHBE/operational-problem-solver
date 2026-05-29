"""End-to-end orchestrator loop with a mocked LLM. No API, no tokens."""

from pathlib import Path

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
