"""elo.run_tournament async path with a mock judge + empty-leaderboard guard. Offline."""

from ops_solver.config import RunConfig
from ops_solver.elo import EloEngine, run_tournament
from ops_solver.models import ContextManifest, JudgeVerdict, WorkerArtifact


def _arts(n):
    return [
        WorkerArtifact(
            worker_id=f"W{i}", strategy="s", language="python",
            filename="solution.py", code=f"x={i}",
        )
        for i in range(n)
    ]


def _manifest():
    return ContextManifest(problem="p", language="python")


class JudgeAWins:
    async def judge(self, *, model, system, user, max_tokens=1500):
        return JudgeVerdict(winner="A", reason="a", a_score=8, b_score=4)


class JudgeAllFail:
    async def judge(self, *, model, system, user, max_tokens=1500):
        return None


async def test_first_of_each_pair_wins():
    board = await run_tournament(JudgeAWins(), RunConfig(problem="p"), _arts(3), _manifest(), {})
    assert board.winner == "W0"            # W0 is "A" in every pair it appears in first
    assert len(board.matches) == 3         # 3 unique pairs, 1 round


async def test_all_judges_fail_no_matches_no_crash():
    board = await run_tournament(JudgeAllFail(), RunConfig(problem="p"), _arts(3), _manifest(), {})
    assert board.matches == []             # every verdict None -> pairing skipped
    assert board.winner is not None        # still returns a (tied) board rather than crashing


async def test_single_artifact_wins_by_default():
    board = await run_tournament(JudgeAWins(), RunConfig(problem="p"), _arts(1), _manifest(), {})
    assert board.winner == "W0"
    assert board.matches == []


def test_empty_engine_leaderboard_no_crash():
    board = EloEngine([]).leaderboard()
    assert board.entries == [] and board.winner is None
