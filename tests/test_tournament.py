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
    """Pure position bias: always picks whichever solution is shown as 'A'."""
    async def judge(self, *, model, system, user, max_tokens=1500):
        return JudgeVerdict(winner="A", reason="a", a_score=8, b_score=4)


class JudgePrefersW0:
    """Consistent judge: prefers worker W0 no matter which side it is shown on."""
    async def judge(self, *, model, system, user, max_tokens=1500):
        a_is_w0 = "id=W0," in user.split("Solution B")[0]
        return JudgeVerdict(
            winner="A" if a_is_w0 else "B", reason="w0",
            a_score=9 if a_is_w0 else 3, b_score=3 if a_is_w0 else 9,
        )


class JudgeAllFail:
    async def judge(self, *, model, system, user, max_tokens=1500):
        return None


async def test_position_bias_is_neutralized():
    # A judge that always picks the first-shown solution must NOT decide matches:
    # judging both orders + averaging collapses every pair to a 0.5 tie.
    board = await run_tournament(JudgeAWins(), RunConfig(problem="p"), _arts(3), _manifest(), {})
    assert len(board.matches) == 3
    assert all(m.score_a == 0.5 for m in board.matches)               # every match a tie
    assert all(e.wins == 0 and e.losses == 0 for e in board.entries)  # bias won nothing


async def test_consistent_judge_still_picks_clear_winner():
    # A real, position-independent preference still yields a decisive winner.
    board = await run_tournament(JudgePrefersW0(), RunConfig(problem="p"), _arts(3), _manifest(), {})
    assert board.winner == "W0"
    assert board.entries[0].wins > 0                                  # genuine wins, not tie-break


async def test_opt_out_restores_single_order():
    # judge_both_orders=False keeps the old single-call behaviour (first-shown wins).
    cfg = RunConfig(problem="p", judge_both_orders=False)
    board = await run_tournament(JudgeAWins(), cfg, _arts(3), _manifest(), {})
    assert board.winner == "W0"
    assert any(m.score_a == 1.0 for m in board.matches)              # un-averaged: A wins outright


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
