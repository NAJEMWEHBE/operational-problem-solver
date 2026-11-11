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


class JudgePrefersW0ButForwardFails(JudgePrefersW0):
    """Only answers when W0 is NOT shown first, i.e. only the reverse order of (W0, W1)."""

    async def judge(self, *, model, system, user, max_tokens=1500):
        if "id=W0," in user.split("Solution B")[0]:
            return None
        return await super().judge(model=model, system=system, user=user, max_tokens=max_tokens)


async def test_reverse_verdict_is_used_when_the_forward_call_fails():
    board = await run_tournament(
        JudgePrefersW0ButForwardFails(), RunConfig(problem="p"), _arts(2), _manifest(), {}
    )
    assert len(board.matches) == 1
    match = board.matches[0]
    assert (match.a, match.b, match.score_a) == ("W0", "W1", 1.0)
    assert board.winner == "W0"


class JudgePrefersW0ButForwardRaises(JudgePrefersW0):
    """The forward order of (W0, W1) raises, as an API error (429, 5xx, timeout) would."""

    async def judge(self, *, model, system, user, max_tokens=1500):
        if "id=W0," in user.split("Solution B")[0]:
            raise RuntimeError("simulated API error")
        return await super().judge(model=model, system=system, user=user, max_tokens=max_tokens)


async def test_reverse_verdict_is_kept_when_the_forward_call_raises():
    # The other order's verdict is already paid for: an exception in one order
    # must not throw it away (same as a None verdict above).
    board = await run_tournament(
        JudgePrefersW0ButForwardRaises(), RunConfig(problem="p"), _arts(2), _manifest(), {}
    )
    assert len(board.matches) == 1
    match = board.matches[0]
    assert (match.a, match.b, match.score_a) == ("W0", "W1", 1.0)
    assert board.winner == "W0"


class JudgeAllRaise:
    async def judge(self, *, model, system, user, max_tokens=1500):
        raise RuntimeError("simulated API error")


async def test_both_orders_raising_drops_the_match_and_logs_the_cause(caplog):
    with caplog.at_level("WARNING"):
        board = await run_tournament(
            JudgeAllRaise(), RunConfig(problem="p"), _arts(2), _manifest(), {}
        )
    assert board.matches == []
    assert "simulated API error" in caplog.text


class RecordingJudge(JudgeAWins):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def judge(self, *, model, system, user, max_tokens=1500):
        self.prompts.append(user)
        return await super().judge(model=model, system=system, user=user, max_tokens=max_tokens)


async def test_each_extra_round_asks_a_different_question():
    # No sampling parameters any more: an identical prompt would buy an
    # identical verdict, so each round must vary what it asks.
    judge = RecordingJudge()
    await run_tournament(judge, RunConfig(problem="p", elo_rounds=3), _arts(3), _manifest(), {})
    assert len(judge.prompts) == 3 * 3 * 2  # pairs x rounds x both orders
    assert len(set(judge.prompts)) == len(judge.prompts)
