"""ELO math + tournament aggregation. Pure, offline, no API."""

from ops_solver.elo import EloEngine, expected_score, pairings, update_pair


def test_expected_score_symmetry():
    assert expected_score(1000, 1000) == 0.5
    assert expected_score(1200, 1000) > 0.5
    assert expected_score(1000, 1200) < 0.5
    # Complementary probabilities sum to 1.
    assert abs(expected_score(1300, 900) + expected_score(900, 1300) - 1.0) < 1e-9


def test_update_pair_zero_sum_and_direction():
    a, b = update_pair(1000, 1000, score_a=1.0, k=32.0)
    # Equal ratings, A wins -> A up by K/2, B down by K/2.
    assert round(a, 6) == 1016.0
    assert round(b, 6) == 984.0
    # Total rating is conserved.
    assert round((a + b), 6) == 2000.0


def test_update_pair_draw_is_noop_when_equal():
    a, b = update_pair(1000, 1000, score_a=0.5, k=32.0)
    assert round(a, 6) == 1000.0
    assert round(b, 6) == 1000.0


def test_pairings_count():
    # 4 unique players -> 6 unique pairs; 2 rounds -> 12 matches.
    assert len(pairings(["a", "b", "c", "d"], rounds=1)) == 6
    assert len(pairings(["a", "b", "c", "d"], rounds=2)) == 12
    assert pairings(["solo"], rounds=3) == []


def test_engine_ranks_consistent_winner_first():
    eng = EloEngine(["A", "B", "C"], start=1000.0, k=32.0)
    # A beats everyone, B beats C.
    eng.record_match("A", "B", 1.0)
    eng.record_match("A", "C", 1.0)
    eng.record_match("B", "C", 1.0)
    board = eng.leaderboard()
    assert board.winner == "A"
    assert [e.worker_id for e in board.entries] == ["A", "B", "C"]
    assert board.entries[0].wins == 2
    assert board.entries[-1].losses == 2


def test_leaderboard_blend_objective_can_flip_winner():
    eng = EloEngine(["A", "B"], start=1000.0, k=32.0)
    eng.record_match("A", "B", 1.0)  # A leads on ELO
    # But B has a far better objective score; with objective weighted heavily,
    # B should overtake.
    board = eng.leaderboard(objective={"A": 0.0, "B": 1.0}, blend_elo=0.2)
    assert board.winner == "B"
