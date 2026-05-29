"""ELO tournament engine (QA Layer 2).

Pure math (`expected_score`, `update_pair`, `EloEngine`, `pairings`) is fully
deterministic and unit-tested offline with no API calls. `run_tournament` wires
the pure engine to an async LLM judge.
"""

from __future__ import annotations

from .config import RunConfig
from .models import (
    ContextManifest,
    JudgeVerdict,
    Leaderboard,
    LeaderboardEntry,
    MatchResult,
    WorkerArtifact,
)

# --- Pure ELO math --------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability A beats B under the logistic ELO model."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_pair(
    rating_a: float, rating_b: float, score_a: float, k: float = 32.0
) -> tuple[float, float]:
    """Return updated (A, B) ratings after a match. `score_a` in {1, 0.5, 0}."""
    exp_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - exp_a))
    return new_a, new_b


def pairings(ids: list[str], rounds: int = 1) -> list[tuple[str, str]]:
    """Round-robin of every unique pair, repeated `rounds` times. Deterministic."""
    base: list[tuple[str, str]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            base.append((ids[i], ids[j]))
    return base * max(1, rounds)


class EloEngine:
    """Accumulates match outcomes into ratings + a ranked leaderboard."""

    def __init__(self, ids: list[str], start: float = 1000.0, k: float = 32.0) -> None:
        self.k = k
        self.ratings: dict[str, float] = {i: start for i in ids}
        self.wins: dict[str, int] = {i: 0 for i in ids}
        self.losses: dict[str, int] = {i: 0 for i in ids}
        self.draws: dict[str, int] = {i: 0 for i in ids}

    def record_match(self, a: str, b: str, score_a: float) -> None:
        ra, rb = self.ratings[a], self.ratings[b]
        self.ratings[a], self.ratings[b] = update_pair(ra, rb, score_a, self.k)
        if score_a > 0.5:
            self.wins[a] += 1
            self.losses[b] += 1
        elif score_a < 0.5:
            self.wins[b] += 1
            self.losses[a] += 1
        else:
            self.draws[a] += 1
            self.draws[b] += 1

    def leaderboard(
        self, objective: dict[str, float] | None = None, blend_elo: float = 0.6
    ) -> Leaderboard:
        objective = objective or {}
        ids = list(self.ratings)
        vals = list(self.ratings.values())
        lo, hi = min(vals), max(vals)
        span = hi - lo

        def norm(r: float) -> float:
            return 0.5 if span == 0 else (r - lo) / span

        entries: list[LeaderboardEntry] = []
        for i in ids:
            obj = objective.get(i, 0.0)
            final = blend_elo * norm(self.ratings[i]) + (1.0 - blend_elo) * obj
            entries.append(
                LeaderboardEntry(
                    worker_id=i,
                    elo=round(self.ratings[i], 1),
                    wins=self.wins[i],
                    losses=self.losses[i],
                    draws=self.draws[i],
                    objective_score=round(obj, 3),
                    final_score=round(final, 4),
                )
            )
        entries.sort(key=lambda e: (e.final_score, e.elo), reverse=True)
        winner = entries[0].worker_id if entries else None
        return Leaderboard(entries=entries, winner=winner)


def _score_from_verdict(verdict: JudgeVerdict) -> float:
    """Map a verdict to A's match score."""
    if verdict.winner == "A":
        return 1.0
    if verdict.winner == "B":
        return 0.0
    return 0.5


# --- Async judging loop ---------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an impartial senior engineer judging two candidate solutions to the "
    "same problem, head-to-head. Compare them on: correctness, code cleanliness "
    "and readability, performance/speed, and resource footprint (in that priority "
    "order). Be decisive. Return 'A', 'B', or 'tie', a one-sentence reason, and a "
    "0-10 score for each. Judge only what is in front of you; do not run code."
)


def _judge_system_blocks(manifest: ContextManifest) -> list[dict]:
    """Frozen judge instructions + problem + manifest, cached as a shared prefix
    so every pairwise call in the run reads from cache."""
    context = (
        f"PROBLEM:\n{manifest.problem}\n\n"
        f"LANGUAGE: {manifest.language}\n"
        f"CONSTRAINTS: {'; '.join(manifest.constraints) or 'none stated'}\n"
        f"NOTES: {manifest.notes or 'none'}"
    )
    return [
        {"type": "text", "text": _JUDGE_SYSTEM},
        {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
    ]


def _pair_user_prompt(a: WorkerArtifact, b: WorkerArtifact) -> str:
    return (
        f"### Solution A (id={a.worker_id}, strategy: {a.strategy})\n"
        f"```{a.language}\n{a.code}\n```\n\n"
        f"### Solution B (id={b.worker_id}, strategy: {b.strategy})\n"
        f"```{b.language}\n{b.code}\n```\n\n"
        "Which solution is better? Respond with the structured verdict."
    )


async def run_tournament(
    llm,
    cfg: RunConfig,
    artifacts: list[WorkerArtifact],
    manifest: ContextManifest,
    objective: dict[str, float] | None = None,
) -> Leaderboard:
    """Judge every pairing (concurrently, cache-warmed) then fold the outcomes
    into deterministic ELO ratings in a fixed order."""
    from .llm import fan_out  # local import to keep elo.py importable without the SDK

    ids = [a.worker_id for a in artifacts]
    by_id = {a.worker_id: a for a in artifacts}
    engine = EloEngine(ids, start=cfg.elo_start, k=cfg.k_factor)

    if len(ids) < 2:
        # Nothing to compare; single survivor wins by default.
        return engine.leaderboard(objective, cfg.blend_elo)

    pairs = pairings(ids, cfg.elo_rounds)
    system = _judge_system_blocks(manifest)

    async def judge_one(pair: tuple[str, str]) -> MatchResult | None:
        a_id, b_id = pair
        verdict = await llm.judge(
            model=cfg.judge_model,
            system=system,
            user=_pair_user_prompt(by_id[a_id], by_id[b_id]),
            max_tokens=cfg.judge_max_tokens,
        )
        if verdict is None:
            return None
        return MatchResult(
            a=a_id, b=b_id, score_a=_score_from_verdict(verdict), verdict=verdict
        )

    results = await fan_out([lambda p=p: judge_one(p) for p in pairs])

    matches: list[MatchResult] = []
    for mr in results:
        if mr is None:
            continue
        engine.record_match(mr.a, mr.b, mr.score_a)
        matches.append(mr)

    board = engine.leaderboard(objective, cfg.blend_elo)
    board.matches = matches
    return board
