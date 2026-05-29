"""Typed data models passed between the teams. Pydantic for validation and
for the judge's structured output."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ManifestDraft(BaseModel):
    """A single Intelligence worker's partial read of the environment.

    Used as the structured-output schema for each Crow. Numeric/length
    constraints are intentionally avoided (the structured-output API does not
    enforce them; the SDK validates client-side)."""

    dependencies: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    parameters: dict[str, str] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    notes: str = ""


class ContextManifest(BaseModel):
    """Unified context handed to every Execution worker."""

    problem: str
    language: str = "auto"
    dependencies: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    parameters: dict[str, str] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    notes: str = ""


class WorkerArtifact(BaseModel):
    """One Execution worker's standalone solution."""

    worker_id: str           # "Finch-01"
    strategy: str            # diversity hint applied
    language: str
    filename: str            # suggested file name for the solution
    code: str                # extracted solution source
    explanation: str = ""    # worker's own notes
    workspace: str = ""      # path to this worker's isolated dir


class EvalResult(BaseModel):
    """QA Layer 1 objective check for one artifact."""

    worker_id: str
    stable: bool
    syntax_ok: bool
    syntax_checked: bool = True   # False => language has no parser here; syntax_ok is assumed
    lint_ran: bool = False
    lint_ok: bool = True
    lint_score: float = 1.0          # 0..1
    tests_ran: bool = False
    tests_passed: bool = False
    test_summary: str = ""
    duration_s: float = 0.0
    size_bytes: int = 0
    notes: str = ""


class JudgeVerdict(BaseModel):
    """Structured output from a single pairwise judge call."""

    winner: Literal["A", "B", "tie"]
    reason: str
    a_score: int = 5                 # 0..10 (clamped in code, not by schema)
    b_score: int = 5


class MatchResult(BaseModel):
    a: str                           # worker_id of solution A
    b: str                           # worker_id of solution B
    score_a: float                   # 1.0 win / 0.5 tie / 0.0 loss (for A)
    verdict: JudgeVerdict


class LeaderboardEntry(BaseModel):
    worker_id: str
    elo: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    objective_score: float = 0.0
    final_score: float = 0.0


class Leaderboard(BaseModel):
    entries: list[LeaderboardEntry] = Field(default_factory=list)
    winner: Optional[str] = None
    matches: list[MatchResult] = Field(default_factory=list)


class RunReport(BaseModel):
    run_id: str
    problem: str
    manifest: ContextManifest
    artifacts: list[WorkerArtifact]
    evals: list[EvalResult]
    leaderboard: Leaderboard
    winner_id: Optional[str] = None
    winner_path: Optional[str] = None
    attempts: int = 1
    consensus_ratio: float = 0.0
    consensus_met: bool = False
    cost_usd: float = 0.0
    tokens: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
