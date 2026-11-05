"""CLI rendering: model- and user-written text must print literally."""

import io

from rich.console import Console

from ops_solver import cli
from ops_solver.models import (
    ContextManifest,
    EvalResult,
    Leaderboard,
    LeaderboardEntry,
    RunReport,
    WorkerArtifact,
)


def _report(problem: str, constraint: str, strategy: str) -> RunReport:
    return RunReport(
        run_id="run-test",
        problem=problem,
        manifest=ContextManifest(problem=problem, constraints=[constraint], dependencies=["[bold]x"]),
        artifacts=[
            WorkerArtifact(worker_id="Finch-01", strategy=strategy, language="python",
                           filename="solution.py", code="x = 1")
        ],
        evals=[EvalResult(worker_id="Finch-01", stable=True, syntax_ok=True)],
        leaderboard=Leaderboard(
            entries=[LeaderboardEntry(worker_id="Finch-01", elo=1000.0)], winner="Finch-01"
        ),
        winner_id="Finch-01",
        winner_path="runs/[x]/WINNER",
        warnings=["odd [/] warning"],
    )


def _render_text(report: RunReport, monkeypatch) -> str:
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=300, color_system=None))
    cli._render(report)
    return buf.getvalue()


def test_render_prints_markup_like_text_literally(monkeypatch):
    out = _render_text(
        _report("parse close tags [/] safely", "[red]must[/red] stay", "[i]odd"), monkeypatch
    )
    assert "parse close tags [/] safely" in out
    assert "[red]must[/red] stay" in out
    assert "[bold]x" in out
    assert "[i]odd" in out
    assert "runs/[x]/WINNER" in out
    assert "odd [/] warning" in out
