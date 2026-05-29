"""`ops-solve "solve: <problem>"` — command-line entry point."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import CHEAP_WORKER_MODEL, RunConfig
from .models import RunReport
from .orchestrator import solve

app = typer.Typer(add_completion=False, help="Multi-team consensus + ELO problem-solving engine.")
console = Console()


_DOTENV_ALLOWED = {"ANTHROPIC_API_KEY", "ANTHROPIC_LOG", "ANTHROPIC_BASE_URL"}


def _load_dotenv(cwd: Path) -> None:
    """Minimal .env loader: only an allow-list of keys we use, so a stray .env in
    the working directory can't inject arbitrary environment variables."""
    env = cwd / ".env"
    if not env.exists() or os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        # Override only when the var is absent OR present-but-empty; a real
        # existing value already short-circuited at the top guard, so we never
        # clobber a genuine env key — but an empty "" no longer shadows .env.
        if key in _DOTENV_ALLOWED and not os.environ.get(key):
            os.environ[key] = val.strip().strip('"').strip("'")


def _render(report: RunReport) -> None:
    m = report.manifest
    console.print(
        Panel(
            f"[bold]{m.problem}[/]\n\n"
            f"language: {m.language}\n"
            f"dependencies: {', '.join(m.dependencies) or 'none'}\n"
            f"constraints: {'; '.join(m.constraints) or 'none'}",
            title="Context Manifest (Intelligence team)",
            border_style="cyan",
        )
    )

    ev = Table(title="QA Layer 1 — objective checks", border_style="yellow")
    for col in ("Worker", "Strategy", "Syntax", "Lint", "Tests", "Stable", "Bytes"):
        ev.add_column(col)
    strat = {a.worker_id: a.strategy for a in report.artifacts}
    for e in report.evals:
        ev.add_row(
            e.worker_id,
            (strat.get(e.worker_id, "")[:28]),
            ("[green]ok[/]" if e.syntax_ok else "[red]fail[/]") if e.syntax_checked else "[dim]n/a[/]",
            ("[green]ok[/]" if e.lint_ok else "[red]issues[/]") if e.lint_ran else "-",
            ("[green]pass[/]" if e.tests_passed else "[red]fail[/]") if e.tests_ran else "-",
            "[green]yes[/]" if e.stable else "[red]no[/]",
            str(e.size_bytes),
        )
    console.print(ev)
    verified = any(e.tests_ran for e in report.evals)
    console.print(
        f"Consensus: {report.consensus_ratio:.0%} stable "
        f"({'met' if report.consensus_met else 'NOT met'}); attempts={report.attempts}"
        + ("" if verified else "  [dim](syntax-only - pass --test-cmd to verify behaviour)[/]")
    )
    for w in report.warnings:
        console.print(f"[yellow]! {w}[/]")

    lb = Table(title="QA Layer 2 — ELO tournament leaderboard", border_style="magenta")
    for col in ("#", "Worker", "ELO", "W-L-D", "Objective", "Final"):
        lb.add_column(col)
    for rank, e in enumerate(report.leaderboard.entries, 1):
        mark = " [bold green](WINNER)[/]" if e.worker_id == report.winner_id else ""
        lb.add_row(
            str(rank),
            e.worker_id + mark,
            f"{e.elo:.0f}",
            f"{e.wins}-{e.losses}-{e.draws}",
            f"{e.objective_score:.2f}",
            f"{e.final_score:.3f}",
        )
    console.print(lb)

    console.print(
        Panel(
            f"winner: [bold green]{report.winner_id or 'none'}[/]\n"
            f"shipped to: {report.winner_path or '(none)'}\n"
            f"calls: {report.tokens.get('calls', 0)}   "
            f"estimated cost: [bold]${report.cost_usd:.4f}[/]",
            title="Result",
            border_style="green",
        )
    )


@app.command()
def main(
    problem: str = typer.Argument(..., help='The goal, e.g. "solve: write a prime sieve"'),
    workers: int = typer.Option(4, "--workers", "-w", help="Execution workers (Finch)."),
    intel: int = typer.Option(2, "--intel", "-i", help="Intelligence workers (Crow)."),
    rounds: int = typer.Option(1, "--rounds", "-r", help="ELO tournament round-robin passes."),
    reloops: int = typer.Option(1, "--reloops", help="Max consensus re-loops."),
    test_cmd: Optional[str] = typer.Option(
        None, "--test-cmd", help="Shell command to test each solution (EXECUTES generated code)."
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Write the winning solution file here."),
    force: bool = typer.Option(False, "--force", help="Allow --out to overwrite an existing file."),
    cheap: bool = typer.Option(False, "--cheap", help="Use Haiku workers (cheaper, less diverse)."),
    lang: str = typer.Option("auto", "--lang", help="Force a language instead of auto-detect."),
    debug: bool = typer.Option(False, "--debug", help="Re-raise full tracebacks instead of a clean message."),
) -> None:
    """Solve a problem with parallel workers, a consensus gate, and an ELO tournament."""
    cwd = Path.cwd()
    _load_dotenv(cwd)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        hint = " (found a .env, but it did not define ANTHROPIC_API_KEY)" if (cwd / ".env").exists() else ""
        console.print(
            f"[red]ANTHROPIC_API_KEY is not set.[/]{hint} Copy .env.example to .env and add your key, "
            "or set ANTHROPIC_API_KEY in your environment. (This is a paid, metered key.)"
        )
        raise typer.Exit(code=2)

    goal = problem
    if goal.lower().startswith("solve:"):
        goal = goal.split(":", 1)[1].strip()

    cfg = RunConfig(
        problem=goal,
        cwd=cwd,
        language=lang,
        intel_workers=max(1, intel),
        exec_workers=max(1, workers),
        elo_rounds=max(1, rounds),
        max_reloops=max(0, reloops),
        test_cmd=test_cmd,
        out_path=out,
        force=force,
    )
    if cheap:
        cfg.worker_model = CHEAP_WORKER_MODEL

    if test_cmd:
        console.print(
            "[yellow]Note:[/] --test-cmd will EXECUTE the generated code for each worker."
        )

    console.print(
        f"[dim]Engine: {cfg.intel_workers} intel + {cfg.exec_workers} exec "
        f"(workers={cfg.worker_model}, judge={cfg.judge_model})[/]"
    )
    try:
        report = asyncio.run(solve(cfg))
    except Exception as exc:  # surface a clean message instead of a traceback wall
        if debug:
            raise
        console.print(f"[red]Run failed:[/] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc

    _render(report)


if __name__ == "__main__":
    app()
