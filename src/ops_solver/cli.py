"""`ops-solve "solve: <problem>"` — command-line entry point."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
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
            os.environ[key] = _dotenv_value(val)


def _dotenv_value(raw: str) -> str:
    """A .env value: quoted values are taken verbatim; unquoted ones lose an
    inline ` # comment` (a '#' with no space before it, as in a URL, stays)."""
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        if end != -1:
            return raw[1:end]
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip().strip('"').strip("'")


def _has_credentials() -> bool:
    """True if the SDK will find credentials: ANTHROPIC_API_KEY, or what its own
    chain resolves (ANTHROPIC_AUTH_TOKEN, a profile from `ant auth login`,
    workload identity). Never reads or prints a credential value here."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    try:
        from anthropic.lib.credentials import default_credentials

        return default_credentials() is not None
    except Exception:  # noqa: BLE001 - a broken or missing profile means "no usable credentials"
        return False


def _render(report: RunReport) -> None:
    # Every dynamic string (problem text, model-written manifest, paths) goes
    # through escape(): rich would otherwise parse "[red]" as a style or raise
    # MarkupError on "[/]" after the paid run.
    m = report.manifest
    console.print(
        Panel(
            f"[bold]{escape(m.problem)}[/]\n\n"
            f"language: {escape(m.language)}\n"
            f"dependencies: {escape(', '.join(m.dependencies) or 'none')}\n"
            f"constraints: {escape('; '.join(m.constraints) or 'none')}",
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
            escape(e.worker_id),
            escape(strat.get(e.worker_id, "")[:28]),
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
        console.print(f"[yellow]! {escape(w)}[/]")

    lb = Table(title="QA Layer 2 — ELO tournament leaderboard", border_style="magenta")
    for col in ("#", "Worker", "ELO", "W-L-D", "Objective", "Final"):
        lb.add_column(col)
    for rank, e in enumerate(report.leaderboard.entries, 1):
        mark = " [bold green](WINNER)[/]" if e.worker_id == report.winner_id else ""
        lb.add_row(
            str(rank),
            escape(e.worker_id) + mark,
            f"{e.elo:.0f}",
            f"{e.wins}-{e.losses}-{e.draws}",
            f"{e.objective_score:.2f}",
            f"{e.final_score:.3f}",
        )
    console.print(lb)

    console.print(
        Panel(
            f"winner: [bold green]{escape(report.winner_id or 'none')}[/]\n"
            f"shipped to: {escape(report.winner_path or '(none)')}\n"
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
    concurrency: int = typer.Option(
        8, "--concurrency", "-c", help="Max API calls in flight at once (lower it on 429s)."
    ),
    test_cmd: str | None = typer.Option(
        None,
        "--test-cmd",
        help="Shell command run in each worker's folder to test its solution; {file} is "
        'replaced with the solution file name, e.g. "pytest -q {file}" (EXECUTES generated code).',
    ),
    out: Path | None = typer.Option(None, "--out", help="Write the winning solution file here."),
    force: bool = typer.Option(False, "--force", help="Allow --out to overwrite an existing file."),
    cheap: bool = typer.Option(False, "--cheap", help="Use Haiku workers (cheaper, less diverse)."),
    lang: str = typer.Option("auto", "--lang", help="Force a language instead of auto-detect."),
    debug: bool = typer.Option(False, "--debug", help="Re-raise full tracebacks instead of a clean message."),
) -> None:
    """Solve a problem with parallel workers, a consensus gate, and an ELO tournament."""
    cwd = Path.cwd()
    _load_dotenv(cwd)
    if not _has_credentials():
        hint = " (found a .env, but it did not define ANTHROPIC_API_KEY)" if (cwd / ".env").exists() else ""
        console.print(
            f"[red]ANTHROPIC_API_KEY is not set[/] and no other Anthropic credentials were found."
            f"{hint} Copy .env.example to .env and add your key, set ANTHROPIC_API_KEY in your "
            "environment, or sign in with `ant auth login`. (This is a paid, metered key.)"
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
        max_concurrency=max(1, concurrency),
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
        console.print(f"[red]Run failed:[/] {type(exc).__name__}: {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    _render(report)


if __name__ == "__main__":
    app()
