"""CLI rendering: model- and user-written text must print literally."""

import io
import os

import pytest
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


# --- credentials check before a paid run ------------------------------------


def _no_credentials(monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    empty = tmp_path / "anthropic-config"
    empty.mkdir()
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(empty))  # no on-disk profile
    monkeypatch.chdir(tmp_path)  # no .env


def _invoke(monkeypatch):
    from typer.testing import CliRunner

    async def fake_solve(cfg):
        raise RuntimeError("stopped before any API call")

    monkeypatch.setattr(cli, "solve", fake_solve)
    return CliRunner().invoke(cli.app, ["solve: x"])


def test_refuses_to_run_without_any_credentials(monkeypatch, tmp_path):
    _no_credentials(monkeypatch, tmp_path)
    result = _invoke(monkeypatch)
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output


def test_runs_with_sdk_discoverable_credentials_instead_of_an_api_key(monkeypatch, tmp_path):
    _no_credentials(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-dummy-token")
    result = _invoke(monkeypatch)
    assert result.exit_code == 1
    assert "stopped before any API call" in result.output
    assert "test-dummy-token" not in result.output


def test_expected_call_count_matches_the_default_run():
    from ops_solver.config import RunConfig

    # 2 intel + 4 workers + C(4,2)=6 pairs x 2 orders; a re-loop adds 4 workers
    assert RunConfig(problem="p").expected_calls() == (18, 22)
    assert RunConfig(problem="p", judge_both_orders=False, max_reloops=0).expected_calls() == (12, 12)
    assert RunConfig(problem="p", exec_workers=3, elo_rounds=2).expected_calls() == (2 + 3 + 12, 20)


def test_cli_prints_the_expected_call_count_before_starting(monkeypatch, tmp_path):
    _no_credentials(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-dummy-token")
    result = _invoke(monkeypatch)
    assert "18 API calls (up to 22 with re-loops)" in " ".join(result.output.split())


# --- arguments reach main() exactly as typed (no Windows argv expansion) -----


def test_console_script_uses_the_literal_argv_entry():
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["ops-solve"] == "ops_solver.cli:run"


class _WindowsOs:
    """typer.core's view of `os`, reporting Windows; everything else is real."""

    name = "nt"

    def __getattr__(self, attr):
        return getattr(os, attr)


def _argv_seen_by_main(monkeypatch, home, entry):
    import sys

    import typer.core

    home.mkdir()
    _no_credentials(monkeypatch, home)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-dummy-token")
    monkeypatch.setenv("OPS_ARGV_PROBE", "EXPANDED")
    monkeypatch.setattr(typer.core, "os", _WindowsOs(), raising=False)
    seen = {}

    async def fake_solve(cfg):
        seen.update(problem=cfg.problem, test_cmd=cfg.test_cmd, lang=cfg.language)
        raise RuntimeError("stopped before any API call")

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ops-solve", "solve: print $OPS_ARGV_PROBE and %OPS_ARGV_PROBE%",
         "--test-cmd", "echo ${OPS_ARGV_PROBE} {file}", "--lang", "$OPS_ARGV_PROBE"],
    )
    try:
        entry()
    except SystemExit:
        pass
    return seen


def test_run_passes_arguments_through_literally(monkeypatch, tmp_path):
    # control: Typer's own default does expand them, so the check below is real
    control = _argv_seen_by_main(monkeypatch, tmp_path / "control", cli.app)
    if "EXPANDED" not in control.get("problem", ""):
        pytest.skip("this Typer version doesn't expand argv in typer.core")
    seen = _argv_seen_by_main(monkeypatch, tmp_path / "run", cli.run)
    assert seen == {
        "problem": "print $OPS_ARGV_PROBE and %OPS_ARGV_PROBE%",
        "test_cmd": "echo ${OPS_ARGV_PROBE} {file}",
        "lang": "$OPS_ARGV_PROBE",
    }


def test_run_turns_off_typer_argv_expansion(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "app", lambda *a, **kw: calls.append((a, kw)))
    cli.run()
    assert calls == [((), {"windows_expand_args": False})]
