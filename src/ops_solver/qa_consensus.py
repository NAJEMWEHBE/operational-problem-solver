"""QA Layer 1: objective evaluation + the consensus stability gate.

Each artifact is checked for syntax validity, linted (if a linter is available),
and optionally run against a user-supplied test command. A run is "stable" if it
parses and (when tests ran) passes them.

SECURITY: syntax checks and linting do NOT execute the generated code. Only a
`test_cmd` executes it — that is opt-in (the user must pass --test-cmd), and it
runs model-written code, so use it only on solutions you are willing to run.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .config import RunConfig
from .models import EvalResult, WorkerArtifact

_TEST_TIMEOUT_S = 120


def _python_syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _syntax_check(artifact: WorkerArtifact) -> bool:
    lang = artifact.language.lower()
    if lang in ("python", "py"):
        return _python_syntax_ok(artifact.code)
    # For other languages we cannot parse safely without a toolchain; treat as
    # unknown-but-not-failing so they still compete (the judge weighs quality).
    return True


def _run_linter(artifact: WorkerArtifact) -> tuple[bool, bool, float, str]:
    """Return (ran, ok, score, summary). Currently wired for Python + ruff."""
    import shutil

    lang = artifact.language.lower()
    path = Path(artifact.workspace) / artifact.filename
    if lang in ("python", "py") and shutil.which("ruff") and path.exists():
        try:
            # JSON output: one entry per issue (the default text format prints ~10
            # lines per issue). --isolated: score every worker on ruff's defaults,
            # not on whatever ruff config sits above the runs dir.
            proc = subprocess.run(
                ["ruff", "check", "--output-format", "json", "--isolated", str(path)],
                capture_output=True,
                check=False,
                text=True,
                timeout=60,
            )
            ok = proc.returncode == 0
            try:
                found = json.loads(proc.stdout or "[]")
            except json.JSONDecodeError:
                return True, False, 0.5, f"linter output unreadable: {(proc.stderr or proc.stdout)[:400]}"
            issues = len(found)
            score = 1.0 if ok else max(0.0, 1.0 - 0.05 * issues)
            summary = "; ".join(
                f"{d.get('code')} line {d.get('location', {}).get('row')}: {d.get('message')}"
                for d in found
            )
            return True, ok, score, (summary or proc.stderr)[:500]
        except (subprocess.SubprocessError, OSError) as exc:
            # linter present but crashed: don't award full credit (would beat a
            # file that merely had lint warnings); mark it ran-but-failed.
            return True, False, 0.5, f"linter error: {exc}"
    return False, True, 1.0, "no linter available for this language"


def _run_tests(artifact: WorkerArtifact, test_cmd: str) -> tuple[bool, str]:
    """Run the user-supplied test command in the artifact's workspace.

    The workspace holds only the solution file (e.g. `solution.py`, which pytest
    does not collect by default), so `{file}` in the command is replaced with
    that file name: `pytest -q {file}`.

    WARNING: this executes the generated code. Opt-in via --test-cmd only."""
    cmd = test_cmd.replace("{file}", artifact.filename)
    # Own process group / session, so a timeout can kill the whole tree: killing
    # only the shell leaves a child (a server, a `while True:`) holding the output
    # pipes, and on Windows communicate() would then wait for it forever.
    group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=artifact.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            **group,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"test command failed to run: {exc}"
    try:
        out, err = proc.communicate(timeout=_TEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # a process escaped the tree; don't wait on its pipes
        return False, f"test command timed out after {_TEST_TIMEOUT_S}s (process tree killed)"
    return proc.returncode == 0, ((out or "") + (err or ""))[-600:]


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` and every process it started."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
            timeout=30,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if proc.poll() is None:
        proc.kill()


def evaluate(artifact: WorkerArtifact, cfg: RunConfig) -> EvalResult:
    start = time.monotonic()
    syntax_ok = _syntax_check(artifact)
    syntax_checked = artifact.language.lower() in ("python", "py")

    lint_ran, lint_ok, lint_score, lint_summary = False, True, 1.0, ""
    if cfg.lint:
        lint_ran, lint_ok, lint_score, lint_summary = _run_linter(artifact)

    tests_ran = tests_passed = False
    test_summary = ""
    if cfg.test_cmd and syntax_ok:
        tests_ran = True
        tests_passed, test_summary = _run_tests(artifact, cfg.test_cmd)

    stable = syntax_ok and (tests_passed if tests_ran else True)

    return EvalResult(
        worker_id=artifact.worker_id,
        stable=stable,
        syntax_ok=syntax_ok,
        syntax_checked=syntax_checked,
        lint_ran=lint_ran,
        lint_ok=lint_ok,
        lint_score=round(lint_score, 3),
        tests_ran=tests_ran,
        tests_passed=tests_passed,
        test_summary=test_summary or lint_summary,
        duration_s=round(time.monotonic() - start, 3),
        size_bytes=len(artifact.code.encode("utf-8")),
        notes=lint_summary if lint_ran and not lint_ok else "",
    )


def consensus(evals: list[EvalResult], threshold: float) -> tuple[list[str], float, bool]:
    """Return (stable_worker_ids, stable_ratio, meets_threshold)."""
    if not evals:
        return [], 0.0, False
    stable = [e.worker_id for e in evals if e.stable]
    ratio = len(stable) / len(evals)
    return stable, ratio, ratio >= threshold


def objective_score(ev: EvalResult) -> float:
    """Blend the objective signals into a 0..1 score for the leaderboard."""
    tests = 1.0 if (ev.tests_ran and ev.tests_passed) else (0.5 if not ev.tests_ran else 0.0)
    syntax = 1.0 if ev.syntax_ok else 0.0
    return round(0.5 * tests + 0.3 * ev.lint_score + 0.2 * syntax, 4)
