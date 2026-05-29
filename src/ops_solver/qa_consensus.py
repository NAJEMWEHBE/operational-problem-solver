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
            proc = subprocess.run(
                ["ruff", "check", str(path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            ok = proc.returncode == 0
            issues = proc.stdout.count("\n") if not ok else 0
            score = 1.0 if ok else max(0.0, 1.0 - 0.05 * issues)
            return True, ok, score, (proc.stdout or proc.stderr)[:500]
        except (subprocess.SubprocessError, OSError) as exc:
            return False, True, 1.0, f"linter error: {exc}"
    return False, True, 1.0, "no linter available for this language"


def _run_tests(artifact: WorkerArtifact, test_cmd: str) -> tuple[bool, str]:
    """Run the user-supplied test command in the artifact's workspace.

    WARNING: this executes the generated code. Opt-in via --test-cmd only."""
    try:
        proc = subprocess.run(
            test_cmd,
            shell=True,
            cwd=artifact.workspace,
            capture_output=True,
            text=True,
            timeout=_TEST_TIMEOUT_S,
        )
        passed = proc.returncode == 0
        tail = (proc.stdout + proc.stderr)[-600:]
        return passed, tail
    except subprocess.TimeoutExpired:
        return False, f"test command timed out after {_TEST_TIMEOUT_S}s"
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"test command failed to run: {exc}"


def evaluate(artifact: WorkerArtifact, cfg: RunConfig) -> EvalResult:
    start = time.monotonic()
    syntax_ok = _syntax_check(artifact)

    lint_ran = lint_ok = False
    lint_score = 1.0
    lint_summary = ""
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
