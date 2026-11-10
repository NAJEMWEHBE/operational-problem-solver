"""QA Layer 1: objective evaluation + the consensus stability gate.

Each artifact is checked for syntax validity, linted (if a linter is available),
and optionally run against a user-supplied test command. A run is "stable" if it
parses and (when tests ran) passes them.

SECURITY: syntax checks and linting do NOT execute the generated code. Only a
`test_cmd` executes it — that is opt-in (the user must pass --test-cmd), and it
runs model-written code, so use it only on solutions you are willing to run.

The working directory ops-solve runs in may be a project you don't control, so
no helper started here takes anything from it: ruff (and, on Windows, taskkill)
is launched by absolute path (Windows searches the current directory before
PATH), ruff writes no cache, and the test command runs in a private folder
outside it.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from .config import RunConfig
from .models import EvalResult, WorkerArtifact

logger = logging.getLogger("ops_solver.qa_consensus")

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


def _trusted_which(name: str) -> str | None:
    """Absolute path of `name` from the absolute PATH entries, never from the
    current directory. On Windows both shutil.which and CreateProcess look in the
    current directory (the project ops-solve runs in) before PATH, so a bare name
    in argv could start a same-named program from there instead."""
    cwd = os.path.normcase(os.path.realpath(os.getcwd()))
    exts = (".exe",) if os.name == "nt" else ("",)
    for d in os.environ.get("PATH", "").split(os.pathsep):
        d = d.strip('"')
        if not d or not os.path.isabs(d) or os.path.normcase(os.path.realpath(d)) == cwd:
            continue
        for ext in exts:
            cand = os.path.join(d, name + ext)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


def _system_tool(name: str) -> str:
    """Absolute path of a Windows system tool (System32); never searched for."""
    import ctypes

    buf = ctypes.create_unicode_buffer(260)
    n = ctypes.windll.kernel32.GetSystemDirectoryW(buf, len(buf))
    if 0 < n < len(buf):
        sysdir = buf.value
    else:
        sysdir = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32")
    return os.path.join(sysdir, name)


def _run_linter(artifact: WorkerArtifact) -> tuple[bool, bool, float, str]:
    """Return (ran, ok, score, summary). Currently wired for Python + ruff."""
    lang = artifact.language.lower()
    path = Path(artifact.workspace) / artifact.filename
    ruff = _trusted_which("ruff") if lang in ("python", "py") else None
    if ruff and path.exists():
        try:
            # JSON output: one entry per issue (the default text format prints ~10
            # lines per issue). --isolated: score every worker on ruff's defaults,
            # not on whatever ruff config sits above the runs dir. --no-cache and
            # cwd=: otherwise ruff creates .ruff_cache in the working directory,
            # following whatever entry of that name is already there.
            proc = subprocess.run(
                [ruff, "check", "--output-format", "json", "--isolated", "--no-cache",
                 os.path.abspath(path)],
                capture_output=True,
                check=False,
                text=True,
                timeout=60,
                cwd=artifact.workspace or None,
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
    if lang in ("python", "py") and not ruff:
        return False, True, 1.0, "ruff not found on PATH outside the working directory"
    return False, True, 1.0, "no linter available for this language"


def _private_test_base() -> Path:
    """Per-user folder for test runs: outside any working directory, and not
    directly in a shared temp folder that another account can write to."""
    root = os.environ.get("LOCALAPPDATA" if os.name == "nt" else "XDG_CACHE_HOME")
    base = Path(root) if root and os.path.isabs(root) else Path.home() / ".cache"
    base = base / "ops-solve" / "test-runs"
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        # A folder that was already there must be this user's own, and closed to
        # other accounts: test runners read config files from parent folders.
        st = base.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
            raise PermissionError(f"not a folder owned by this user: {base}")
        if st.st_mode & 0o077:
            base.chmod(0o700)
    return base


def _run_tests(artifact: WorkerArtifact, test_cmd: str) -> tuple[bool, str]:
    """Run the user-supplied test command against a copy of the solution file.

    The command runs in a fresh private folder outside the working directory,
    holding only the solution file (e.g. `solution.py`, which pytest does not
    collect by default), so `{file}` in the command is replaced with that file
    name: `pytest -q {file}`. Test runners read config and hook files from parent
    folders (pytest.ini + conftest.py, package.json, Cargo.toml), and the working
    directory may be a project you don't control, so the worker folder under
    runs/ is not where the tests run.

    WARNING: this executes the generated code. Opt-in via --test-cmd only."""
    cmd = test_cmd.replace("{file}", artifact.filename)
    try:
        tmp = tempfile.TemporaryDirectory(
            prefix="run-", dir=_private_test_base(), ignore_cleanup_errors=True
        )
    except OSError as exc:
        return False, f"test folder could not be created: {exc}"
    with tmp as test_dir:
        try:
            shutil.copyfile(
                Path(artifact.workspace) / artifact.filename, Path(test_dir) / artifact.filename
            )
        except OSError as exc:
            result = False, f"solution file could not be copied for testing: {exc}"
        else:
            result = _run_test_cmd(cmd, test_dir)
    if os.path.lexists(test_dir):
        # cleanup errors are ignored above (a process may still hold a file);
        # try once more, then say what was left behind
        shutil.rmtree(test_dir, ignore_errors=True)
        if os.path.lexists(test_dir):
            logger.warning("test folder could not be removed: %s", test_dir)
    return result


def _spawn(cmd: str, cwd: str) -> subprocess.Popen:
    """Start the shell command `cmd` in `cwd`, in its own process group / session,
    so a timeout can kill the whole tree: killing only the shell leaves a child
    (a server, a `while True:`) holding the output pipes, and on Windows
    communicate() would then wait for it forever."""
    group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    return subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        **group,
    )


def _run_test_cmd(cmd: str, cwd: str) -> tuple[bool, str]:
    try:
        proc = _spawn(cmd, cwd)
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"test command failed to run: {exc}"
    try:
        out, err = proc.communicate(timeout=_TEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        killed = "process tree killed" if _kill_tree(proc) else "process killed"
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # a process escaped the tree; don't wait on its pipes
        return False, f"test command timed out after {_TEST_TIMEOUT_S}s ({killed})"
    return proc.returncode == 0, ((out or "") + (err or ""))[-600:]


def _kill_tree(proc: subprocess.Popen) -> bool:
    """Kill `proc` and every process it started. Returns True only when the
    whole tree was killed; `proc` itself is killed either way."""
    tree_killed = False
    if os.name == "nt":
        try:
            done = subprocess.run(
                [_system_tool("taskkill.exe"), "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
                timeout=30,
            )
            tree_killed = done.returncode == 0
        except (subprocess.SubprocessError, OSError):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            tree_killed = True
        except (ProcessLookupError, PermissionError):
            pass
    if proc.poll() is None:
        proc.kill()
    return tree_killed


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


async def evaluate_all(artifacts: list[WorkerArtifact], cfg: RunConfig) -> list[EvalResult]:
    """`evaluate` every artifact concurrently in worker threads, so linting and
    test runs (up to the test timeout each) neither block the event loop nor
    run one after another. Results keep the input order."""
    return list(await asyncio.gather(*(asyncio.to_thread(evaluate, a, cfg) for a in artifacts)))


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
