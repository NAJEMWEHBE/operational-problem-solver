"""QA Layer 1 consensus gate + objective scoring. Offline; the --test-cmd tests
run tiny local Python commands."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ops_solver import qa_consensus
from ops_solver.config import RunConfig
from ops_solver.models import EvalResult, WorkerArtifact
from ops_solver.qa_consensus import consensus, evaluate, objective_score


@pytest.fixture(autouse=True)
def _private_cache(tmp_path_factory, monkeypatch):
    """Test runs go to a per-user cache folder; keep it inside the test's tmp."""
    cache = tmp_path_factory.mktemp("user-cache")
    monkeypatch.setenv("LOCALAPPDATA", str(cache))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))


def _cfg(**kw):
    return RunConfig(problem="x", lint=False, **kw)


def test_consensus_empty():
    assert consensus([], 0.5) == ([], 0.0, False)


def test_consensus_threshold_met_and_missed():
    def ev(wid, stable):
        return EvalResult(worker_id=wid, stable=stable, syntax_ok=stable)

    # 1 of 2 stable -> 0.5, meets 0.5 threshold.
    ids, ratio, met = consensus([ev("a", True), ev("b", False)], 0.5)
    assert ids == ["a"] and ratio == 0.5 and met is True

    # 1 of 3 stable -> 0.33, misses.
    _, ratio2, met2 = consensus([ev("a", True), ev("b", False), ev("c", False)], 0.5)
    assert round(ratio2, 2) == 0.33 and met2 is False


def test_evaluate_valid_python_is_stable():
    art = WorkerArtifact(
        worker_id="Finch-01",
        strategy="simple",
        language="python",
        filename="solution.py",
        code="def add(a, b):\n    return a + b\n",
        workspace="",
    )
    res = evaluate(art, _cfg())
    assert res.syntax_ok is True
    assert res.stable is True
    assert res.tests_ran is False


def test_evaluate_broken_python_is_unstable():
    art = WorkerArtifact(
        worker_id="Finch-02",
        strategy="simple",
        language="python",
        filename="solution.py",
        code="def add(a, b)\n    return a + b\n",  # missing colon
        workspace="",
    )
    res = evaluate(art, _cfg())
    assert res.syntax_ok is False
    assert res.stable is False


def test_objective_score_rewards_passing_tests():
    passing = EvalResult(
        worker_id="a", stable=True, syntax_ok=True, tests_ran=True, tests_passed=True, lint_score=1.0
    )
    not_run = EvalResult(worker_id="b", stable=True, syntax_ok=True, lint_score=1.0)
    failing = EvalResult(
        worker_id="c", stable=False, syntax_ok=True, tests_ran=True, tests_passed=False, lint_score=1.0
    )
    assert objective_score(passing) > objective_score(not_run) > objective_score(failing)


# --- --test-cmd runs in the worker folder (executes code; local, tiny) -------

PY = f'"{sys.executable}"'


def _artifact(ws, code: str) -> WorkerArtifact:
    (ws / "solution.py").write_text(code, encoding="utf-8")
    return WorkerArtifact(
        worker_id="Finch-01", strategy="s", language="python",
        filename="solution.py", code=code, workspace=str(ws),
    )


def test_test_cmd_file_placeholder_points_pytest_at_the_solution(tmp_path):
    code = "def add(a, b):\n    return a + b\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    art = _artifact(tmp_path, code)
    res = evaluate(art, _cfg(test_cmd=f"{PY} -m pytest -q -p no:cacheprovider {{file}}"))
    assert res.tests_ran is True
    assert res.tests_passed is True, res.test_summary


def test_test_cmd_timeout_kills_the_whole_process_tree(tmp_path, monkeypatch):
    # The shell starts python, which starts a grandchild that holds the output
    # pipes for 20 s. A timeout must not wait for it.
    monkeypatch.setattr(qa_consensus, "_TEST_TIMEOUT_S", 2)
    grandchild = (
        "import subprocess, sys; "
        "subprocess.run([sys.executable, '-c', 'import time; time.sleep(20)'])"
    )
    art = _artifact(tmp_path, "x = 1\n")
    start = time.monotonic()
    res = evaluate(art, _cfg(test_cmd=f'{PY} -c "{grandchild}"'))
    elapsed = time.monotonic() - start
    assert res.tests_passed is False
    assert "timed out" in res.test_summary
    assert elapsed < 10, f"timeout took {elapsed:.1f}s: the child tree was not killed"


# --- lint score counts real issues (needs ruff on PATH, as in CI) -----------


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")
@pytest.mark.parametrize(("issues", "score"), [(1, 0.95), (2, 0.9), (3, 0.85)])
def test_lint_score_drops_per_issue_and_ignores_the_users_ruff_config(tmp_path, issues, score):
    # a project config above the runs dir that would silence F821
    (tmp_path / "pyproject.toml").write_text('[tool.ruff.lint]\nignore = ["F821"]\n', encoding="utf-8")
    ws = tmp_path / "runs" / "Finch-01"
    ws.mkdir(parents=True)
    code = "".join(f"a{i} = undefined_{i}\n" for i in range(issues))  # one F821 per line
    res = evaluate(_artifact(ws, code), RunConfig(problem="x", lint=True))
    assert res.lint_ran is True and res.lint_ok is False
    assert res.lint_score == score


async def test_workers_are_evaluated_concurrently_off_the_event_loop(tmp_path):
    arts = []
    for i in range(3):
        ws = tmp_path / f"Finch-0{i + 1}"
        ws.mkdir()
        art = _artifact(ws, "x = 1\n")
        arts.append(art.model_copy(update={"worker_id": f"Finch-0{i + 1}"}))
    cfg = _cfg(test_cmd=f'{PY} -c "import time; time.sleep(2)"')
    start = time.monotonic()
    evals = await qa_consensus.evaluate_all(arts, cfg)
    elapsed = time.monotonic() - start
    assert [e.worker_id for e in evals] == ["Finch-01", "Finch-02", "Finch-03"]  # input order
    assert all(e.tests_passed for e in evals)
    assert elapsed < 4.5, f"took {elapsed:.1f}s; one at a time would take 6+ s"


# --- nothing from the working directory runs or gets written ----------------


def test_test_cmd_does_not_load_the_working_directory_projects_conftest(tmp_path):
    proj = tmp_path / "proj"
    (proj / "runs" / "run-x" / "Finch-01").mkdir(parents=True)
    marker = tmp_path / "conftest-ran"
    (proj / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (proj / "conftest.py").write_text(
        f"import pathlib\npathlib.Path({str(marker)!r}).write_text('x')\n", encoding="utf-8"
    )
    code = "def test_ok():\n    assert True\n"
    art = _artifact(proj / "runs" / "run-x" / "Finch-01", code)
    res = evaluate(art, _cfg(test_cmd=f"{PY} -m pytest -q -p no:cacheprovider {{file}}"))
    assert res.tests_passed is True, res.test_summary
    assert not marker.exists()
    assert sorted(p.name for p in Path(art.workspace).iterdir()) == ["solution.py"]


def test_test_runs_in_a_private_folder_outside_the_working_directory(tmp_path):
    code = "import os\nprint('CWD=' + os.getcwd())\n"
    art = _artifact(tmp_path, code)
    res = evaluate(art, _cfg(test_cmd=f"{PY} {{file}}"))
    assert res.tests_passed is True, res.test_summary
    ran_in = Path(res.test_summary.split("CWD=", 1)[1].strip())
    assert ran_in.parent.resolve() == qa_consensus._private_test_base().resolve()
    assert not ran_in.exists()  # removed afterwards


def _fake_exe(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / (name + (".exe" if os.name == "nt" else ""))
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


def test_helpers_are_never_found_in_the_current_directory(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    _fake_exe(proj, "ruff")
    monkeypatch.chdir(proj)
    monkeypatch.setenv("PATH", os.pathsep.join([".", "", str(proj)]))
    assert qa_consensus._trusted_which("ruff") is None
    real = _fake_exe(tmp_path / "bin", "ruff")
    monkeypatch.setenv("PATH", os.pathsep.join([".", str(proj), str(real.parent)]))
    assert qa_consensus._trusted_which("ruff") == str(real)


def test_linter_starts_ruff_by_absolute_path_without_a_cache(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, stdout="[]", stderr="")

    ruff = str(tmp_path / "bin" / "ruff")
    monkeypatch.setattr(qa_consensus, "_trusted_which", lambda name: ruff)
    monkeypatch.setattr(qa_consensus.subprocess, "run", fake_run)
    art = _artifact(tmp_path, "x = 1\n")
    res = evaluate(art, RunConfig(problem="x", lint=True))
    assert res.lint_ran is True and res.lint_ok is True
    (argv, kw), = calls
    assert argv[0] == ruff
    assert "--no-cache" in argv and "--isolated" in argv
    assert os.path.isabs(argv[-1])
    assert kw["cwd"] == art.workspace


def test_no_linter_when_ruff_is_only_in_the_current_directory(tmp_path, monkeypatch):
    _fake_exe(tmp_path, "ruff")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", ".")
    res = evaluate(_artifact(tmp_path, "x = 1\n"), RunConfig(problem="x", lint=True))
    assert res.lint_ran is False


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")
def test_lint_writes_nothing_through_a_ruff_cache_link_in_the_working_directory(
    tmp_path, monkeypatch
):
    outside = tmp_path / "outside"
    outside.mkdir()
    proj = tmp_path / "proj"
    proj.mkdir()
    link = proj / ".ruff_cache"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        import _winapi

        _winapi.CreateJunction(str(outside), str(link))
    ws = proj / "runs" / "run-x" / "Finch-01"
    ws.mkdir(parents=True)
    monkeypatch.chdir(proj)
    res = evaluate(_artifact(ws, "a = undefined_name\n"), RunConfig(problem="x", lint=True))
    assert res.lint_ran is True and res.lint_score == 0.95
    assert list(outside.iterdir()) == []
    assert sorted(p.name for p in ws.iterdir()) == ["solution.py"]


@pytest.mark.skipif(os.name != "nt", reason="Windows-only kill path")
def test_timeout_kill_runs_taskkill_from_the_system_folder(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    class Proc:
        pid = 4242
        killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    monkeypatch.setattr(qa_consensus.subprocess, "run", fake_run)
    proc = Proc()
    assert qa_consensus._kill_tree(proc) is True
    exe = calls[0][0]
    assert os.path.isabs(exe) and exe.lower().endswith(r"\system32\taskkill.exe")
    assert proc.killed


@pytest.mark.skipif(os.name != "nt", reason="Windows-only kill path")
def test_a_failed_tree_kill_still_kills_the_process_and_says_so(monkeypatch):
    def failing_run(argv, **kw):
        raise OSError("taskkill missing")

    class Proc:
        pid = 4242
        killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    monkeypatch.setattr(qa_consensus.subprocess, "run", failing_run)
    proc = Proc()
    assert qa_consensus._kill_tree(proc) is False
    assert proc.killed


def test_python_without_a_usable_ruff_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(qa_consensus, "_trusted_which", lambda name: None)
    res = evaluate(_artifact(tmp_path, "x = 1\n"), RunConfig(problem="x", lint=True))
    assert res.lint_ran is False and res.lint_score == 1.0
    assert res.test_summary == "ruff not found on PATH outside the working directory"


def test_a_test_folder_that_cannot_be_removed_is_reported(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(qa_consensus.shutil, "rmtree", lambda *a, **kw: None)
    art = _artifact(tmp_path, "x = 1\n")
    with caplog.at_level("WARNING", logger="ops_solver.qa_consensus"):
        res = evaluate(art, _cfg(test_cmd=f"{PY} {{file}}"))
    assert res.tests_passed is True, res.test_summary
    left = [r.args[0] for r in caplog.records if "could not be removed" in r.getMessage()]
    assert len(left) == 1
    assert Path(left[0]).parent.resolve() == qa_consensus._private_test_base().resolve()
    assert Path(left[0]).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows-only kill path")
def test_kill_tree_really_kills_a_child_tree_on_windows(tmp_path):
    proc = qa_consensus._spawn("cmd /c ping -n 30 127.0.0.1", str(tmp_path))
    time.sleep(1)  # let cmd start ping
    assert qa_consensus._kill_tree(proc) is True
    # the pipes close only once ping, the grandchild, is gone too
    proc.communicate(timeout=10)
    assert proc.poll() is not None
