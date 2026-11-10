"""Deterministic local-environment probe (Intelligence team input).

We collect real, verifiable facts here; the Crows reason over them. We do NOT
pretend the LLM "scrapes logs" — it interprets what this module gathered.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

_LINTERS = ["ruff", "eslint", "flake8", "pylint", "golangci-lint", "clippy"]
_TEST_RUNNERS = ["pytest", "npm", "go", "cargo", "jest", "vitest"]

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "runs", "dist", "build"}

# Everything read here ends up in every Intelligence request, so every read and
# every entry has a bound.
_README_CHARS = 1200
_MANIFEST_BYTES = 256 * 1024     # package.json / requirements.txt
_MAX_DEPS = 50
_DEP_CHARS = 100                 # per dependency entry; real package names are far shorter


def is_regular_file(path: Path) -> bool:
    """True only when `path` itself is a regular file. A symlink (or junction,
    directory, FIFO, device) never counts, so it can't make a reader look at a
    path outside the folder it was given."""
    try:
        return stat.S_ISREG(Path(path).lstat().st_mode)
    except OSError:
        return False


def read_regular_file(path: Path, max_bytes: int) -> str | None:
    """Up to `max_bytes` of the regular file at `path`, decoded as UTF-8; None when
    it is absent, unreadable or not a regular file. Opened without following a
    link where the OS allows it, and the opened file must be the one lstat() saw.
    Used for every file read from the working directory."""
    p = Path(path)
    try:
        before = p.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(before.st_mode):
        return None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        fd = os.open(p, flags)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "rb") as fh:
            st = os.fstat(fh.fileno())
            if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                return None  # swapped between lstat() and open()
            data = fh.read(max_bytes)
    except OSError:
        return None
    return data.decode("utf-8", errors="ignore")


def _detect_language(cwd: Path, files: list[str]) -> str:
    markers = {
        "pyproject.toml": "python",
        "requirements.txt": "python",
        "setup.py": "python",
        "package.json": "typescript",
        "tsconfig.json": "typescript",
        "go.mod": "go",
        "Cargo.toml": "rust",
        "pom.xml": "java",
        "build.gradle": "java",
    }
    for marker, lang in markers.items():
        if is_regular_file(cwd / marker):
            return lang
    ext_counts: dict[str, int] = {}
    ext_lang = {
        ".py": "python",
        ".ts": "typescript",
        ".js": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
    }
    for f in files:
        suffix = Path(f).suffix
        if suffix in ext_lang:
            ext_counts[ext_lang[suffix]] = ext_counts.get(ext_lang[suffix], 0) + 1
    if ext_counts:
        return max(ext_counts, key=ext_counts.get)
    return "auto"


def _read_deps(cwd: Path) -> list[str]:
    deps: list[str] = []
    pkg = read_regular_file(cwd / "package.json", _MANIFEST_BYTES)
    if pkg is not None:
        try:
            data = json.loads(pkg)
        except (ValueError, RecursionError):
            # not JSON, a number too long to convert, or nesting too deep
            data = None
        if isinstance(data, dict):
            for section in ("dependencies", "devDependencies"):
                names = data.get(section)
                if isinstance(names, dict):
                    deps += [str(n) for n in names]
    req = read_regular_file(cwd / "requirements.txt", _MANIFEST_BYTES)
    if req is not None:
        for line in req.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(line.split("==")[0].split(">")[0].split("<")[0].strip())
    return [d[:_DEP_CHARS] for d in deps[:_MAX_DEPS]]


def _read_readme(cwd: Path, limit: int = _README_CHARS) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        if is_regular_file(cwd / name):
            # bounded read (UTF-8 is at most 4 bytes per character)
            return (read_regular_file(cwd / name, limit * 4) or "")[:limit]
    return ""


def probe(cwd: Path, max_files: int = 60) -> dict:
    """Gather a lightweight, safe snapshot of the working directory."""
    cwd = Path(cwd)
    files: list[str] = []
    if cwd.exists():
        # os.walk lets us prune skipped dirs (node_modules, .git, ...) before
        # descending, and the skip test sees only names *inside* cwd, so a parent
        # folder called build/ or runs/ doesn't hide the whole project.
        for root, dirnames, filenames in os.walk(cwd):   # followlinks=False by default
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            base = Path(root)
            for name in sorted(filenames):
                p = base / name
                if p.is_symlink():                 # don't follow symlinks (loop / escape / DoS)
                    continue
                files.append(p.relative_to(cwd).as_posix())
                if len(files) >= max_files:     # bound the walk, don't materialize the whole tree
                    break
            if len(files) >= max_files:
                break
        files.sort()

    return {
        "language": _detect_language(cwd, files),
        "file_tree": files,
        "dependencies": _read_deps(cwd),
        "linters_available": [t for t in _LINTERS if shutil.which(t)],
        "test_runners_available": [t for t in _TEST_RUNNERS if shutil.which(t)],
        "readme_excerpt": _read_readme(cwd),
    }
