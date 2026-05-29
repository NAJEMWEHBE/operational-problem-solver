"""Deterministic local-environment probe (Intelligence team input).

We collect real, verifiable facts here; the Crows reason over them. We do NOT
pretend the LLM "scrapes logs" — it interprets what this module gathered.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_LINTERS = ["ruff", "eslint", "flake8", "pylint", "golangci-lint", "clippy"]
_TEST_RUNNERS = ["pytest", "npm", "go", "cargo", "jest", "vitest"]

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "runs", "dist", "build"}


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
        if (cwd / marker).exists():
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
    pkg = cwd / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
            deps += list(data.get("dependencies", {}).keys())
            deps += list(data.get("devDependencies", {}).keys())
        except (json.JSONDecodeError, OSError):
            pass
    req = cwd / "requirements.txt"
    if req.exists():
        try:
            for line in req.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    deps.append(line.split("==")[0].split(">")[0].split("<")[0].strip())
        except OSError:
            pass
    return deps[:50]


def _read_readme(cwd: Path, limit: int = 1200) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        p = cwd / name
        if p.exists():
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as fh:
                    return fh.read(limit)   # bounded read; don't load a huge file into memory
            except OSError:
                return ""
    return ""


def probe(cwd: Path, max_files: int = 60) -> dict:
    """Gather a lightweight, safe snapshot of the working directory."""
    cwd = Path(cwd)
    files: list[str] = []
    if cwd.exists():
        for p in cwd.rglob("*"):
            if p.is_symlink():                 # don't follow symlinks (loop / escape / DoS)
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if p.is_file():
                files.append(str(p.relative_to(cwd)))
                if len(files) >= max_files:     # bound the walk, don't materialize the whole tree
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
