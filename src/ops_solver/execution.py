"""Execution team (Finch-01 .. Finch-NN).

Parallel, isolated solution workers. All workers share ONE model and the same
cached ContextManifest prefix (so caching hits across them); each gets a distinct
strategy nudge to avoid groupthink. Each worker's output is written
to its own workspace directory.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import RunConfig
from .llm import LLM, FanOutResults, fan_out
from .models import ContextManifest, WorkerArtifact

_EXEC_SYSTEM = (
    "You are an expert engineer. Produce a single, complete, self-contained "
    "solution to the problem. Output exactly one fenced code block containing the "
    "full solution (no ellipses, no placeholders, no 'rest of code here'), then a "
    "few sentences of explanation. Honour the context manifest's language, "
    "dependencies, and constraints."
)

_STRATEGIES = [
    "the simplest correct implementation",
    "a performance-optimised implementation",
    "a robust, defensive implementation that handles edge cases and bad input",
    "a clean, idiomatic, highly readable implementation",
    "a test-driven implementation that includes its own tests",
    "a minimal-dependency, portable implementation",
]

_LANG_EXT = {
    "python": "py",
    "javascript": "js",
    "typescript": "ts",
    "go": "go",
    "rust": "rs",
    "java": "java",
    "ruby": "rb",
    "c": "c",
    "cpp": "cpp",
    "csharp": "cs",
    "bash": "sh",
    "shell": "sh",
    "html": "html",
}

# Short fence tags models often write, mapped to the names _LANG_EXT knows.
_LANG_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "node": "javascript",
    "ts": "typescript",
    "rs": "rust",
    "golang": "go",
    "rb": "ruby",
    "c++": "cpp",
    "cs": "csharp",
    "c#": "csharp",
    "sh": "bash",
    "zsh": "bash",
}


def _canonical_lang(language: str) -> str:
    language = language.strip()
    return _LANG_ALIASES.get(language.lower(), language)


_FENCE_RE = re.compile(r"```([\w+#-]*)\n(.*?)```", re.DOTALL)


def _manifest_block(manifest: ContextManifest) -> str:
    return (
        f"PROBLEM:\n{manifest.problem}\n\n"
        f"LANGUAGE: {manifest.language}\n"
        f"DEPENDENCIES: {', '.join(manifest.dependencies) or 'use only the standard library unless necessary'}\n"
        f"CONSTRAINTS: {'; '.join(manifest.constraints) or 'none stated'}\n"
        f"RISKS / EDGE CASES: {'; '.join(manifest.risks) or 'none stated'}\n"
        f"NOTES: {manifest.notes or 'none'}"
    )


def extract_solution(text: str, default_lang: str) -> tuple[str, str, str]:
    """Return (code, language, explanation) from a worker response.

    Picks the largest fenced code block as the solution; everything outside the
    blocks is treated as explanation. Falls back to the whole text as code."""
    default_lang = _canonical_lang(default_lang)
    blocks = _FENCE_RE.findall(text or "")
    if not blocks:
        return text.strip(), default_lang, ""
    lang, code = max(blocks, key=lambda b: len(b[1]))
    language = _canonical_lang(lang or "") or default_lang
    explanation = _FENCE_RE.sub("", text).strip()
    return code.strip(), language, explanation


def _filename(language: str) -> str:
    ext = _LANG_EXT.get(_canonical_lang(language).lower(), "txt")
    return f"solution.{ext}"


async def run_workers(
    llm: LLM,
    cfg: RunConfig,
    manifest: ContextManifest,
    run_dir: Path,
    attempt: int = 0,
    feedback: str = "",
) -> list[WorkerArtifact]:
    system = [
        {"type": "text", "text": _EXEC_SYSTEM},
        {"type": "text", "text": _manifest_block(manifest), "cache_control": {"type": "ephemeral"}},
    ]
    # Re-loops inject prior-failure feedback into every worker's prompt.
    feedback_note = f"\n\nA previous attempt had problems: {feedback}\nAvoid them." if feedback else ""

    def worker(i: int):
        wid = f"Finch-{i + 1:02d}"
        strategy = _STRATEGIES[i % len(_STRATEGIES)]
        user = (
            f"Solve the problem using this approach: {strategy}.{feedback_note}\n\n"
            "Produce the complete solution now."
        )

        async def run() -> WorkerArtifact | None:
            text = await llm.complete(
                model=cfg.worker_model,
                system=system,
                user=user,
                max_tokens=cfg.worker_max_tokens,
            )
            if not text:
                return None
            code, language, explanation = extract_solution(text, manifest.language)
            fname = _filename(language)
            ws = run_dir / wid
            if ws.exists():  # a re-loop: drop the previous attempt's files
                shutil.rmtree(ws)
            ws.mkdir(parents=True)
            (ws / fname).write_text(code, encoding="utf-8")
            if explanation:
                (ws / "NOTES.md").write_text(explanation, encoding="utf-8")
            return WorkerArtifact(
                worker_id=wid,
                strategy=strategy,
                language=language,
                filename=fname,
                code=code,
                explanation=explanation,
                workspace=str(ws),
            )

        return run

    results = await fan_out([worker(i) for i in range(cfg.exec_workers)])
    # keep the failures' causes so the orchestrator can report them
    return FanOutResults((a for a in results if isinstance(a, WorkerArtifact)), results.errors)
