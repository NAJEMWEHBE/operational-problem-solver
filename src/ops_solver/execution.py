"""Execution team (Finch-01 .. Finch-NN).

Parallel, isolated solution workers. All workers share ONE model and the same
cached ContextManifest prefix (so caching hits across them); each gets a distinct
strategy nudge + temperature to avoid groupthink. Each worker's output is written
to its own workspace directory.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import RunConfig, supports_sampling
from .llm import LLM, fan_out
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

_FENCE_RE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


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
    blocks = _FENCE_RE.findall(text or "")
    if not blocks:
        return text.strip(), default_lang, ""
    lang, code = max(blocks, key=lambda b: len(b[1]))
    language = (lang or "").strip() or default_lang
    explanation = _FENCE_RE.sub("", text).strip()
    return code.strip(), language, explanation


def _filename(language: str) -> str:
    ext = _LANG_EXT.get(language.lower(), "txt")
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
    # Re-loops raise the diversity temperature and inject prior-failure feedback.
    temp_bump = 0.1 * attempt
    feedback_note = f"\n\nA previous attempt had problems: {feedback}\nAvoid them." if feedback else ""

    def worker(i: int):
        wid = f"Finch-{i + 1:02d}"
        strategy = _STRATEGIES[i % len(_STRATEGIES)]
        user = (
            f"Solve the problem using this approach: {strategy}.{feedback_note}\n\n"
            "Produce the complete solution now."
        )
        temperature = None
        if supports_sampling(cfg.worker_model):
            temperature = min(1.0, cfg.base_temperature + cfg.temperature_step * i + temp_bump)

        async def run() -> WorkerArtifact | None:
            text = await llm.complete(
                model=cfg.worker_model,
                system=system,
                user=user,
                max_tokens=cfg.worker_max_tokens,
                temperature=temperature,
            )
            if not text:
                return None
            code, language, explanation = extract_solution(text, manifest.language)
            fname = _filename(language)
            ws = run_dir / wid
            ws.mkdir(parents=True, exist_ok=True)
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
    return [a for a in results if isinstance(a, WorkerArtifact)]
