"""Intelligence team (Crow-Alpha, Crow-Beta, ...).

Parallel context gathering: each Crow interprets the probed environment from a
distinct angle and returns a structured ManifestDraft. The drafts merge into one
ContextManifest handed to every Execution worker.
"""

from __future__ import annotations

import logging

from .config import RunConfig
from .llm import LLM, fan_out
from .models import ContextManifest, ManifestDraft

logger = logging.getLogger("ops_solver.intelligence")

_INTEL_SYSTEM = (
    "You are a software intelligence analyst. Given a problem statement and facts "
    "about the local environment, extract the dependencies, constraints, key "
    "parameters, and risks that a developer must respect when solving it. Be "
    "concrete and concise. Do not write a solution."
)

_ANGLES = [
    "dependencies, libraries, and the runtime/environment",
    "hard constraints, required behaviours, and edge cases / failure modes",
    "performance, scalability, and resource footprint considerations",
    "testing, validation, and what 'correct' must mean for this problem",
]


def _facts_block(probe: dict) -> str:
    tree = "\n".join(probe.get("file_tree", [])[:40]) or "(empty directory)"
    return (
        f"DETECTED LANGUAGE: {probe.get('language')}\n"
        f"DEPENDENCIES FOUND: {', '.join(probe.get('dependencies', [])) or 'none'}\n"
        f"LINTERS AVAILABLE: {', '.join(probe.get('linters_available', [])) or 'none'}\n"
        f"TEST RUNNERS: {', '.join(probe.get('test_runners_available', [])) or 'none'}\n"
        f"FILE TREE:\n{tree}\n\n"
        f"README EXCERPT:\n{probe.get('readme_excerpt', '') or '(none)'}"
    )


def _merge(drafts: list[ManifestDraft]) -> dict:
    deps: list[str] = []
    cons: list[str] = []
    risks: list[str] = []
    params: dict[str, str] = {}
    notes: list[str] = []

    def add(dst: list[str], items: list[str]) -> None:
        for it in items:
            it = (it or "").strip()
            if it and it not in dst:
                dst.append(it)

    for d in drafts:
        add(deps, d.dependencies)
        add(cons, d.constraints)
        add(risks, d.risks)
        params.update(d.parameters)
        if d.notes.strip():
            notes.append(d.notes.strip())

    return {
        "dependencies": deps,
        "constraints": cons,
        "risks": risks,
        "parameters": params,
        "notes": "\n".join(notes),
    }


async def gather_context(llm: LLM, cfg: RunConfig, problem: str, probe: dict) -> ContextManifest:
    facts = _facts_block(probe)
    system = [
        {"type": "text", "text": _INTEL_SYSTEM},
        {"type": "text", "text": facts, "cache_control": {"type": "ephemeral"}},
    ]

    def crow(i: int):
        angle = _ANGLES[i % len(_ANGLES)]
        user = (
            f"PROBLEM:\n{problem}\n\n"
            f"Focus your analysis on: {angle}.\n"
            "Return a structured draft of the context manifest for this focus."
        )
        return lambda: llm.structured(
            model=cfg.intel_model,
            system=system,
            user=user,
            schema=ManifestDraft,
            max_tokens=cfg.intel_max_tokens,
            temperature=0.3 + 0.15 * i,
        )

    raw = await fan_out([crow(i) for i in range(cfg.intel_workers)])
    drafts = [d for d in raw if isinstance(d, ManifestDraft)]
    if cfg.intel_workers > 0 and not drafts:
        raise RuntimeError(
            f"Intelligence phase produced 0 usable manifest drafts out of "
            f"{cfg.intel_workers} ({cfg.intel_model}). All Crow calls failed or returned "
            f"unparseable output - check ANTHROPIC_API_KEY, the model id, and rate limits."
        )
    if len(drafts) < cfg.intel_workers:
        logger.warning("intelligence: %d/%d Crow drafts survived", len(drafts), cfg.intel_workers)
    merged = _merge(drafts)

    language = cfg.language if cfg.language != "auto" else probe.get("language", "auto")
    return ContextManifest(problem=problem, language=language, **merged)
