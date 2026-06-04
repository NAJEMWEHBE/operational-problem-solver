# Session Handoff — operational-problem-solver build (2026-05-29)

Full log of the session that created this project, so a fresh session can resume with zero context loss. For steady-state project info see [HANDOFF.md](HANDOFF.md).

## Goal (user's original ask)
Build a multi-team, multi-tool agentic engine for Claude Code: parallel worker groups (Intelligence / Execution / QA) that solve a problem, gate on a consensus threshold, and run an ELO tournament to ship the single best solution. First check if it already exists; if not, build it and publish as a GitHub repo.

## What happened, in order
1. **Prior-art research** (deep-research, 3 parallel agents). Verdict: the *pieces* are proven (AlphaCode, self-consistency, CodeT, LLM-as-judge/MT-Bench, Chatbot-Arena Elo, Mixture-of-Agents; tools RAGElo/CodeElo/llm-council; AlphaEvolve is the closest all-in-one but Gemini/closed). The *exact packaged combo* (Intelligence→Execution→QA consensus→ELO tournament as one Claude Code skill) was not on the shelf. Conclusion: **novel as an integration, not as research.**
2. **Decisions** (confirmed with user): skill + Python engine; spec-default fan-out (2 intel + 4 exec + 1 judge); repo `operational-problem-solver`, public, MIT.
3. **Built the engine** — Python 3.11, `anthropic` Messages SDK (NOT the heavier agent SDK), prompt caching, cache-warmed async fan-out, tiered models (Haiku/Sonnet workers, Opus judge), token/cost ledger; env probe; the three teams; consensus gate; ELO tournament; orchestrator; Typer CLI. Plus the skill (`.claude/skills/problem-solver/SKILL.md`) and 13 offline tests. Verified offline (pytest + ruff). Pushed to GitHub (public, MIT).
4. **Landing page** — `docs/index.html` (vibrant dark-tech, animated hero, interactive ELO demo, code window), hand-built animated `docs/assets/hero.svg` (no diffusion image-gen on this Windows box — SVG instead), README banner + shields. GitHub Pages enabled (`main` `/docs`), site live.
5. **Review + hardening** — opened a PR, ran a 4-bot panel (correctness, silent-failure, security, test-coverage). Found real bugs: wasteful double API call in `structured()`, silent-failure pipeline (no logging), `.env` arbitrary-var injection, `--out` clobber, symlink/unbounded reads, lint/syntax mislabels. Fixed all → **PR #1** (13→43 tests, added CI). CodeRabbit app present but **out of credits**; cubic pending.
6. **Merged PR #1**, set **branch protection** on `main` (CI-gated, PR-required, no force-push/delete, linear), **installed the skill globally** (`~/.claude/skills/problem-solver/`).
7. **Found a real bug live in Chrome** — hero invisible under `prefers-reduced-motion` (user's browser has it on). Fixed via **PR #2** (merged through the new protection).
8. **Live smoke** — user put their `ANTHROPIC_API_KEY` in `.env`. First run failed: an **empty `ANTHROPIC_API_KEY` in the environment shadowed `.env`** because the loader used `setdefault`. Fixed loader to override empty values → **PR #3** (47 tests). Live run then succeeded: **$0.028**, winner shipped.
9. **Auto-use wiring** — added a `problem-solver` route to `~/.claude/hooks/skill-routes.json` so Claude reaches for the skill on its own *only when a task wants several competing solutions → pick the best*. Default is NOT to use it; cost-confirm before any non-trivial metered run.

## Current state
- Repo: https://github.com/NAJEMWEHBE/operational-problem-solver — public, MIT, 3 PRs merged, CI green.
- Live: https://najemwehbe.github.io/operational-problem-solver/ (hero confirmed rendering).
- 47 tests passing; engine verified live ($0.028/run).
- `main` branch-protected; skill global + auto-route active.
- Key: in `.env` only (gitignored, verified uncommitted).

## Gotchas for the next session
- **GitHub handle is `NAJEMWEHBE`** (not "najmwahba"); `gh` authed (repo+workflow scopes).
- **Merging PRs:** `gh pr merge --squash` reports `BLOCKED` even with green CI (solo repo, 0 approvals) → use `--admin` or the GitHub UI as owner.
- **`gh pr checks --required --watch`** can race right after `pr create` ("no checks reported") — wait a few seconds.
- The engine venv is `F:\ai\problem solver\.venv`; `ops-solve` is not on global PATH (run from the venv).
- User works in the Claude **Desktop app** (not CLI); skills/hooks apply in Code-tab sessions, not the plain Chat tab.

## Open / deferred
- CodeRabbit needs org credits to actually review; cubic reviewer pending.
- Optional: publish engine to PyPI; JS/Go support in QA Layer 1; demo GIF for README; self-host landing-page fonts.
