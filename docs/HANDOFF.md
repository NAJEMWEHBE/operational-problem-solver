# Handoff — operational-problem-solver

Snapshot for whoever picks this up next. Built and shipped 2026-05-29.

## What it is
A multi-team agentic engine for Claude Code: parallel **Intelligence** (Crow) → **Execution** (Finch) worker teams generate competing solutions, a **consensus stability gate** drops the unstable ones, and an **ELO tournament** (LLM-as-judge) ranks the survivors and ships the single best. Two halves: a Python engine (`ops-solve`) + a Claude Code skill.

## Status: shipped + verified
- Repo: https://github.com/NAJEMWEHBE/operational-problem-solver (public, MIT)
- Live page: https://najemwehbe.github.io/operational-problem-solver/
- Tests: **47 passing**, `ruff` clean, CI green on Python 3.11/3.12/3.13
- Live engine run verified end-to-end: **~$0.028 / run** (4 calls, `--cheap`), winner written to `runs/<id>/WINNER/`

## Run the engine
```powershell
cd "F:\ai\problem solver"
.\.venv\Scripts\Activate.ps1            # venv already created
# ANTHROPIC_API_KEY is in .env (gitignored). Then:
ops-solve "solve: <problem>" --workers 4 --intel 2
#   --cheap        Haiku workers (cheapest)
#   --test-cmd "pytest -q"   gate consensus on real tests (EXECUTES generated code)
#   --out <path>   write winner file there   (--force to overwrite)
```
Offline logic tests (no API spend): `pytest -q`.

## The skill (auto-use)
Installed globally at `C:\Users\NINOH\.claude\skills\problem-solver\SKILL.md` → available in every Claude Code session. A `problem-solver` route in `~/.claude/hooks/skill-routes.json` makes Claude reach for it **only** when a task clearly wants "several competing solutions → pick the best" (or `solve:` / the name). Default is NOT to use it; trivial asks get a plain answer. The engine (~8–12 metered calls) only runs when it genuinely fits, with a cost-confirm first.

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md). Modules under `src/ops_solver/`: `config`, `models`, `llm` (Anthropic adapter, prompt caching, fan-out, cost ledger), `env_probe`, `intelligence`, `execution`, `qa_consensus`, `elo`, `orchestrator`, `cli`.

## Dev workflow (important)
- **`main` is branch-protected** — no direct pushes. Branch → PR → required CI (`test 3.11/3.12/3.13`) must pass.
- **Merge quirk (solo repo):** `gh pr merge <n> --squash` reports `BLOCKED` even with green CI + 0 required approvals. Merge with `--admin` (owner bypass, CI already verified) or the GitHub UI as owner.
- After `gh pr create`, give Actions a few seconds before `gh pr checks --required --watch` (it can race with "no checks reported").

## Security / keys
- `ANTHROPIC_API_KEY` lives **only** in `.env` (gitignored, verified uncommitted). Never in the repo or git history. Rotate/revoke anytime in the Anthropic console.
- The engine never logs or serializes the key (`report.json` stores token counts only).
- `--test-cmd` executes model-generated code — opt-in, documented; run untrusted problems in a sandbox.

## Deferred / known
- **CodeRabbit** app installed but **out of org credits** — can't review until topped up at app.coderabbit.ai. **cubic** AI reviewer also installed (sat pending). Neither is a required check.
- Landing page loads Google Fonts (visitor-IP/SRI) — self-host later if it matters.
- Optional next steps: publish engine to PyPI; add JS/Go syntax+lint to QA Layer 1; record a real-run demo GIF for the README.
