---
name: operational-problem-solver
description: A multi-team, multi-tool agentic engine that orchestrates parallel worker groups (Intelligence, Execution, QA) to solve complex technical problems, using a consensus stability gate and an ELO tournament to select and ship the single best solution. Trigger when the user says "solve:", /problem-solver, or asks to generate several competing solutions and pick the best.
---

# Multi-Team Operational Sandbox

## Core Architecture
A central orchestrator commands three specialized worker teams. Each team runs
multiple instances in parallel.

```
                 [ Central Orchestrator ]
                /          |             \
   [ Intelligence ]   [ Execution ]   [ Quality Assurance ]
      Crow x2            Finch x4       Consensus + ELO
```

### 1. Intelligence Team (Research & Context)
- **Workers:** 2+ parallel instances (`Crow-Alpha`, `Crow-Beta`).
- **Objective:** Read the codebase/directory, detect language and dependencies,
  scan local docs, and pull baseline requirements.
- **Output:** A unified **Context Manifest** (dependencies, constraints, parameters, risks).

### 2. Execution Team (Multi-Tool Workers)
- **Workers:** 4+ parallel instances (`Finch-01` .. `Finch-04`).
- **Objective:** Process the Context Manifest independently and simultaneously.
  Each worker writes its own standalone solution.
- **Rule:** Workers are isolated (own workspace dir, distinct strategy + temperature)
  to prevent groupthink.

### 3. Quality Assurance Team (Consensus & Tournament)
- **Layer 1 — Consensus:** Syntax-check, lint, and (optionally) test every
  artifact. If fewer than 50% are stable, discard, mutate the prompt variables,
  and re-loop Execution.
- **Layer 2 — ELO Tournament:** Force the survivors into head-to-head pairwise
  judging (correctness, cleanliness, speed, footprint). Update ELO ratings, blend
  with the objective scores, and produce a ranked leaderboard. The top entry wins.

## How this actually runs in Claude Code

A skill is instructions — it cannot spawn OS processes itself. There are two
execution paths; **prefer Path A.**

### Path A — the Python engine (primary, deterministic ELO)
If the `ops-solver` engine is installed and `ANTHROPIC_API_KEY` is set, shell out
to it. This does the real parallel fan-out, runs linters/tests, and computes ELO
in code:

```bash
ops-solve "solve: <the problem>" --workers 4 --intel 2
# add --test-cmd "pytest -q" to gate consensus on real tests (executes code)
# add --cheap to use Haiku workers; --out path/to/file to write the winner
```

Then report the leaderboard, the winning solution, and the printed cost.

### Path B — native subagents (fallback, no engine/key)
If the engine or key is unavailable, orchestrate it in-session with the Task tool:
1. **Phase I:** Spawn 2 Explore/research subagents to build the Context Manifest
   (deps, constraints, params) for the directory + problem.
2. **Phase II:** Spawn 4 independent subagents, each given the SAME manifest and a
   distinct strategy (simplest / fastest / most robust / cleanest). Each returns a
   complete standalone solution.
3. **Phase III:** Run a judge subagent that pairwise-compares the solutions and
   reports a ranked order. ELO here is reasoned by the model, not computed —
   say so; it is less rigorous than Path A.
4. If fewer than half the solutions are viable, re-prompt and re-run Phase II once.
5. Present the winner.

## ⚠️ Cost (read before running)
Every worker and judge call is a **metered Claude API call** (Path A) or spends
**your in-session tokens** (Path B). The spec-default fan-out is ~8–12 calls per
run. This is **not free**. Use `--cheap`, fewer `--workers`, and prompt caching
(built into the engine) to control spend. The engine prints an estimated cost.

## Execution Protocol (summary)
1. **Trigger:** user says `solve: <problem>` or invokes `/problem-solver`.
2. **Phase I:** Intelligence audits the environment → Context Manifest.
3. **Phase II:** Execution workers generate diverse standalone solutions in parallel.
4. **Phase III:** QA runs the consensus gate (re-loop if <50% stable), then the
   ELO tournament.
5. **Output:** ship the highest-rated solution (engine writes it to `runs/<id>/WINNER/`
   and, with `--out`, to your chosen path).
