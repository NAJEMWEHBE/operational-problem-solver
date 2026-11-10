---
name: problem-solver
description: 'solve: <problem>, or /problem-solver. Generates competing solutions in parallel, drops unstable ones at a consensus gate, ranks the rest in an ELO tournament and ships the winner. Use when a task wants several competing solutions and the best one picked.'
---

# Problem solver

A tournament of competing solutions: Intelligence workers (Crow) build a Context
Manifest, Execution workers (Finch) each write a standalone solution, a consensus
gate drops the unstable ones, and an ELO tournament ranks the survivors. How the
engine works: `docs/ARCHITECTURE.md` in this repo.

**Credentials guardrail:** check only that a credential *exists*, and refer to it
by name (`ANTHROPIC_API_KEY`); its value stays out of the chat, commands, logs
and files.

## Path A: the `ops-solve` engine (prefer it)

Computes ELO in code and runs real lint and tests.

1. **Detect.** `ops-solve --help` exits 0. If it doesn't, take Path B.
2. **Confirm the cost.** Every call is metered: 18 API calls with the defaults
   (2 Crow + 4 Finch + 6 judge pairs x 2 orders on Opus), 22 with a re-loop. Tell
   the user the count and get a yes before running.
3. **Run.**
   ```bash
   ops-solve "solve: <the problem>" --workers 4 --intel 2
   ```
   Options worth offering: `--cheap` (Haiku workers), `--workers 3` (11 calls),
   `--out path/to/file` (`--force` to overwrite), and
   `--test-cmd "pytest -q {file}"` to gate on real tests. `--test-cmd` executes
   generated code, so use it only with the user's yes; `{file}` is the solution
   file name and the command runs in a private folder outside the working
   directory that holds a copy of the solution file.
4. **Finish line:** exit code 0 and `runs/<run-id>/report.json` exists (the newest
   folder in `runs/`). Report the leaderboard, the `WINNER/` path, the printed
   calls and cost, and every `!` warning line.
5. **On failure:**
   - exit 2, "ANTHROPIC_API_KEY is not set": no credentials. Ask the user to set
     the key (or `ant auth login`), or take Path B.
   - exit 1, "Run failed: <Type>: <message>": report the message. For
     `RateLimitError`, offer a re-run with `--concurrency 2`; for
     `AuthenticationError`, the key is wrong; otherwise re-run with `--debug` for
     the traceback.

## Path B: subagents in this session (no engine or no key)

Spends the session's own tokens. Use the Agent tool:

1. **Phase I:** 2 research subagents build the Context Manifest (dependencies,
   constraints, parameters, risks) for the directory and problem.
2. **Phase II:** 4 independent subagents get the SAME manifest and one distinct
   strategy each (simplest / fastest / most robust / cleanest), and each returns
   a complete standalone solution.
3. **Gate:** if fewer than half the solutions are viable, re-prompt with what
   failed and re-run Phase II once.
4. **Phase III:** a judge subagent compares the solutions pairwise and returns a
   ranked order. This ELO is reasoned by the model, not computed; say so.
5. **Finish line:** the winning solution presented with the ranking and the
   judge's one-line reasons.
