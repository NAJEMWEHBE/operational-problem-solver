# operational-problem-solver

A multi-team agentic engine for Claude Code. It spins up **parallel worker teams**
to solve a technical problem, gates their output on a **consensus stability
threshold**, then runs an **ELO tournament** to rank the surviving solutions
head-to-head and ship the single best one.

```
                 [ Central Orchestrator ]
                /          |             \
   [ Intelligence ]   [ Execution ]   [ Quality Assurance ]
      Crow x2            Finch x4       Consensus + ELO
   context manifest   N solutions     gate -> tournament -> winner
```

Two halves, one repo:
- **A Claude Code skill** (`.claude/skills/problem-solver/`) — the in-session
  orchestration layer.
- **A standalone Python engine** (`ops-solve`) — the real work: parallel fan-out
  over the Anthropic Messages API, linters/tests via subprocess, and a
  deterministic ELO tournament.

---

## Has this been done before?

The **pieces** are well-established; this **exact packaged combination** for
Claude Code was not on the shelf when this was built. Honest prior art:

- **Parallel Claude-agent orchestration already exists** —
  [claude-flow](https://github.com/ruvnet/claude-flow),
  [claude_code_agent_farm](https://github.com/Dicklesworthstone/claude_code_agent_farm),
  [swarms](https://github.com/kyegomez/swarms), and the native Claude Code Task /
  Workflow tools. None ship a consensus-gated ELO tournament over competing code
  solutions as a packaged skill.
- **The algorithms are proven research** —
  [Self-Consistency](https://arxiv.org/abs/2203.11171),
  [AlphaCode (generate → filter → cluster → rank)](https://arxiv.org/abs/2203.07814),
  [CodeT](https://arxiv.org/abs/2207.10397),
  [LLM-as-judge / MT-Bench](https://arxiv.org/abs/2306.05685),
  [Chatbot Arena (Bradley-Terry / Elo)](https://arxiv.org/abs/2403.04132),
  [Mixture-of-Agents](https://arxiv.org/abs/2406.04692).
- **ELO-ranking of agents/pipelines exists as tooling** —
  [RAGElo](https://github.com/zetaalphavector/RAGElo),
  [CodeElo](https://github.com/QwenLM/CodeElo),
  [llm-council](https://github.com/karpathy/llm-council). The closest end-to-end
  analog is [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
  (generate + evaluate + evolve) — but it is Gemini/Google and not packaged for Claude.

So this project is **novel as an integration, not as research.** It reuses
known-good ELO / Bradley-Terry math and the AlphaCode / LLM-as-judge patterns.

---

## ⚠️ Cost — this is not free

Each worker and the judge are **metered Claude API calls**. The default fan-out
(2 Intelligence + 4 Execution + a judge running pairwise rounds) is roughly
**8–12 calls per run**. Controls:

- `--cheap` runs workers on Haiku 4.5; the judge stays on Opus 4.8.
- Fewer `--workers`, fewer `--rounds`.
- Prompt caching is built in: all workers share one model + a cached Context
  Manifest prefix, and the judge caches the shared problem context across rounds.

The engine prints an **estimated USD cost** at the end of every run.

The offline test suite (`pytest`) exercises the ELO math, the consensus gate, and
the full orchestrator loop with a **mocked** model — **zero API spend.**

---

## Setup

```powershell
git clone https://github.com/NAJEMWEHBE/operational-problem-solver
cd operational-problem-solver
python -m venv .venv; .venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -e .
Copy-Item .env.example .env                            # then edit: ANTHROPIC_API_KEY=sk-ant-...
pytest -q                                              # offline logic tests, no API spend
```

macOS / Linux: `source .venv/bin/activate` and `cp .env.example .env`.

---

## Usage

```bash
# Smallest real smoke test (cheap):
ops-solve "solve: write a Python is_prime(n) with a few tests" --workers 2 --cheap

# Gate consensus on real tests (EXECUTES the generated code — see Security):
ops-solve "solve: write a Python is_prime(n) with tests" --workers 4 --test-cmd "pytest -q"

# Write the winning file somewhere:
ops-solve "solve: a CLI that reverses stdin" --out ./reverse.py
```

Options: `--workers/-w`, `--intel/-i`, `--rounds/-r`, `--reloops`, `--test-cmd`,
`--out`, `--cheap`, `--lang`.

Each run writes to `runs/<run-id>/`:
- `Finch-NN/` — each worker's isolated solution + notes
- `WINNER/` — the shipped solution
- `report.json` — manifest, evals, leaderboard, winner, token/cost summary

---

## Using the skill in Claude Code

Copy the skill into a project so Claude Code discovers it:

```
.claude/skills/problem-solver/SKILL.md
```

Then `/problem-solver` or say `solve: <problem>`. With the engine installed and a
key present, the skill shells out to `ops-solve` (deterministic ELO). Otherwise it
orchestrates native Task subagents in-session (ELO reasoned by the model, less
rigorous). See [SKILL.md](.claude/skills/problem-solver/SKILL.md).

---

## How it works

1. **Intelligence (Crow):** `env_probe` collects real facts (file tree, language,
   deps, available linters). Parallel Crows interpret them into one
   `ContextManifest`.
2. **Execution (Finch):** parallel isolated workers — same manifest, distinct
   strategy + temperature — each emit a complete standalone solution into its own
   workspace.
3. **QA Layer 1 (Consensus):** syntax-check + lint + optional tests per artifact.
   If `< 50%` are stable, mutate prompts and re-loop Execution.
4. **QA Layer 2 (ELO):** survivors are judged pairwise by Opus; a standard ELO
   update (`K=32`, start `1000`) plus the objective scores produces a ranked
   leaderboard. The top entry is shipped.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow and formulas.

---

## Security

- Syntax checks (`ast.parse`) and linting (`ruff`) **do not execute** the
  generated code.
- `--test-cmd` **does** execute it (that is the point of a test gate). It is
  opt-in. Only use it on solutions you are willing to run; the engine runs the
  command in each worker's workspace directory.
- Worker "isolation" means separate directories + no shared state + diversity
  prompts — **not** OS-level sandboxing.

## License

MIT — see [LICENSE](LICENSE).
