# Architecture

## Data flow

```
problem ──► env_probe.probe(cwd)                     # deterministic local facts
                │
                ▼
        intelligence.gather_context                  # Phase I — Crow x N (parallel)
                │   (N structured ManifestDrafts → merge)
                ▼
          ContextManifest ──────────────┐
                │                        │ (cached prefix, shared by all workers)
                ▼                        │
        execution.run_workers            │            # Phase II — Finch x M (parallel)
                │   M WorkerArtifacts (isolated dirs)
                ▼
        qa_consensus.evaluate (per artifact)          # Phase III, Layer 1
                │   EvalResult[]  → consensus(threshold)
                │
        ratio < threshold? ──yes──► mutate prompts, re-loop Phase II (≤ max_reloops)
                │ no / exhausted
                ▼
        elo.run_tournament  ──► pairwise Opus judge (cache-warmed fan-out)
                │   EloEngine: expected_score / update_pair
                │   final = blend·norm(elo) + (1−blend)·objective
                ▼
            Leaderboard ──► ship WINNER → runs/<id>/WINNER/ (+ --out)
                            report.json
```

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | `RunConfig`, model tiers, price table, sampling rules |
| `models.py` | Pydantic types: `ContextManifest`, `WorkerArtifact`, `EvalResult`, `JudgeVerdict`, `MatchResult`, `Leaderboard`, `RunReport` |
| `llm.py` | Async Anthropic adapter, prompt caching, structured output, `fan_out` cache-warming, `TokenLedger` cost accounting |
| `env_probe.py` | Deterministic local-environment snapshot |
| `intelligence.py` | Crow team → merged `ContextManifest` |
| `execution.py` | Finch team → isolated `WorkerArtifact[]`; solution extraction |
| `qa_consensus.py` | Layer 1: syntax/lint/test evals + the 50% stability gate + objective scoring |
| `elo.py` | Layer 2: pure ELO math + async pairwise-judge tournament |
| `orchestrator.py` | Phase I→II→III state machine, re-loop, ship winner, run report |
| `cli.py` | `ops-solve` entry point + rich rendering |

## Formulas

**Consensus gate.** `stable_ratio = |{e : e.stable}| / |evals|`. Re-loop Phase II
while `stable_ratio < stability_threshold` (default `0.5`) and attempts remain.
A run is *stable* iff it parses and (when a `--test-cmd` ran) passes.

**ELO.** Expected score of A vs B:

```
E_A = 1 / (1 + 10^((R_B − R_A) / 400))
R_A' = R_A + K·(S_A − E_A)        # S_A ∈ {1, 0.5, 0}; K = 32; start = 1000
```

Judging is fanned out concurrently (cache-warmed: fire one, await it, then the
rest), and the outcomes are folded into ratings in a fixed, deterministic order.

**Final score** blends normalized ELO with the objective signal:

```
final = blend_elo · norm(elo) + (1 − blend_elo) · objective       # blend_elo = 0.6
objective = 0.5·tests + 0.3·lint_score + 0.2·syntax
```

## Caching strategy

Caching is a prefix match and is **model-scoped**, so:
- All Execution workers use **one** model; the Context Manifest is the last
  cached `system` block. Worker 2..M read the cache worker 1 wrote.
- Per-worker volatile content (strategy nudge) goes in the **user** message,
  after the cached prefix.
- The judge (Opus) caches the shared problem context; each pairwise call appends
  only the two specific solutions.

## Prior art credited

Reuses patterns/math from: Self-Consistency (Wang et al. 2022), AlphaCode
(DeepMind 2022), CodeT (Chen et al. 2022), LLM-as-judge / MT-Bench (Zheng et al.
2023), Chatbot Arena / Bradley-Terry (LMSYS 2024), Mixture-of-Agents (2024). Tool
analogs: RAGElo, CodeElo, llm-council, AlphaEvolve. See the README for links.
