"""Central orchestrator: Phase I (Intelligence) -> Phase II (Execution) ->
Phase III (QA consensus + ELO tournament) -> ship the winner.

Implements the consensus re-loop: if fewer than `stability_threshold` of the
workers produce stable output, the prompt variables are mutated (temperature
bumped, failure feedback injected) and Phase II re-runs, up to `max_reloops`.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from . import elo, env_probe, execution, intelligence, qa_consensus
from .config import RunConfig
from .llm import LLM, TokenLedger
from .models import RunReport

logger = logging.getLogger("ops_solver.orchestrator")


async def solve(cfg: RunConfig, llm: LLM | None = None) -> RunReport:
    """Run one full solve. Pass `llm` to inject a (mock) adapter in tests."""
    if llm is None:
        llm = LLM(ledger=TokenLedger(), api_max_retries=cfg.api_max_retries)
    ledger = llm.ledger

    run_id = time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = cfg.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Phase I — Intelligence
    probe = env_probe.probe(cfg.cwd)
    manifest = await intelligence.gather_context(llm, cfg, cfg.problem, probe)

    # Phase II + Layer 1 with consensus re-loop
    artifacts = []
    evals = []
    stable_ids: list[str] = []
    ratio = 0.0
    met = False
    feedback = ""
    attempts = 0

    for attempt in range(cfg.max_reloops + 1):
        attempts = attempt + 1
        artifacts = await execution.run_workers(
            llm, cfg, manifest, run_dir, attempt=attempt, feedback=feedback
        )
        evals = [qa_consensus.evaluate(a, cfg) for a in artifacts]
        stable_ids, ratio, met = qa_consensus.consensus(evals, cfg.stability_threshold)
        if met or attempt == cfg.max_reloops:
            break
        fails = [e for e in evals if not e.stable]
        feedback = "; ".join(
            f"{e.worker_id}: {'syntax error' if not e.syntax_ok else 'tests failed'}"
            for e in fails
        )[:400]

    warnings: list[str] = []
    if len(artifacts) < cfg.exec_workers:
        warnings.append(
            f"{cfg.exec_workers - len(artifacts)}/{cfg.exec_workers} execution workers "
            f"produced no usable solution"
        )
        logger.warning(warnings[-1])
    if not artifacts:
        raise RuntimeError(
            f"Execution phase produced 0 solutions out of {cfg.exec_workers} workers "
            f"({cfg.worker_model}) after {attempts} attempt(s). All workers failed or "
            f"returned empty output - check ANTHROPIC_API_KEY and rate limits."
        )

    # Phase III — ELO tournament over the stable survivors (or all, if none stable)
    competing = [a for a in artifacts if a.worker_id in stable_ids] or artifacts
    eval_by_id = {e.worker_id: e for e in evals}
    objective = {
        a.worker_id: qa_consensus.objective_score(eval_by_id[a.worker_id])
        for a in competing
        if a.worker_id in eval_by_id
    }
    leaderboard = await elo.run_tournament(llm, cfg, competing, manifest, objective)
    if len(competing) >= 2 and not leaderboard.matches:
        warnings.append(
            "0 judge matches completed - every pairwise judging call failed; ranking "
            "fell back to objective score / sort order, not head-to-head ELO."
        )
        logger.warning(warnings[-1])

    # Ship the winner
    winner_id = leaderboard.winner
    winner_path = None
    if winner_id:
        win_art = next((a for a in competing if a.worker_id == winner_id), None)
        if win_art and win_art.workspace and Path(win_art.workspace).exists():
            dest = run_dir / "WINNER"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(win_art.workspace, dest)
            winner_path = str(dest)
            if cfg.out_path is not None:
                out_dest = cfg.out_path.resolve()
                if out_dest.exists() and not cfg.force:
                    warnings.append(
                        f"--out target exists, not overwritten (use --force): {out_dest}"
                    )
                    logger.warning(warnings[-1])
                else:
                    out_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(Path(win_art.workspace) / win_art.filename, out_dest)

    report = RunReport(
        run_id=run_id,
        problem=cfg.problem,
        manifest=manifest,
        artifacts=artifacts,
        evals=evals,
        leaderboard=leaderboard,
        winner_id=winner_id,
        winner_path=winner_path,
        attempts=attempts,
        consensus_ratio=round(ratio, 3),
        consensus_met=met,
        cost_usd=round(ledger.cost_usd(), 4),
        tokens=ledger.summary(),
        warnings=warnings,
    )
    (run_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
