"""Run configuration, model tiers, and pricing.

Model IDs and per-1M-token prices come from the Anthropic model catalogue
(Opus 4.8 / Sonnet 4.6 / Haiku 4.5). Use the exact alias strings only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Model tiers ----------------------------------------------------------
JUDGE_MODEL = "claude-opus-4-8"        # tournament judge: strongest reasoning
WORKER_MODEL = "claude-sonnet-4-6"     # execution workers: balanced quality/cost
INTEL_MODEL = "claude-sonnet-4-6"      # intelligence team
CHEAP_WORKER_MODEL = "claude-haiku-4-5"  # --cheap: fastest, lowest cost

# Opus 4.7/4.8 reject sampling params (temperature/top_p/top_k) -> 400.
NO_SAMPLING_MODELS = ("claude-opus-4-8", "claude-opus-4-7")

# USD per 1,000,000 tokens: (input, output). Cache read ~= 0.1x input,
# cache write ~= 1.25x input (5-minute TTL).
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
DEFAULT_PRICE = (3.0, 15.0)


def supports_sampling(model: str) -> bool:
    """Whether `temperature` may be passed for this model."""
    return not any(model.startswith(m) for m in NO_SAMPLING_MODELS)


@dataclass
class RunConfig:
    """Everything one solve run needs. Sized for the spec default fan-out."""

    problem: str
    cwd: Path = field(default_factory=Path.cwd)
    runs_dir: Path | None = None
    language: str = "auto"

    # Team sizes (spec default: 2 intel + 4 exec + 1 judge).
    intel_workers: int = 2
    exec_workers: int = 4
    elo_rounds: int = 1            # round-robin passes in the tournament

    # Consensus gate
    stability_threshold: float = 0.5  # spec: re-loop if < 50% stable
    max_reloops: int = 1              # extra Phase-II attempts after the first

    # Models
    worker_model: str = WORKER_MODEL
    intel_model: str = INTEL_MODEL
    judge_model: str = JUDGE_MODEL

    # Generation knobs
    worker_max_tokens: int = 4096
    intel_max_tokens: int = 1500
    judge_max_tokens: int = 1500
    base_temperature: float = 0.6     # worker diversity nudge (non-Opus only)
    temperature_step: float = 0.12

    # ELO
    k_factor: float = 32.0
    elo_start: float = 1000.0
    blend_elo: float = 0.6            # final = blend*elo_norm + (1-blend)*objective

    # QA Layer 1
    test_cmd: str | None = None       # set => runs generated code (opt-in, see README)
    lint: bool = True

    # Output
    out_path: Path | None = None

    # SDK-level API retries (429 / 5xx auto-retried by the client)
    api_max_retries: int = 4

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd)
        if self.runs_dir is None:
            self.runs_dir = self.cwd / "runs"
        self.runs_dir = Path(self.runs_dir)
        if self.out_path is not None:
            self.out_path = Path(self.out_path)
