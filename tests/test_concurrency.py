"""The in-flight API call cap (rebuilds PR #6 on the current code).

The cap lives in `LLM`, so it counts real API calls, not fan_out thunks: a
tournament's judge_one sends two calls at once, which a thunk cap would let
double past the limit."""

import asyncio
from types import SimpleNamespace

import anthropic
import pytest

from ops_solver.config import RunConfig
from ops_solver.llm import LLM
from ops_solver.orchestrator import solve

MANIFEST_JSON = '{"dependencies": [], "constraints": [], "parameters": {}, "risks": [], "notes": ""}'
VERDICT_JSON = '{"winner": "A", "reason": "r", "a_score": 7, "b_score": 5}'


class SlowClient:
    """Duck-typed AsyncAnthropic: each call takes a moment; tracks peak in-flight."""

    def __init__(self, **_ignored) -> None:
        self.messages = self
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def create(self, **kw):
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        fmt = (kw.get("output_config") or {}).get("format")
        if fmt:
            text = VERDICT_JSON if "winner" in fmt["schema"].get("properties", {}) else MANIFEST_JSON
        else:
            text = "```python\ndef f():\n    return 1\n```"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(),
            stop_reason="end_turn",
        )


async def test_llm_never_has_more_calls_in_flight_than_its_cap():
    client = SlowClient()
    llm = LLM(client=client, max_concurrency=3)
    out = await asyncio.gather(
        *(llm.complete(model="m", system="s", user=str(i)) for i in range(20))
    )
    assert len(out) == 20 and all(out)
    assert client.peak == 3


async def test_solve_applies_the_run_config_cap_to_every_phase(tmp_path, monkeypatch):
    client = SlowClient()
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **kw: client)
    cfg = RunConfig(
        problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs", lint=False, max_concurrency=2
    )
    report = await solve(cfg)
    assert report.winner_id is not None
    assert client.calls == 2 + 4 + 12
    assert client.peak == 2


def test_cap_must_be_positive():
    with pytest.raises(ValueError):
        LLM(client=SlowClient(), max_concurrency=0)
