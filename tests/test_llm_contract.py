"""Contract tests: the real `LLM` adapter against the installed anthropic SDK.

The SDK talks to an offline `httpx2.MockTransport`, so every request body the
adapter would send is visible and no network or real key is involved. These
catch SDK/API drift (request keys, call counts, ledger counts, error flow) that
the `MockLLM` tests elsewhere cannot.
"""

from __future__ import annotations

import json

import anthropic
import httpx2
import pytest

from ops_solver.llm import LLM

CODE_REPLY = "```python\ndef f():\n    return 1\n```\nok"


def message(text: str, *, stop_reason: str = "end_turn", model: str = "m") -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def error(status: int) -> httpx2.Response:
    return httpx2.Response(
        status, json={"type": "error", "error": {"type": "test_error", "message": f"HTTP {status}"}}
    )


class FakeAPI:
    """Records each request body; `reply(body)` decides the response."""

    def __init__(self, reply=None) -> None:
        self.requests: list[dict] = []
        self._reply = reply or (lambda body: httpx2.Response(200, json=message(CODE_REPLY)))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        return self._reply(body)

    def client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key="test-dummy-key",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle)),
        )


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


async def test_complete_uses_the_injected_client(api: FakeAPI):
    llm = LLM(client=api.client())
    text = await llm.complete(model="claude-sonnet-4-6", system="s", user="u")
    assert text == CODE_REPLY
    assert len(api.requests) == 1
    assert llm.ledger.calls == 1


# --- a whole run through the real adapter ---------------------------------

MANIFEST_JSON = '{"dependencies": [], "constraints": ["be correct"], "parameters": {}, "risks": [], "notes": "n"}'
VERDICT_JSON = '{"winner": "A", "reason": "cleaner", "a_score": 8, "b_score": 6}'


def pipeline_reply(body: dict) -> httpx2.Response:
    """Answer like the API would for each of the engine's three call kinds."""
    fmt = (body.get("output_config") or {}).get("format")
    if fmt:
        props = fmt.get("schema", {}).get("properties", {})
        return httpx2.Response(200, json=message(VERDICT_JSON if "winner" in props else MANIFEST_JSON))
    return httpx2.Response(200, json=message(CODE_REPLY))


async def test_solve_end_to_end_through_the_real_adapter(tmp_path):
    from ops_solver.config import RunConfig
    from ops_solver.orchestrator import solve

    api = FakeAPI(pipeline_reply)
    cfg = RunConfig(problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs", lint=False)
    report = await solve(cfg, llm=LLM(client=api.client()))

    assert report.winner_id is not None
    # 2 intel + 4 workers + 6 pairs x 2 orders
    assert len(api.requests) == 2 + 4 + 12
    assert report.tokens["calls"] == len(api.requests)


async def test_no_request_carries_sampling_parameters(tmp_path):
    from ops_solver.config import RunConfig
    from ops_solver.orchestrator import solve

    api = FakeAPI(pipeline_reply)
    cfg = RunConfig(problem="p", cwd=tmp_path, runs_dir=tmp_path / "runs", lint=False, max_reloops=0)
    await solve(cfg, llm=LLM(client=api.client()))
    sent_keys = set().union(*(set(b) for b in api.requests))
    assert not sent_keys & {"temperature", "top_p", "top_k"}
