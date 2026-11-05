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
