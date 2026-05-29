"""Pure helpers in llm.py: cost math, fan_out contract, JSON extraction. Offline."""

import asyncio
from types import SimpleNamespace

from ops_solver.llm import TokenLedger, _coerce, _extract_json, _json_instruction, fan_out
from ops_solver.models import JudgeVerdict


def _usage(**kw):
    base = dict(
        input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    base.update(kw)
    return SimpleNamespace(**base)


# --- TokenLedger cost math ---

def test_cost_usd_blends_all_token_tiers():
    led = TokenLedger()
    led.record(
        "claude-sonnet-4-6",
        _usage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_input_tokens=1_000_000,
            cache_read_input_tokens=1_000_000,
        ),
    )
    # sonnet in=$3 out=$15 per 1M: 3 + 3*1.25 + 3*0.1 + 15 = 22.05
    assert round(led.cost_usd(), 2) == 22.05
    assert led.calls == 1


def test_cost_usd_unknown_model_uses_default_price():
    led = TokenLedger()
    led.record("some-future-model", _usage(input_tokens=1_000_000))
    assert led.cost_usd() == 3.0  # DEFAULT_PRICE input rate


def test_record_tolerates_missing_usage_fields():
    led = TokenLedger()
    led.record("claude-haiku-4-5", SimpleNamespace())  # no attributes at all
    assert led.cost_usd() == 0.0


# --- fan_out contract (order preserved, exceptions -> None) ---

async def _ret(v):
    return v


async def _boom():
    raise ValueError("x")


def test_fan_out_preserves_order():
    assert asyncio.run(fan_out([lambda i=i: _ret(i) for i in (1, 2, 3)])) == [1, 2, 3]


def test_fan_out_converts_exceptions_to_none():
    assert asyncio.run(fan_out([lambda: _ret("a"), lambda: _boom(), lambda: _ret("c")])) == [
        "a",
        None,
        "c",
    ]


def test_fan_out_first_failure_is_none():
    assert asyncio.run(fan_out([lambda: _boom(), lambda: _ret("b")])) == [None, "b"]


def test_fan_out_empty():
    assert asyncio.run(fan_out([])) == []


# --- JSON extraction / coercion ---

def test_extract_json_fenced():
    assert _extract_json('text\n```json\n{"a": 1}\n```\nmore') == {"a": 1}


def test_extract_json_fenced_no_newline():
    assert _extract_json('```json{"a": 1}```') == {"a": 1}


def test_extract_json_bare_and_skips_malformed():
    assert _extract_json('noise {bad json} then {"ok": true} tail') == {"ok": True}


def test_extract_json_garbage_returns_none():
    assert _extract_json("no json here") is None
    assert _extract_json("") is None


def test_coerce_valid_and_invalid():
    obj = _coerce('{"winner":"A","reason":"r","a_score":7,"b_score":3}', JudgeVerdict)
    assert obj is not None and obj.winner == "A"
    assert _coerce('{"winner":"Z"}', JudgeVerdict) is None  # schema violation -> None, not raise


def test_json_instruction_lists_fields():
    s = _json_instruction(JudgeVerdict)
    assert "winner" in s and "a_score" in s
