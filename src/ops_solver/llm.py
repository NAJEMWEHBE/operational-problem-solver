"""Anthropic Messages-SDK adapter: tiered models, prompt caching, structured
output, concurrency helpers, and token/cost accounting.

Caching strategy (prefix match):
  - Callers pass `system` as a list of text blocks with `cache_control` on the
    LAST block. All workers share ONE model + the same system prefix, so the 2nd
    worker onward reads the cache the 1st wrote. Per-call volatile content
    (strategy hint, the specific pair of solutions) goes in the user message,
    AFTER the cached prefix.
  - A cache entry is only readable once the first response begins. `fan_out`
    therefore fires the first call, awaits it, then fans out the rest.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import DEFAULT_PRICE, PRICES, supports_sampling
from .models import JudgeVerdict

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("ops_solver.llm")


# --- Token / cost accounting ---------------------------------------------


class TokenLedger:
    """Accumulates per-model token usage and estimates USD cost."""

    def __init__(self) -> None:
        self.calls: int = 0
        self.by_model: dict[str, dict[str, int]] = {}

    def record(self, model: str, usage: Any) -> None:
        self.calls += 1
        slot = self.by_model.setdefault(
            model,
            {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0},
        )
        slot["input"] += int(getattr(usage, "input_tokens", 0) or 0)
        slot["output"] += int(getattr(usage, "output_tokens", 0) or 0)
        slot["cache_write"] += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        slot["cache_read"] += int(getattr(usage, "cache_read_input_tokens", 0) or 0)

    def cost_usd(self) -> float:
        total = 0.0
        for model, t in self.by_model.items():
            pin, pout = PRICES.get(model, DEFAULT_PRICE)
            total += (
                t["input"] * pin
                + t["cache_write"] * pin * 1.25
                + t["cache_read"] * pin * 0.1
                + t["output"] * pout
            ) / 1_000_000.0
        return total

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "cost_usd": round(self.cost_usd(), 4),
            "by_model": self.by_model,
        }


# --- Concurrency ----------------------------------------------------------


async def fan_out(thunks: list[Callable[[], Awaitable[Any]]]) -> list[Any]:
    """Run async thunks; warm the shared cache by awaiting the first, then
    fan the rest out concurrently. Failures become `None` (aligned to input)."""
    if not thunks:
        return []
    out: list[Any] = []
    try:
        out.append(await thunks[0]())
    except Exception:
        out.append(None)
    if len(thunks) > 1:
        rest = await asyncio.gather(*(t() for t in thunks[1:]), return_exceptions=True)
        out.extend(None if isinstance(r, Exception) else r for r in rest)
    return out


# --- Helpers --------------------------------------------------------------


def _text_of(resp: Any) -> str:
    parts = [b.text for b in getattr(resp, "content", []) if getattr(b, "type", "") == "text"]
    return "".join(parts)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of model text (fenced or bare)."""
    if not text:
        return None
    fence = text.find("```")
    if fence != -1:
        end = text.find("```", fence + 3)
        if end != -1:
            block = text[fence + 3 : end]
            if "\n" in block:
                block = block.split("\n", 1)[1]
            else:
                # ```json{...} with no newline: strip the leading language tag
                block = block.lstrip(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-_"
                )
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
    # raw_decode is string-literal-aware: braces inside JSON string values
    # (e.g. a judge's free-form `reason`) don't derail the scan.
    dec = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = dec.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def _coerce(text: str, schema: type[T]) -> T | None:
    data = _extract_json(text)
    if data is None:
        return None
    try:
        return schema.model_validate(data)
    except Exception:
        return None


def _json_instruction(schema: type[BaseModel]) -> str:
    fields = ", ".join(schema.model_fields.keys())
    return (
        f"Respond with ONLY a single JSON object (no prose, no code fence) "
        f"containing these keys: {fields}."
    )


# --- The adapter ----------------------------------------------------------


class LLM:
    """Thin async wrapper over the Anthropic Messages API."""

    def __init__(self, ledger: TokenLedger | None = None, api_max_retries: int = 4) -> None:
        import anthropic  # imported lazily so offline tests need no SDK/key

        self.ledger = ledger or TokenLedger()
        self.client = anthropic.AsyncAnthropic(max_retries=api_max_retries)

    async def _complete_raw(
        self,
        *,
        model: str,
        system: list[dict] | str,
        user: str,
        max_tokens: int,
        temperature: float | None = None,
        thinking: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        elif temperature is not None and supports_sampling(model):
            kwargs["temperature"] = temperature
        resp = await self.client.messages.create(**kwargs)
        self.ledger.record(model, resp.usage)
        return _text_of(resp)

    async def complete(
        self,
        *,
        model: str,
        system: list[dict] | str,
        user: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        thinking: bool = False,
    ) -> str:
        return await self._complete_raw(
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
        )

    async def structured(
        self,
        *,
        model: str,
        system: list[dict] | str,
        user: str,
        schema: type[T],
        max_tokens: int = 1500,
        temperature: float | None = None,
        thinking: bool = False,
    ) -> T | None:
        """Return a validated `schema` instance.

        Primary path is the native structured-output API (one call). Only if that
        call *raises* (e.g. an SDK build without `messages.parse`) do we make a
        single fallback call with instruction-guided JSON extraction. We never make
        a second API call just because the first returned unusable output, and we
        never record usage twice for one call.
        """
        messages = [{"role": "user", "content": user}]
        kw: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if thinking:
            kw["thinking"] = {"type": "adaptive"}
        elif temperature is not None and supports_sampling(model):
            kw["temperature"] = temperature  # keep the per-worker diversity gradient

        try:
            resp = await self.client.messages.parse(output_format=schema, **kw)
            self.ledger.record(model, resp.usage)
            parsed = getattr(resp, "parsed_output", None)
            if parsed is not None:
                return parsed
            return _coerce(_text_of(resp), schema)
        except Exception as exc:
            logger.warning("structured(): native parse failed, falling back to JSON: %s", exc)

        text = await self._complete_raw(
            model=model,
            system=system,
            user=user + "\n\n" + _json_instruction(schema),
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
        )
        return _coerce(text, schema)

    async def judge(
        self,
        *,
        model: str,
        system: list[dict] | str,
        user: str,
        max_tokens: int = 1500,
    ) -> JudgeVerdict | None:
        verdict = await self.structured(
            model=model,
            system=system,
            user=user,
            schema=JudgeVerdict,
            max_tokens=max_tokens,
            thinking=True,
        )
        if verdict is not None:
            verdict.a_score = max(0, min(10, verdict.a_score))
            verdict.b_score = max(0, min(10, verdict.b_score))
        return verdict
