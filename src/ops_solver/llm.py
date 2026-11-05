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

from .config import DEFAULT_PRICE, PRICES
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


class FanOutResults(list):
    """`fan_out`'s result: a plain list aligned to the input thunks (a failure is
    `None` in its slot), plus `.errors`, the exceptions behind those Nones."""

    def __init__(self, items=(), errors: list[BaseException] | None = None) -> None:
        super().__init__(items)
        self.errors: list[BaseException] = list(errors or [])


def describe_errors(errors: list[BaseException]) -> str:
    """One-line cause for a phase that lost calls, e.g. ' First error: TypeError: ...'."""
    if not errors:
        return ""
    first = errors[0]
    return f" First error ({len(errors)} failed): {type(first).__name__}: {str(first)[:300]}"


async def fan_out(thunks: list[Callable[[], Awaitable[Any]]]) -> FanOutResults:
    """Run async thunks; warm the shared cache by awaiting the first, then
    fan the rest out concurrently. Failures become `None` (aligned to input);
    each one is logged with its cause and kept in the result's `.errors`."""
    out = FanOutResults()
    if not thunks:
        return out

    def failed(i: int, exc: BaseException) -> None:
        logger.warning("call %d/%d failed: %s: %s", i + 1, len(thunks), type(exc).__name__, exc)
        out.errors.append(exc)
        out.append(None)

    try:
        out.append(await thunks[0]())
    except Exception as exc:  # noqa: BLE001 - a failed call becomes a None result; the gate scores the rest
        failed(0, exc)
    if len(thunks) > 1:
        rest = await asyncio.gather(*(t() for t in thunks[1:]), return_exceptions=True)
        for i, r in enumerate(rest, start=1):
            if isinstance(r, BaseException):
                failed(i, r)
            else:
                out.append(r)
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
    except Exception:  # noqa: BLE001 - invalid model output means "no structured result", never a crash
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

    def __init__(
        self,
        ledger: TokenLedger | None = None,
        api_max_retries: int = 4,
        *,
        client: Any | None = None,
    ) -> None:
        """`client` is an `anthropic.AsyncAnthropic` (or compatible) instance.
        Pass one to control transport, auth or retries (tests inject a client over
        an offline mock transport); by default one is built from the environment."""
        self.ledger = ledger or TokenLedger()
        if client is None:
            import anthropic  # imported lazily so offline tests need no SDK/key

            client = anthropic.AsyncAnthropic(max_retries=api_max_retries)
        self.client = client

    async def _send(self, kwargs: dict[str, Any]) -> Any:
        """One Messages API call; usage is recorded before anything is parsed."""
        resp = await self.client.messages.create(**kwargs)
        self.ledger.record(kwargs["model"], resp.usage)
        return resp

    @staticmethod
    def _request(
        *, model: str, system: list[dict] | str, user: str, max_tokens: int, thinking: bool
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        return kwargs

    async def complete(
        self,
        *,
        model: str,
        system: list[dict] | str,
        user: str,
        max_tokens: int = 4096,
        thinking: bool = False,
    ) -> str:
        req = self._request(
            model=model, system=system, user=user, max_tokens=max_tokens, thinking=thinking
        )
        return _text_of(await self._send(req))

    async def structured(
        self,
        *,
        model: str,
        system: list[dict] | str,
        user: str,
        schema: type[T],
        max_tokens: int = 1500,
        thinking: bool = False,
    ) -> T | None:
        """Return a validated `schema` instance, or None if the reply doesn't fit it.

        One call with native structured output (`output_config.format`), validated
        locally. Only if the API refuses that request (400, e.g. a schema it can't
        compile) is a single fallback call made with an instruction-guided JSON
        prompt. Auth, rate-limit, server and connection errors propagate to the
        caller, and unusable output never triggers a second call.
        """
        import anthropic

        req = self._request(
            model=model, system=system, user=user, max_tokens=max_tokens, thinking=thinking
        )
        req["output_config"] = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)}
        }
        try:
            resp = await self._send(req)
        except anthropic.BadRequestError as exc:
            logger.warning(
                "structured(): API refused the %s output schema (%s); retrying once with a "
                "JSON instruction",
                schema.__name__,
                exc,
            )
            fallback = self._request(
                model=model,
                system=system,
                user=user + "\n\n" + _json_instruction(schema),
                max_tokens=max_tokens,
                thinking=thinking,
            )
            resp = await self._send(fallback)
        return _coerce(_text_of(resp), schema)

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
