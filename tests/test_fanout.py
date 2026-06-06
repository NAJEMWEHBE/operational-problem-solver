"""fan_out concurrency-cap regression (robustness fix).

Without the cap, a large judge fan-out fired as one synchronized burst, exhausting
SDK retries under a rate limit and silently dropping matches -> biased ELO ranking.
"""
import asyncio

from ops_solver.llm import fan_out


def test_fan_out_caps_concurrency():
    state = {"cur": 0, "max": 0}

    def make(i):
        async def thunk():
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.01)
            state["cur"] -= 1
            return i
        return thunk

    thunks = [make(i) for i in range(30)]
    out = asyncio.run(fan_out(thunks, limit=4))
    assert out == list(range(30))   # order preserved, all succeed
    assert state["max"] <= 4        # never exceeded the cap


def test_fan_out_failures_become_none_aligned():
    def ok(v):
        async def t():
            return v
        return t

    def boom():
        async def t():
            raise RuntimeError("rate limited")
        return t

    out = asyncio.run(fan_out([ok(1), boom(), ok(3)], limit=2))
    assert out == [1, None, 3]      # failure -> None, positions aligned


def test_fan_out_empty():
    assert asyncio.run(fan_out([])) == []
