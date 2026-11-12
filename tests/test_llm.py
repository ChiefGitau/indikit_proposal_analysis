"""Provider layer: limits, retries, caching, usage tracking, resumability (docs/phase_3.md Task 1). Mock provider only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from pipeline import config
from pipeline.agent import llm
from pipeline.agent.llm import BudgetExceeded, LLMResult, MockLLMProvider, RateLimiter, ResponseCache, UsageTracker, cache_key, estimate_cost, get_provider, price_per_token, run_parallel, strict_json_schema


class Toy(BaseModel):
    answer: str
    n: int = 0


def test_mock_provider_parses_and_records_usage():
    usage = UsageTracker(max_calls=10, max_cost_usd=1.0, prices={"m": (1.0, 2.0)})
    p = MockLLMProvider([{"answer": "yes", "n": 1}], usage=usage)
    r = p.complete_json("sys", "user", Toy, model="m")
    assert r.ok and r.parsed.answer == "yes" and r.attempts == 1 and not r.cached and r.prompt_tokens > 0
    assert usage.calls == 1 and usage.cost_usd > 0 and usage.per_model["m"]["calls"] == 1
    assert p.calls[0]["schema"] == "Toy" and p.calls[0]["system"] == "sys"


def test_invalid_json_triggers_exactly_one_retry_then_failure():
    p = MockLLMProvider(["not json at all", '{"answer": 5, "n": "x"}'])
    r = p.complete_json("s", "u", Toy, model="m")
    assert not r.ok and r.attempts == 2 and "schema validation failed" in (r.error or "") and r.parsed is None
    # the retry message carries the validation error
    assert p.calls[1]["n_messages"] == 4 and "invalid" in p.calls[1]["user"]
    # invalid then valid -> success on the second attempt
    p2 = MockLLMProvider(["{oops", {"answer": "ok"}])
    r2 = p2.complete_json("s", "u", Toy, model="m")
    assert r2.ok and r2.attempts == 2


def test_transport_error_is_retried_with_backoff_then_recorded():
    sleeps = []
    p = MockLLMProvider([RuntimeError("boom"), {"answer": "fine"}], max_retries=3, sleep=sleeps.append)
    r = p.complete_json("s", "u", Toy, model="m")
    assert r.ok and sleeps == [1.0]
    p2 = MockLLMProvider([RuntimeError("down")] * 3, max_retries=3, sleep=sleeps.append)
    r2 = p2.complete_json("s", "u", Toy, model="m")
    assert not r2.ok and r2.error.startswith("transport")


def test_budget_limits_stop_calls():
    usage = UsageTracker(max_calls=2, max_cost_usd=100.0, prices={"m": (0, 0)})
    p = MockLLMProvider(default={"answer": "a"}, usage=usage)
    p.complete_json("s", "u", Toy, model="m")
    p.complete_json("s", "u", Toy, model="m")
    with pytest.raises(BudgetExceeded, match="MAX_CALLS_PER_RUN"):
        p.complete_json("s", "u", Toy, model="m")
    cost_usage = UsageTracker(max_calls=100, max_cost_usd=0.000001, prices={"m": (1000.0, 1000.0)})
    p2 = MockLLMProvider(default={"answer": "a"}, usage=cost_usage)
    p2.complete_json("s", "u", Toy, model="m")
    with pytest.raises(BudgetExceeded, match="MAX_COST_USD_PER_RUN"):
        p2.complete_json("s", "u", Toy, model="m")
    assert cost_usage.stopped_reason


def test_usage_tracker_persists_and_resumes(tmp_path: Path):
    path = tmp_path / "usage.json"
    u = UsageTracker(max_calls=5, max_cost_usd=1.0, prices={"m": (1.0, 1.0)}, path=path)
    u.record(LLMResult(parsed=1, prompt_tokens=1000, completion_tokens=500, model="m"))
    u.record(LLMResult(parsed=None, cached=True, model="m"))
    data = json.loads(path.read_text())
    assert data["calls"] == 1 and data["cached_hits"] == 1 and data["prompt_tokens"] == 1000
    u2 = UsageTracker(max_calls=5, max_cost_usd=1.0, prices={"m": (1.0, 1.0)}, path=path)
    assert u2.calls == 1 and u2.cost_usd == pytest.approx(0.0015)
    assert "key" not in json.dumps(data).lower() or "api" not in json.dumps(data).lower()


def test_response_cache_roundtrip_and_key(tmp_path: Path):
    c = ResponseCache(tmp_path / "agent")
    k = cache_key(doc_sha256="abc", indikit_id="5138", evidence_fingerprint="ef", prompt_version="v2", model="m")
    assert k == cache_key(model="m", prompt_version="v2", evidence_fingerprint="ef", indikit_id="5138", doc_sha256="abc")
    assert k != cache_key(doc_sha256="abc", indikit_id="5138", evidence_fingerprint="ef", prompt_version="v3", model="m")
    assert c.get(k) is None and c.misses == 1
    c.put(k, {"a": 1}, doc_id="d")
    assert c.get(k)["response"] == {"a": 1} and c.hits == 1
    assert ResponseCache(None).get(k) is None


def test_prices_estimate_and_strict_schema():
    assert price_per_token("gpt-4o-mini") == (0.15e-6, 0.60e-6)
    assert price_per_token("unknown-model") == tuple(x / 1e6 for x in config.UNKNOWN_MODEL_PRICE_USD_PER_M)
    est = estimate_cost(100, "gpt-4o-mini", 1000, 100)
    assert est["calls"] == 100 and est["cost_usd"] == pytest.approx(100 * (1000 * 0.15e-6 + 100 * 0.6e-6))
    s = strict_json_schema(Toy)
    assert s["additionalProperties"] is False and s["required"] == ["answer", "n"]


def test_rate_limiter_and_run_parallel_stop():
    rl = RateLimiter(per_minute=6000)
    assert rl.wait() == 0.0 and rl.wait() >= 0.0
    seen = []

    def fn(x):
        if x == 2:
            raise BudgetExceeded("limit")
        seen.append(x)
        return x * 2

    out = run_parallel(fn, [0, 1, 2, 3], workers=1)
    assert out[0] == 0 and out[1] == 2 and isinstance(out[2], BudgetExceeded) and isinstance(out[3], BudgetExceeded)


def test_get_provider_refuses_when_unconfigured(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "")
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        get_provider()
    with pytest.raises(RuntimeError, match="mock provider is for tests"):
        get_provider("mock")
    assert isinstance(get_provider("mock", allow_mock=True), MockLLMProvider)
    ok, msg = llm.provider_status("")
    assert not ok and "not configured" in msg
    # real providers are constructed lazily and never contact the network here
    assert get_provider("openai").name == "openai" and get_provider("azure").name == "azure" and get_provider("ollama").name == "ollama"
    with pytest.raises(RuntimeError):
        get_provider("nope")
