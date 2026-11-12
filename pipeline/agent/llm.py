"""LLM provider layer (docs/phase_3.md Task 1).

- ``LLMProvider.complete_json(system, user, schema, *, model)`` returns an ``LLMResult`` whose ``parsed`` object is
  validated with Pydantic. Invalid output is retried **once** with the validation error appended; a second failure
  is recorded as a failure (``LLMResult.error``), never parsed from free text.
- Back ends: ``OpenAIProvider`` (also serves a local Ollama server through its OpenAI-compatible endpoint),
  ``AzureOpenAIProvider`` (endpoint, deployment and API version from the environment) and ``MockLLMProvider``
  (scripted responses; the only provider tests may use).
- Keys are read from the environment / ``.env`` at call time and never logged, cached or written anywhere.
- ``UsageTracker`` accumulates calls, tokens and cost per model, enforces ``MAX_CALLS_PER_RUN`` /
  ``MAX_COST_USD_PER_RUN`` (``BudgetExceeded``) and writes ``usage.json``. ``RateLimiter`` + ``run_parallel`` give rate
  limiting and bounded concurrency. ``ResponseCache`` stores every answer under ``outputs/cache/agent/``.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pipeline import config


class BudgetExceeded(RuntimeError):
    """Raised by ``UsageTracker.check`` when a run reaches its call or cost limit."""


class LLMResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    parsed: Optional[Any] = None          # validated Pydantic object (None on failure)
    raw_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    model: str = ""
    attempts: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.parsed is not None and self.error is None


# --------------------------------------------------------------------------- #
# Secrets and environment
# --------------------------------------------------------------------------- #
_DOTENV_LOADED = False


def _load_env() -> None:
    """Load ``.env`` once (python-dotenv), without printing or returning anything."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(config.REPO_ROOT / ".env", override=False)
    except Exception:
        pass
    _DOTENV_LOADED = True


def _secret(name: str) -> str:
    _load_env()
    return os.getenv(name, "") or ""


def model_role(role: str) -> str:
    """Resolve a model role (ANSWER_MODEL / STRONG_MODEL / EXTRACTION_MODEL): environment first, then config."""
    _load_env()
    return os.getenv(role, "") or getattr(config, role, "") or ""


def provider_name() -> str:
    _load_env()
    return (os.getenv("LLM_PROVIDER", "") or config.LLM_PROVIDER or "").strip().lower()


# --------------------------------------------------------------------------- #
# Rate limiting, retries, concurrency
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Spaces calls so that at most ``per_minute`` start per minute (thread-safe)."""

    def __init__(self, per_minute: int = config.LLM_RATE_LIMIT_PER_MIN):
        self.interval = 60.0 / per_minute if per_minute and per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> float:
        if not self.interval:
            return 0.0
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)
        return delay


def with_retries(fn: Callable[[], Any], max_retries: int = config.LLM_MAX_RETRIES, backoff: float = 1.0, sleep: Callable[[float], None] = time.sleep,
                 retry_on: tuple[type, ...] = (Exception,)) -> Any:
    """Exponential backoff around a transport call (the Phase 1 ``_call_openai_with_retries`` logic, provider-agnostic)."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except retry_on:
            if attempt == max_retries:
                raise
            sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError("unreachable")


def run_parallel(fn: Callable[[Any], Any], items: Sequence[Any], workers: int = config.LLM_MAX_WORKERS, on_result: Optional[Callable[[Any, Any], None]] = None,
                 stop_on: tuple[type, ...] = (BudgetExceeded,)) -> list[Any]:
    """Apply ``fn`` to ``items`` with bounded concurrency, preserving order. A ``stop_on`` exception stops new work; results
    for the remaining items are the exception instance. ``on_result(item, result)`` runs in the caller's thread."""
    results: list[Any] = [None] * len(items)
    if not items:
        return results
    workers = max(1, min(workers, len(items)))
    stop = threading.Event()

    def _one(i: int):
        if stop.is_set():
            return BudgetExceeded("stopped")
        try:
            return fn(items[i])
        except stop_on as exc:
            stop.set()
            return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {i: pool.submit(_one, i) for i in range(len(items))}
        for i in range(len(items)):
            results[i] = futures[i].result()
            if on_result is not None:
                on_result(items[i], results[i])
    return results


# --------------------------------------------------------------------------- #
# Usage and budget
# --------------------------------------------------------------------------- #
def price_per_token(model: str, prices: Optional[dict] = None) -> tuple[float, float]:
    prices = prices if prices is not None else config.MODEL_PRICES_USD_PER_M
    key = model
    if key not in prices:
        base = next((k for k in prices if key.startswith(k)), None)
        key = base or ""
    p_in, p_out = prices.get(key, config.UNKNOWN_MODEL_PRICE_USD_PER_M) if key else config.UNKNOWN_MODEL_PRICE_USD_PER_M
    return p_in / 1e6, p_out / 1e6


def estimate_cost(n_calls: int, model: str, prompt_tokens: int = config.EST_PROMPT_TOKENS_PER_PAIR, completion_tokens: int = config.EST_COMPLETION_TOKENS_PER_PAIR,
                  prices: Optional[dict] = None) -> dict:
    p_in, p_out = price_per_token(model, prices)
    return {"model": model, "calls": int(n_calls), "prompt_tokens": int(n_calls * prompt_tokens), "completion_tokens": int(n_calls * completion_tokens),
            "cost_usd": round(n_calls * (prompt_tokens * p_in + completion_tokens * p_out), 4)}


class UsageTracker:
    """Accumulates calls / tokens / cost per model and enforces the run limits. Thread-safe. Persisted as usage.json so
    a resumed run continues against the same budget."""

    def __init__(self, max_calls: int = config.MAX_CALLS_PER_RUN, max_cost_usd: float = config.MAX_COST_USD_PER_RUN, prices: Optional[dict] = None, path: Optional[Path] = None):
        self.max_calls, self.max_cost_usd, self.prices = max_calls, max_cost_usd, prices
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.calls = 0
        self.cached_hits = 0
        self.failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.latency_s = 0.0
        self.per_model: dict[str, dict] = {}
        self.stopped_reason: Optional[str] = None
        if self.path and self.path.exists():
            self._load()

    def cost_of(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        p_in, p_out = price_per_token(model, self.prices)
        return prompt_tokens * p_in + completion_tokens * p_out

    def check(self) -> None:
        """Raise ``BudgetExceeded`` when the next live call would exceed a limit."""
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                self.stopped_reason = f"MAX_CALLS_PER_RUN reached ({self.calls} >= {self.max_calls})"
            elif self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
                self.stopped_reason = f"MAX_COST_USD_PER_RUN reached ({self.cost_usd:.4f} >= {self.max_cost_usd})"
            else:
                return
            raise BudgetExceeded(self.stopped_reason)

    def record(self, result: LLMResult) -> None:
        with self._lock:
            m = self.per_model.setdefault(result.model, {"calls": 0, "cached": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            if result.cached:
                self.cached_hits += 1
                m["cached"] += 1
                cached = True
            else:
                cached = False
        if cached:
            if self.path:
                self.write()
            return
        with self._lock:
            m = self.per_model[result.model]
            self.calls += 1
            m["calls"] += 1
            if result.error:
                self.failures += 1
                m["failures"] += 1
            self.prompt_tokens += result.prompt_tokens
            self.completion_tokens += result.completion_tokens
            cost = self.cost_of(result.model, result.prompt_tokens, result.completion_tokens)
            self.cost_usd += cost
            self.latency_s += result.latency_s
            m["prompt_tokens"] += result.prompt_tokens
            m["completion_tokens"] += result.completion_tokens
            m["cost_usd"] = round(m["cost_usd"] + cost, 6)
        if self.path:
            self.write()

    def to_dict(self) -> dict:
        return {"calls": self.calls, "cached_hits": self.cached_hits, "failures": self.failures, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6), "latency_s": round(self.latency_s, 2), "max_calls": self.max_calls, "max_cost_usd": self.max_cost_usd,
                "per_model": self.per_model, "stopped_reason": self.stopped_reason}

    def write(self, path: Optional[Path] = None) -> Path:
        path = Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = self.to_dict()
            tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(path)
        return path

    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.calls = int(d.get("calls", 0))
        self.cached_hits = int(d.get("cached_hits", 0))
        self.failures = int(d.get("failures", 0))
        self.prompt_tokens = int(d.get("prompt_tokens", 0))
        self.completion_tokens = int(d.get("completion_tokens", 0))
        self.cost_usd = float(d.get("cost_usd", 0.0))
        self.latency_s = float(d.get("latency_s", 0.0))
        self.per_model = dict(d.get("per_model", {}))


# --------------------------------------------------------------------------- #
# Response cache
# --------------------------------------------------------------------------- #
def cache_key(**parts: Any) -> str:
    """Deterministic key over named parts, e.g. (doc_sha256, indikit_id, evidence_fingerprint, prompt_version, model)."""
    blob = json.dumps({k: ("" if v is None else str(v)) for k, v in sorted(parts.items())}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """JSON files under ``outputs/cache/agent/<key>.json`` holding the validated response. Never stores secrets."""

    def __init__(self, directory: Union[Path, str, None] = config.AGENT_CACHE_DIR, enabled: bool = True):
        self.dir = Path(directory) if directory else None
        self.enabled = enabled and self.dir is not None
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            self.misses += 1
            return None
        self.hits += 1
        return data

    def put(self, key: str, response: dict, **meta: Any) -> Optional[Path]:
        if not self.enabled:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._path(key)
        payload = {"key": key, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **{k: v for k, v in meta.items() if "key" not in k.lower()}, "response": response}
        tmp = p.with_name(f"{p.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
        return p


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema adjusted for schema-enforced ("strict") modes: every property required, no additional
    properties, ``$defs`` kept."""
    schema = model.model_json_schema()

    def fix(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _extract_json(text: str) -> Optional[Any]:
    """First JSON value in ``text`` (schema-enforced modes return clean JSON; a plain json_object mode may add noise)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    decoder = json.JSONDecoder()
    start = 0
    while start < len(text):
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            start += 1
    return None


class LLMProvider(ABC):
    """Common loop: rate limit -> transport call with backoff -> JSON parse -> Pydantic validation -> one retry with the
    validation error appended -> record usage."""

    name: str = "base"

    def __init__(self, usage: Optional[UsageTracker] = None, rate_limiter: Optional[RateLimiter] = None, max_retries: int = config.LLM_MAX_RETRIES, sleep: Callable[[float], None] = time.sleep):
        self.usage = usage
        self.rate_limiter = rate_limiter or RateLimiter(0)
        self.max_retries = max_retries
        self._sleep = sleep
        self.calls_made = 0

    @abstractmethod
    def _call(self, messages: list[dict], model: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        """Return (raw text, prompt tokens, completion tokens)."""

    def complete_json(self, system: str, user: str, schema: type[BaseModel], *, model: str, validation_retries: int = 1) -> LLMResult:
        if self.usage is not None:
            self.usage.check()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.time()
        attempts = 0
        p_tok = c_tok = 0
        raw = ""
        error: Optional[str] = None
        parsed = None
        while attempts <= validation_retries:
            attempts += 1
            self.rate_limiter.wait()
            try:
                text, pt, ct = with_retries(lambda: self._call(messages, model, schema), self.max_retries, sleep=self._sleep)
            except Exception as exc:
                error = f"transport: {type(exc).__name__}: {str(exc)[:300]}"
                break
            self.calls_made += 1
            p_tok += pt
            c_tok += ct
            raw = text or ""
            obj = _extract_json(raw)
            if obj is None:
                error = "no JSON object in the response"
            else:
                try:
                    parsed = schema.model_validate(obj)
                    error = None
                    break
                except ValidationError as exc:
                    error = f"schema validation failed: {str(exc)[:600]}"
            if attempts <= validation_retries:
                messages = messages + [{"role": "assistant", "content": raw[:4000]}, {"role": "user", "content": f"Your previous answer was invalid: {error}. Respond again with ONLY a JSON object that conforms to the schema."}]
        result = LLMResult(parsed=parsed, raw_text=raw, prompt_tokens=p_tok, completion_tokens=c_tok, latency_s=round(time.time() - t0, 3), cached=False, model=model, attempts=attempts, error=error)
        if self.usage is not None:
            self.usage.record(result)
        return result


class OpenAIProvider(LLMProvider):
    """OpenAI chat completions with the JSON-schema response format. With ``base_url`` it also serves Ollama's
    OpenAI-compatible endpoint (falls back to ``json_object`` when the server rejects ``json_schema``)."""

    name = "openai"

    def __init__(self, api_key_env: str = "OPENAI_API_KEY", base_url: Optional[str] = None, *, timeout: float = config.LLM_TIMEOUT_S, max_completion_tokens: int = config.LLM_MAX_COMPLETION_TOKENS, **kw):
        super().__init__(**kw)
        self.api_key_env, self.base_url, self.timeout, self.max_completion_tokens = api_key_env, base_url, timeout, max_completion_tokens
        self._client = None
        self._schema_mode = "json_schema"
        if base_url:
            self.name = "ollama" if "11434" in base_url else "openai-compatible"

    def _make_client(self):
        from openai import OpenAI

        key = _secret(self.api_key_env)
        if not key and self.base_url:
            key = "local"   # local servers ignore the key but the SDK requires one
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set (environment / .env)")
        return OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout)

    @property
    def client(self):
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _response_format(self, schema: type[BaseModel]) -> dict:
        if self._schema_mode == "json_schema":
            return {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": strict_json_schema(schema), "strict": True}}
        return {"type": "json_object"}

    def _call(self, messages: list[dict], model: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        def _do():
            return self.client.chat.completions.create(model=model, messages=messages, temperature=0, seed=0, response_format=self._response_format(schema), max_completion_tokens=self.max_completion_tokens)

        try:
            resp = _do()
        except Exception as exc:
            msg = str(exc).lower()
            if self._schema_mode == "json_schema" and ("response_format" in msg or "json_schema" in msg or "strict" in msg or "unsupported" in msg):
                self._schema_mode = "json_object"
                resp = _do()
            else:
                raise
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else _approx_tokens(json.dumps(messages))
        ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else _approx_tokens(text)
        return text, pt, ct


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI: endpoint, API version and key from AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_VERSION /
    AZURE_OPENAI_API_KEY. The ``model`` argument is the deployment name (AZURE_OPENAI_DEPLOYMENT can map roles)."""

    name = "azure"

    def __init__(self, **kw):
        super().__init__(api_key_env="AZURE_OPENAI_API_KEY", **kw)

    def _make_client(self):
        from openai import AzureOpenAI

        endpoint = _secret("AZURE_OPENAI_ENDPOINT")
        key = _secret("AZURE_OPENAI_API_KEY")
        version = _secret("AZURE_OPENAI_API_VERSION") or config.AZURE_OPENAI_API_VERSION_DEFAULT
        missing = [n for n, v in (("AZURE_OPENAI_ENDPOINT", endpoint), ("AZURE_OPENAI_API_KEY", key)) if not v]
        if missing:
            raise RuntimeError(f"Azure OpenAI not configured: missing {', '.join(missing)}")
        return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=version, timeout=self.timeout)


ScriptedResponse = Union[dict, str, BaseModel, Exception, Callable[..., Any]]


class MockLLMProvider(LLMProvider):
    """Scripted provider for tests. ``responses`` is a list consumed in order or a callable
    ``(system, user, schema, model) -> dict | str | BaseModel | Exception``. A dict/BaseModel is serialised to JSON
    text (so the normal parse/validate path runs); a str is returned verbatim (use it for invalid JSON); an Exception
    is raised from the transport (use it for failures). Every call is recorded in ``calls``."""

    name = "mock"

    def __init__(self, responses: Union[Sequence[ScriptedResponse], Callable[..., Any], None] = None, default: Optional[ScriptedResponse] = None, **kw):
        kw.setdefault("sleep", lambda s: None)
        kw.setdefault("max_retries", 1)
        super().__init__(**kw)
        self._queue = list(responses) if isinstance(responses, (list, tuple)) else None
        self._fn = responses if callable(responses) else None
        self.default = default
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def _next(self, system: str, user: str, schema: type[BaseModel], model: str) -> ScriptedResponse:
        with self._lock:
            if self._fn is not None:
                return self._fn(system, user, schema, model)
            if self._queue:
                return self._queue.pop(0)
        if self.default is not None:
            return self.default
        raise RuntimeError("MockLLMProvider has no scripted response left")

    def _call(self, messages: list[dict], model: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        with self._lock:
            self.calls.append({"system": system, "user": user, "model": model, "schema": schema.__name__, "n_messages": len(messages)})
        resp = self._next(system, user, schema, model)
        if callable(resp) and not isinstance(resp, BaseModel):
            resp = resp(system, user, schema, model)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, BaseModel):
            text = resp.model_dump_json()
        elif isinstance(resp, dict):
            text = json.dumps(resp, ensure_ascii=False)
        else:
            text = str(resp)
        return text, _approx_tokens(json.dumps(messages)), _approx_tokens(text)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
PROVIDERS = ("openai", "azure", "ollama", "mock")


def get_provider(spec: Optional[str] = None, *, usage: Optional[UsageTracker] = None, rate_limiter: Optional[RateLimiter] = None, allow_mock: bool = False) -> LLMProvider:
    """Build the configured provider. Raises ``RuntimeError`` with a clear message when nothing is configured, so no
    document is ever sent to a service the owner did not name."""
    name = (spec or provider_name() or "").strip().lower()
    if not name:
        raise RuntimeError("No LLM provider configured. Set LLM_PROVIDER (openai | azure | ollama) in pipeline/config.py or the environment, "
                           "plus ANSWER_MODEL / EXTRACTION_MODEL and the budget (MAX_CALLS_PER_RUN, MAX_COST_USD_PER_RUN). Tests use the mock provider.")
    if name not in PROVIDERS:
        raise RuntimeError(f"Unknown LLM provider '{name}'. Use one of {PROVIDERS}.")
    kw = dict(usage=usage, rate_limiter=rate_limiter or RateLimiter(config.LLM_RATE_LIMIT_PER_MIN))
    if name == "openai":
        return OpenAIProvider(**kw)
    if name == "azure":
        return AzureOpenAIProvider(**kw)
    if name == "ollama":
        _load_env()
        return OpenAIProvider(api_key_env="OLLAMA_API_KEY", base_url=os.getenv("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL), **kw)
    if not allow_mock:
        raise RuntimeError("the mock provider is for tests only (pass allow_mock=True for debugging)")
    return MockLLMProvider(default={"indicators": []}, **kw)


def provider_status(spec: Optional[str] = None) -> tuple[bool, str]:
    """(configured, message) without making a call or revealing any secret."""
    name = (spec or provider_name() or "").strip().lower()
    if not name:
        return False, "LLM provider not configured (LLM_PROVIDER empty)"
    if name == "openai":
        return (bool(_secret("OPENAI_API_KEY")), "OpenAI: key present" if _secret("OPENAI_API_KEY") else "OpenAI: OPENAI_API_KEY not set")
    if name == "azure":
        missing = [n for n in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY") if not _secret(n)]
        return (not missing, "Azure OpenAI: configured" if not missing else f"Azure OpenAI: missing {', '.join(missing)}")
    if name == "ollama":
        return True, f"Ollama (local) at {os.getenv('OLLAMA_BASE_URL', config.OLLAMA_BASE_URL)}"
    if name == "mock":
        return True, "mock provider (no live calls)"
    return False, f"unknown provider '{name}'"
