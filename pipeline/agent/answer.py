"""LLM call plumbing and response caching (paper Section 3.4.2 "LLM answering").

Moved from ``extraction_utils.py`` (second ``LLMClient`` block and the cache
helpers). Contains:

- provider configuration (``configure_llm``, ``_make_client``, ``llm_available``);
- per-chunk cache helpers (``_chunk_hash``, ``cache_response_for_chunk``,
  ``load_cached_response``) used by the legacy extraction cache in
  ``outputs/api_cache``;
- ``_call_openai_with_retries`` and ``_extract_json_from_response``.

Phase 3 adds the disclosure answering loop (``answer_indicator``) which
builds the five-part prompt from ``pipeline.agent.prompts``, calls the model
and caches by (document hash, indicator ID, prompt version, model).

The module-level settings (``LLM_PROVIDER``, ``MODEL_NAME``, ``CACHE_DIR``,
``FORCE_REFRESH`` ...) are the single source of truth; ``extraction_utils``
forwards reads and writes of these names here so older notebook code and the
existing tests keep working.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Optional, Tuple

from openai import AzureOpenAI, OpenAI

from pipeline import config
from pipeline.models import DocumentChunk

# --------------------------------------------------------------------------- #
# Provider settings (mutable at runtime through configure_llm)
# --------------------------------------------------------------------------- #
SUPPORTED_PROVIDERS = ("ollama", "openai", "azure")

LLM_PROVIDER: str = config.DEFAULT_LLM_PROVIDER
MODEL_NAME: str = config.DEFAULT_MODEL_NAME
OLLAMA_BASE_URL: str = config.OLLAMA_BASE_URL
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
LEGACY_CACHE_MODEL: str = config.LEGACY_CACHE_MODEL  # cache entries written before the model became part of the key
FORCE_REFRESH: bool = False
CACHE_DIR: str = str(config.API_CACHE_DIR)


def configure_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
) -> Dict[str, str]:
    """Set the active LLM provider/model for this module (called from the notebook config cell).

    base_url applies to Ollama. azure_endpoint / azure_api_version apply to Azure OpenAI and fall back to the
    AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_VERSION environment variables when not given.
    """
    global LLM_PROVIDER, MODEL_NAME, OLLAMA_BASE_URL, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
    if provider:
        provider = provider.lower().strip()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{provider}'. Use one of {SUPPORTED_PROVIDERS}.")
        LLM_PROVIDER = provider
    if model_name:
        MODEL_NAME = model_name
    if base_url:
        OLLAMA_BASE_URL = base_url
    if azure_endpoint:
        AZURE_OPENAI_ENDPOINT = azure_endpoint
    elif not AZURE_OPENAI_ENDPOINT:
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if azure_api_version:
        AZURE_OPENAI_API_VERSION = azure_api_version
    endpoint = {
        "ollama": OLLAMA_BASE_URL,
        "openai": "https://api.openai.com/v1",
        "azure": AZURE_OPENAI_ENDPOINT or "<AZURE_OPENAI_ENDPOINT not set>",
    }[LLM_PROVIDER]
    return {"provider": LLM_PROVIDER, "model": MODEL_NAME, "base_url": endpoint}


def _make_client() -> OpenAI:
    """Build an OpenAI-SDK client for the active provider. Ollama and Azure speak the same chat-completions protocol."""
    if LLM_PROVIDER == "ollama":
        # Ollama ignores the key but the SDK requires a non-empty value.
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key=os.getenv("OLLAMA_API_KEY", "ollama"))
    if LLM_PROVIDER == "azure":
        endpoint = AZURE_OPENAI_ENDPOINT or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not key:
            raise RuntimeError("LLM_PROVIDER is 'azure' but AZURE_OPENAI_ENDPOINT and/or AZURE_OPENAI_API_KEY are not set.")
        # With AzureOpenAI the `model` argument of chat.completions.create is the deployment name.
        return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=AZURE_OPENAI_API_VERSION)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER is 'openai'.")
    return OpenAI(api_key=key)


def llm_available(timeout: float = 3.0) -> Tuple[bool, str]:
    """Cheap readiness check for the active provider. Returns (ok, human-readable message)."""
    if LLM_PROVIDER == "openai":
        if os.getenv("OPENAI_API_KEY"):
            return True, f"OpenAI configured, model {MODEL_NAME}"
        return False, "OPENAI_API_KEY is not set"
    if LLM_PROVIDER == "azure":
        endpoint = AZURE_OPENAI_ENDPOINT or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        missing = [name for name, val in (("AZURE_OPENAI_ENDPOINT", endpoint), ("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"))) if not val]
        if missing:
            return False, f"Azure OpenAI not configured: missing {', '.join(missing)}"
        if not MODEL_NAME or MODEL_NAME.startswith("<"):
            return False, "Azure OpenAI: MODEL_NAME must be set to your deployment name"
        return True, f"Azure OpenAI configured at {endpoint}, deployment {MODEL_NAME}, api-version {AZURE_OPENAI_API_VERSION}"
    try:
        import httpx

        root = OLLAMA_BASE_URL.rsplit("/v1", 1)[0]
        resp = httpx.get(f"{root}/api/tags", timeout=timeout)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        wanted = MODEL_NAME if ":" in MODEL_NAME else f"{MODEL_NAME}:latest"
        if MODEL_NAME in names or wanted in names:
            return True, f"Ollama reachable at {root}, model {MODEL_NAME} available"
        return False, f"Ollama reachable at {root} but model '{MODEL_NAME}' is not pulled. Run: ollama pull {MODEL_NAME}. Available: {names or 'none'}"
    except Exception as exc:
        return False, f"Ollama server not reachable at {OLLAMA_BASE_URL} ({type(exc).__name__}). Install from https://ollama.com and run: ollama pull {MODEL_NAME}"


# --------------------------------------------------------------------------- #
# Per-chunk response cache (legacy extraction cache in outputs/api_cache)
# --------------------------------------------------------------------------- #
class CacheMiss(LookupError):
    """Raised by LLMClient.extract_indicators_from_chunk(cache_only=True) when no cached response exists."""


def _chunk_hash(chunk: DocumentChunk, prompt_version: str = "v1", model_name: Optional[str] = None) -> str:
    """Cache key for an extraction call. The model is part of the key so that responses from different
    models (e.g. llama3.2 vs gpt-4o-mini) never get mixed. model_name=None gives the legacy key format."""
    h = hashlib.sha256()
    h.update(chunk.content.encode("utf-8"))
    h.update(chunk.doc_id.encode("utf-8"))
    h.update(prompt_version.encode("utf-8"))
    if model_name:
        h.update(b"|model=" + model_name.encode("utf-8"))
    return h.hexdigest()


def cache_response_for_chunk(chunk: DocumentChunk, response: dict, prompt_version: str = "v1", model_name: Optional[str] = None) -> str:
    key = _chunk_hash(chunk, prompt_version, model_name)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"chunk_hash": key, "model": model_name, "prompt_version": prompt_version, "response": response}, fh, ensure_ascii=False, indent=2)
    return path


def load_cached_response(chunk: DocumentChunk, prompt_version: str = "v1", model_name: Optional[str] = None) -> Optional[dict]:
    if FORCE_REFRESH:
        return None
    candidates = [os.path.join(CACHE_DIR, f"{_chunk_hash(chunk, prompt_version, model_name)}.json")]
    if model_name == LEGACY_CACHE_MODEL:
        # Entries written before the model was part of the key were all produced with this model.
        candidates.append(os.path.join(CACHE_DIR, f"{_chunk_hash(chunk, prompt_version, None)}.json"))
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh).get("response")
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Model call with retries, JSON extraction
# --------------------------------------------------------------------------- #
def _extract_json_from_response(text: str) -> Optional[object]:
    """Return the first JSON value found in ``text`` (models sometimes wrap JSON in prose), or None."""
    decoder = json.JSONDecoder()
    start = 0
    while start < len(text):
        try:
            obj, end = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            start += 1
    return None


def _call_openai_with_retries(
    prompt: str,
    model_name: str,
    max_retries: int = 3,
    response_format: Optional[dict] = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> str:
    """Send one user message to the configured chat-completions endpoint with exponential backoff."""
    client = _make_client()
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response_format:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return content or ""
        except Exception:
            if attempt == max_retries:
                raise
            wait = backoff * (2 ** (attempt - 1))
            time.sleep(wait)
    raise RuntimeError(f"LLM request to {LLM_PROVIDER} failed after retries")


# Backwards-compatible alias: the function works for any provider configured via configure_llm().
_call_llm_with_retries = _call_openai_with_retries


def answer_indicator(*args, **kwargs):
    """Phase 3: build the five-part prompt, call the model, validate ``DisclosureAnswer``, cache per
    (document hash, indicator ID, prompt version, model)."""
    raise NotImplementedError("Phase 3")
