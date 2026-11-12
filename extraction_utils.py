"""
extraction_utils.py

Core utilities for proposal indicator extraction per plan/agent_task_proposal_indicator_extraction.md

This module focuses on deterministic, unit-testable functions: chunk scoring, deduplication, classification thresholds, simple parsing helpers, and caching helpers for API responses.

Note: The live LLM extraction client code is provided as a safe, importable skeleton that expects an OpenAI API key to be present when used. Tests must mock any network calls.
"""
from __future__ import annotations

from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - import guard for environments missing rapidfuzz
    fuzz = None

# Configuration constants (kept here so notebook can import and tune)
# LLM provider:
#   "ollama" runs a local model through Ollama's OpenAI-compatible endpoint
#   "openai" uses the OpenAI API (OPENAI_API_KEY)
#   "azure"  uses an Azure OpenAI deployment (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION);
#            for Azure, MODEL_NAME is the *deployment name*, not the base model name.
LLM_PROVIDER = "ollama"
MODEL_NAME = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
# LLM_PROVIDER = "openai"
# MODEL_NAME = "gpt-4o-mini"
# LLM_PROVIDER = "azure"
# MODEL_NAME = "<your-deployment-name>"
LEGACY_CACHE_MODEL = "gpt-4o-mini"  # extraction cache entries written before the model became part of the cache key
FORCE_REFRESH = False
CACHE_DIR = "outputs/api_cache"
THRESHOLDS = {
    "exact_embedding": 0.95,
    "exact_lexical": 90,
    "close_embedding": 0.80,
    "related_embedding": 0.65,
}

INDICATOR_KEYWORDS = [
    "logframe",
    "logical framework",
    "results framework",
    "indicator",
    "means of verification",
    "mov",
    "baseline",
    "target",
    "output",
    "outcome",
    "impact",
    "m&e",
    "monitoring",
]

TABLE_HEADER_PATTERNS = [
    r"indicator",
    r"baseline",
    r"target",
    r"mov",
    r"means of verification",
]

os.makedirs(CACHE_DIR, exist_ok=True)


SUPPORTED_PROVIDERS = ("ollama", "openai", "azure")


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


class DocumentChunk(BaseModel):
    doc_id: str
    source_file: str
    chunk_type: str  # "table" | "text"
    page_or_section: Optional[str]
    content: str
    char_count: int


INDICATOR_LEVELS = ("impact", "outcome", "output", "activity", "unclear")
CONFIDENCE_LEVELS = ("high", "medium", "low")


def normalize_indicator_level(value) -> str:
    """Map free-text level values from the LLM onto the plan's enum; anything unrecognised becomes 'unclear'."""
    if value is None:
        return "unclear"
    text = str(value).strip().lower()
    if text in INDICATOR_LEVELS:
        return text
    # e.g. "Intermediate outcome", "Output 2", "Impact indicator", "Result 2 - Indicator 1"
    for lvl in ("impact", "outcome", "output", "activity"):
        if re.search(rf"\b{lvl}s?\b", text):
            return lvl
    if re.search(r"\b(goal|objective)\b", text):
        return "impact"
    return "unclear"


def normalize_confidence(value) -> str:
    if value is None:
        return "low"
    text = str(value).strip().lower()
    return text if text in CONFIDENCE_LEVELS else "low"


class Indicator(BaseModel):
    indicator_text: str
    level: str = "unclear"  # impact | outcome | output | activity | unclear
    baseline: Optional[str] = None
    target: Optional[str] = None
    means_of_verification: Optional[str] = None
    sector: Optional[str] = None
    disaggregation: Optional[str] = None
    unit_or_type: Optional[str] = None
    source_confidence: str = "low"  # high | medium | low
    doc_id: Optional[str] = None
    chunk_ref: Optional[str] = None

    @field_validator("level", mode="before")
    @classmethod
    def _coerce_level(cls, v):
        return normalize_indicator_level(v)

    @field_validator("source_confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v):
        return normalize_confidence(v)

    @field_validator("baseline", "target", "means_of_verification", "sector", "disaggregation", "unit_or_type", mode="before")
    @classmethod
    def _coerce_optional_text(cls, v):
        # The LLM often returns numbers (e.g. target: 60) or lists; keep them as strings rather than dropping the row.
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            v = "; ".join(str(x) for x in v if x is not None)
        s = str(v).strip()
        return s if s and s.lower() not in ("null", "none", "n/a", "na") else None

    @field_validator("indicator_text", mode="before")
    @classmethod
    def _coerce_indicator_text(cls, v):
        if v is None:
            raise ValueError("indicator_text is required")
        s = re.sub(r"\s+", " ", str(v)).strip()
        if not s:
            raise ValueError("indicator_text is empty")
        return s


# ------------------ Deterministic utilities (unit-test targets) ------------------

def normalize_text_for_dedupe(text: str) -> str:
    """Normalize indicator text for deduplication: collapse whitespace, lower, and normalize common symbol forms."""
    t = text.lower()
    t = t.replace("%", " percent ")
    t = re.sub(r"\band\b", " & ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # keep alphanumerics, spaces, percent text, slash, ampersand, dots, and hyphens
    t = re.sub(r"[^a-z0-9 %/.&-]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def score_chunk_for_indicators(chunk: DocumentChunk) -> float:
    """Return a 0-1 score representing likelihood the chunk contains indicators.

    Heuristics:
    - presence of keyword increases score
    - table-like header patterns boost score
    - table chunks get a base boost
    """
    text = chunk.content.lower()
    score = 0.0

    # base by chunk type
    if chunk.chunk_type == "table":
        score += 0.4

    # keyword hits
    kw_hits = sum(1 for kw in INDICATOR_KEYWORDS if kw in text)
    if kw_hits:
        score += min(0.4, 0.08 * kw_hits)

    # table header heuristics
    header_hits = sum(1 for pat in TABLE_HEADER_PATTERNS if re.search(pat, text))
    if header_hits:
        score += 0.3

    # length penalty: extremely short chunks are less likely
    if chunk.char_count < 40:
        score *= 0.5

    # clamp
    return max(0.0, min(1.0, score))


def is_table_like_text(text: str) -> bool:
    """Rudimentary detection of table-like text (rows with consistent separators or headers)."""
    # rows with tabs or multiple consecutive spaces or pipes
    if "\t" in text:
        return True
    if "|" in text and text.count("\n") >= 1:
        return True
    # repeated sequences of 2+ spaces (common in extracted tables)
    if re.search(r"\s{2,}", text) and text.count("\n") >= 1:
        return True
    return False


def deduplicate_indicators(indicators: List[Indicator]) -> Tuple[List[Indicator], List[Tuple[int, int, int]]]:
    """Deduplicate indicators within a list (typically per-document).

    Returns (unique_indicators, merged_pairs) where merged_pairs is list of tuples (kept_idx, removed_idx, score)
    """
    if fuzz is None:
        raise RuntimeError("rapidfuzz is required for deduplication but not available")

    normalized = [normalize_text_for_dedupe(i.indicator_text) for i in indicators]
    keep = []
    merged = []

    for idx, text in enumerate(normalized):
        kept = False
        for kidx, ktext in enumerate([normalize_text_for_dedupe(x.indicator_text) for x in keep]):
            score = fuzz.token_sort_ratio(text, ktext)
            if score >= 90:
                # consider duplicate; prefer existing (keep) over new
                merged.append((kidx, idx, int(score)))
                kept = True
                break
        if not kept:
            keep.append(indicators[idx])
    return keep, merged


def classify_match_status(embedding_score: float, lexical_score: float, thresholds: Dict = THRESHOLDS) -> str:
    """Classify a combined match status per plan thresholds.

    lexical_score is 0-100; embedding_score is 0-1
    """
    if embedding_score >= thresholds["exact_embedding"] and lexical_score >= thresholds["exact_lexical"]:
        return "exact"
    if embedding_score >= thresholds["close_embedding"]:
        return "close"
    if embedding_score >= thresholds["related_embedding"]:
        return "related"
    return "novel"


# ------------------ Simple cache helpers ------------------

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


    """Skeleton LLM client that wraps OpenAI calls and uses the cache helpers above.

    This class intentionally does not call the API in import-time code. The method "extract_indicators_from_chunk"
    is designed to be invoked by the notebook/script where OPENAI_API_KEY is set.
    """

    # load_dotenv()  # load .env if present
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # client = OpenAI(api_key=OPENAI_API_KEY) 

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str = CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = cache_dir
        # import openai lazily in methods that need it so tests can import module without network deps

    def extract_indicators_from_chunk(self, chunk: DocumentChunk, prompt_version: str = "v1") -> List[Indicator]:
        """Call the LLM to extract indicators from a chunk, using caching; returns list of Indicator.

        This method will try to load cached response first. Network calls should be mocked in tests.
        """
        # try cache
        cached = load_cached_response(chunk, prompt_version)
        if cached is not None:
            # Expect cached response to be a dict with 'indicators' key
            raw = cached
        else:
            # Real call path: import openai and call. To keep this module importable in environments
            # without credentials, do the import here and allow callers to mock this method in tests.
            try:
                import openai
            except Exception as exc:
                raise RuntimeError("openai package is required to call the LLM") from exc

            # Build a simple prompt (not fully implemented here). The notebook should craft a richer prompt.
            prompt = f"Extract monitoring indicators from the following chunk. Respond as JSON list under key 'indicators': {chunk.content}"

            # Basic wrapper (synchronous). Caller is responsible for setting OPENAI_API_KEY in env.

            # type: ignore[attr-defined]
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )
            # simplistic parsing: assume response.choices[0].message.content is JSON
            text = response.choices[0].message.get("content", "")
            try:
                raw = json.loads(text)
            except Exception:
                # fallback: wrap as empty
                raw = {"indicators": []}
            # cache
            try:
                cache_response_for_chunk(chunk, raw, prompt_version)
            except Exception:
                pass

        # parse raw into Indicator objects
        out = []
        for rec in raw.get("indicators", []) if isinstance(raw, dict) else []:
            try:
                ind = Indicator(**rec)
                # attach provenance
                ind.doc_id = chunk.doc_id
                ind.chunk_ref = f"{chunk.source_file}:{chunk.page_or_section}"
                out.append(ind)
            except ValidationError:
                continue
        return out


# ------------------ Deterministic parse helpers ------------------

DISCARD_KEYWORDS = [
    "budget",
    "annex",
    "cv",
    "résumé",
    "procurement",
    "finance",
    "financial",
    "appendix",
    "terms of reference",
    "borrowing",
    "personal history",
]


def _serialize_docx_table(table) -> str:
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    sep_cells = ["---" if cell else "" for cell in table.rows[0].cells]
    sep_line = "| " + " | ".join(sep_cells) + " |"
    return "\n".join([rows[0], sep_line] + rows[1:])


def simple_parse_docx(path: str) -> List[DocumentChunk]:
    try:
        from docx import Document
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("python-docx is required to parse .docx files") from exc

    doc = Document(path)
    chunks: List[DocumentChunk] = []
    paragraph_texts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    for idx, p in enumerate(paragraph_texts):
        chunks.append(
            DocumentChunk(
                doc_id=os.path.basename(path),
                source_file=path,
                chunk_type="text",
                page_or_section=f"para_{idx + 1}",
                content=p.strip(),
                char_count=len(p.strip()),
            )
        )

    for idx, table in enumerate(doc.tables):
        table_text = _serialize_docx_table(table)
        if table_text:
            chunks.append(
                DocumentChunk(
                    doc_id=os.path.basename(path),
                    source_file=path,
                    chunk_type="table",
                    page_or_section=f"table_{idx + 1}",
                    content=table_text,
                    char_count=len(table_text),
                )
            )
    return chunks


def simple_parse_pdf(path: str) -> List[DocumentChunk]:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pymupdf (fitz) is required to parse PDFs") from exc

    doc = fitz.open(path)
    chunks: List[DocumentChunk] = []
    for pnum in range(len(doc)):
        page = doc.load_page(pnum)
        text = page.get_text("text")
        chunk_type = "table" if is_table_like_text(text) else "text"
        chunks.append(
            DocumentChunk(
                doc_id=os.path.basename(path),
                source_file=path,
                chunk_type=chunk_type,
                page_or_section=f"page_{pnum + 1}",
                content=text.strip(),
                char_count=len(text.strip()),
            )
        )
    return chunks


def extract_chunks_from_file(path: str) -> Tuple[List[DocumentChunk], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            chunks = simple_parse_pdf(path)
        elif ext == ".docx":
            chunks = simple_parse_docx(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        if not chunks:
            errors.append({"doc_id": os.path.basename(path), "source_file": path, "error": "no content extracted"})
        return chunks, errors
    except Exception as exc:
        errors.append({"doc_id": os.path.basename(path), "source_file": path, "error": str(exc)})
        return [], errors


def chunk_discard_reason(chunk: DocumentChunk) -> Optional[str]:
    text = chunk.content.lower()
    for kw in DISCARD_KEYWORDS:
        if kw in text:
            return f"discard keyword: {kw}"
    return None


def filter_indicator_chunks(
    chunks: List[DocumentChunk],
    min_score: float = 0.30,
    keep_neighbors: int = 1,
) -> Tuple[List[DocumentChunk], Dict[str, object]]:
    scored = [(idx, chunk, score_chunk_for_indicators(chunk)) for idx, chunk in enumerate(chunks)]
    keep_indices = {idx for idx, _, score in scored if score >= min_score}
    if keep_neighbors:
        expanded = set(keep_indices)
        for idx in sorted(keep_indices):
            for neighbor in range(idx - keep_neighbors, idx + keep_neighbors + 1):
                if 0 <= neighbor < len(chunks):
                    expanded.add(neighbor)
        keep_indices = expanded

    selected = [chunks[idx] for idx in sorted(keep_indices)]
    discarded = [chunks[idx] for idx in range(len(chunks)) if idx not in keep_indices]
    coverage = sum(len(c.content) for c in selected) / max(1, sum(len(c.content) for c in chunks))
    discard_details = [
        {"doc_id": c.doc_id, "source_file": c.source_file, "reason": chunk_discard_reason(c) or "low score"}
        for c in discarded
    ]
    summary = {
        "total_chunks": len(chunks),
        "selected_chunks": len(selected),
        "discarded_chunks": len(discarded),
        "coverage_fraction": round(coverage, 4),
        "discard_details": discard_details,
    }
    return selected, summary


def _extract_json_from_response(text: str) -> Optional[object]:
    decoder = json.JSONDecoder()
    start = 0
    while start < len(text):
        try:
            obj, end = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            start += 1
    return None


def _build_extraction_prompt(chunk: DocumentChunk) -> str:
    schema = {
        "indicator_text": "string",
        "level": "impact | outcome | output | activity | unclear",
        "baseline": "string or null",
        "target": "string or null",
        "means_of_verification": "string or null",
        "sector": "string or null",
        "disaggregation": "string or null",
        "unit_or_type": "string or null",
        "source_confidence": "high | medium | low",
    }
    return (
        "Extract all monitoring and evaluation indicators from the following chunk. "
        "Return only valid JSON with one top-level key, \"indicators\", whose value is an array of objects. "
        "Each object must include exactly these fields: indicator_text, level, baseline, target, means_of_verification, sector, disaggregation, unit_or_type, source_confidence. "
        "Use null when a field is not available. If there are no indicators, return {\"indicators\": []}. "
        "Do not include any markdown syntax in field values. Do not return any extra keys. "
        f"Chunk content:\n\n{chunk.content}\n"
    )


def _call_openai_with_retries(
    prompt: str,
    model_name: str,
    max_retries: int = 3,
    response_format: Optional[dict] = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> str:

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


class CacheMiss(LookupError):
    """Raised by LLMClient.extract_indicators_from_chunk(cache_only=True) when no cached response exists."""


class LLMClient:
    """LLM client that wraps chat-completion calls (OpenAI or Ollama) and uses the cache helpers above."""

    def __init__(self, model_name: Optional[str] = None, cache_dir: str = CACHE_DIR):
        self.model_name = model_name or MODEL_NAME
        self.cache_dir = cache_dir

    def extract_indicators_from_chunk(self, chunk: DocumentChunk, prompt_version: str = "v1", cache_only: bool = False) -> List[Indicator]:
        """Extract indicators from one chunk, using the on-disk cache when available.

        With cache_only=True no API call is made: a cache miss raises CacheMiss so the caller can count it.
        This lets the notebook rebuild its dataset from cached responses when the LLM server is down.
        """
        cached = load_cached_response(chunk, prompt_version, self.model_name)
        if cached is not None:
            raw = cached
        elif cache_only:
            raise CacheMiss(f"no cached response for chunk {chunk.doc_id}:{chunk.page_or_section} with model {self.model_name}")
        else:
            prompt = _build_extraction_prompt(chunk)
            text = _call_openai_with_retries(prompt, self.model_name, response_format={"type": "json_object"})
            raw = _extract_json_from_response(text)
            if raw is None or not isinstance(raw, dict):
                raw = {"indicators": []}
            # Small local models sometimes return a bare list or wrap it under another key.
            if "indicators" not in raw:
                lists = [v for v in raw.values() if isinstance(v, list)]
                raw = {"indicators": lists[0] if lists else []}
            try:
                cache_response_for_chunk(chunk, raw, prompt_version, self.model_name)
            except Exception:
                pass

        out = []
        for rec in raw.get("indicators", []) if isinstance(raw, dict) else []:
            try:
                ind = Indicator(**rec)
                ind.doc_id = chunk.doc_id
                ind.chunk_ref = f"{chunk.source_file}:{chunk.page_or_section}"
                out.append(ind)
            except ValidationError:
                continue
        return out


def _split_indikit_wording(wording: str) -> Tuple[str, str]:
    """Split an IndiKit "Indicator's Wording" block into (english, french).

    The block is normally "English\\nFrench\\n<english wording>\\n<french wording>", with occasional
    hard line breaks (\\r\\n) inside a sentence. Returns empty strings when nothing usable is present.
    """
    if not wording or not isinstance(wording, str):
        return "", ""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in wording.replace("\r\n", " ").split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return "", ""
    if len(lines) >= 3 and lines[0].lower() == "english" and lines[1].lower() == "french":
        return lines[2], " ".join(lines[3:])
    return lines[0], ""


def build_reference_library_from_zoa(zoa_path: str, out_path: str):
    """Convert the ZOA IndiKit export into the reference_library.csv format the matcher expects.

    Output columns: indicator_id (IndiKit slug), indicator_text (English wording, or title when wording is
    missing), indicator_text_fr, title, sector, level (blank: IndiKit has no result level), disaggregation,
    url, text_source ("wording" | "title"). Returns the DataFrame that was written.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to build the reference library") from exc

    zoa = pd.read_csv(zoa_path)
    wording_col = next((c for c in zoa.columns if c.startswith("section:Indicator") and "Wording" in c), None)
    disagg_col = next((c for c in zoa.columns if c.startswith("section:Disaggregate")), None)

    rows = []
    for _, r in zoa.iterrows():
        english, french = _split_indikit_wording(r[wording_col] if wording_col and pd.notna(r[wording_col]) else "")
        title = str(r.get("title", "")).strip()
        text = english or title
        disagg = str(r[disagg_col]).replace("\r\n", " ").replace("\n", " ").strip() if disagg_col and pd.notna(r[disagg_col]) else ""
        rows.append(
            {
                "indicator_id": str(r.get("slug", "")).strip(),
                "indicator_text": text,
                "indicator_text_fr": french,
                "title": title,
                "sector": str(r.get("sector", "")).strip(),
                "level": "",
                "disaggregation": re.sub(r"\s+", " ", disagg),
                "url": str(r.get("url", "")).strip(),
                "text_source": "wording" if english else "title",
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["indicator_text"].astype(str).str.strip() != ""]
    df = df.sort_values("indicator_id").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def load_reference_library(path: str):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to load the reference library") from exc

    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str).fillna("")
        for col in ["indicator_id", "indicator_text", "sector", "level"]:
            if col not in df.columns:
                df[col] = ""
        return df, False

    dummy = [
        {"indicator_id": "REF-001", "indicator_text": "Percentage of households with access to safe drinking water", "sector": "WASH", "level": "output"},
        {"indicator_id": "REF-002", "indicator_text": "Number of farmers adopting climate-smart agriculture practices", "sector": "FSL", "level": "outcome"},
        {"indicator_id": "REF-003", "indicator_text": "Proportion of children under five vaccinated against measles", "sector": "health", "level": "impact"},
    ]
    df = pd.DataFrame(dummy)
    return df, True


def _best_lexical_match(query: str, choices: List[str]) -> Tuple[int, int]:
    if fuzz is None:
        raise RuntimeError("rapidfuzz is required for lexical matching")
    best_score = -1
    best_idx = -1
    for idx, choice in enumerate(choices):
        score = fuzz.token_sort_ratio(query, choice)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


# Embedding backends, tried in order. fastembed runs the same MiniLM model through ONNX Runtime and needs
# no PyTorch, which matters on Intel Macs where PyTorch stopped at 2.2.2 (incompatible with numpy 2 and
# transformers 5). sentence-transformers is kept as a fallback for machines where PyTorch works.
EMBEDDING_BACKENDS = ("fastembed", "sentence-transformers")
_EMBEDDING_MODEL_CACHE: Dict[Tuple[str, str], object] = {}
_EMBEDDING_BACKEND_USED: Optional[str] = None


def _fastembed_model_name(model_name: str) -> str:
    """fastembed registers Sentence-Transformers models under their full Hugging Face id."""
    if "/" in model_name:
        return model_name
    return f"sentence-transformers/{model_name}"


def _load_embedding_model(backend: str, model_name: str):
    key = (backend, model_name)
    if key in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[key]
    if backend == "fastembed":
        from fastembed import TextEmbedding

        model = TextEmbedding(_fastembed_model_name(model_name))
    elif backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
    _EMBEDDING_MODEL_CACHE[key] = model
    return model


def _compute_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    """Return L2-normalised embeddings (numpy array, shape n x dim) using the first backend that works.

    Raises RuntimeError listing every backend failure when none is usable.
    """
    global _EMBEDDING_BACKEND_USED
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for embedding similarity") from exc

    failures = []
    backends = (_EMBEDDING_BACKEND_USED,) if _EMBEDDING_BACKEND_USED else EMBEDDING_BACKENDS
    for backend in backends:
        try:
            model = _load_embedding_model(backend, model_name)
            if backend == "fastembed":
                embeddings = np.asarray(list(model.embed(list(texts))), dtype="float32")
            else:
                embeddings = np.asarray(
                    model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False),
                    dtype="float32",
                )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
                raise RuntimeError(f"{backend} returned embeddings of shape {embeddings.shape} for {len(texts)} texts")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            if _EMBEDDING_BACKEND_USED != backend:
                _EMBEDDING_BACKEND_USED = backend
                print(f"Embeddings: using {backend} backend with model {model_name}")
            return embeddings
        except Exception as exc:
            failures.append(f"{backend}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
    raise RuntimeError("no embedding backend available (" + "; ".join(failures) + ")")


def _best_embedding_matches(query_embeddings, reference_embeddings):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for embedding similarity") from exc
    query = query_embeddings.reshape(1, -1)
    scores = reference_embeddings.dot(query.T).flatten().tolist()
    best_idx = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else -1
    return best_idx, float(scores[best_idx]) if best_idx >= 0 else 0.0


def match_indicators_to_library(indicators: List[Indicator], reference_path: str, embedding_model: Optional[str] = "all-MiniLM-L6-v2"):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for matching indicators") from exc

    reference_df, dummy_library = load_reference_library(reference_path)
    ref_texts = reference_df["indicator_text"].astype(str).tolist()
    match_rows = []

    if not indicators:
        return pd.DataFrame(match_rows), dummy_library

    extracted_texts = [ind.indicator_text for ind in indicators]

    # Lexical matches (always available)
    lex_results = [_best_lexical_match(q, ref_texts) for q in extracted_texts]

    # Embedding matches (optional)
    use_embeddings = bool(embedding_model)
    ref_embeddings = None
    query_embeddings = None
    if use_embeddings:
        try:
            ref_embeddings = _compute_embeddings(ref_texts, embedding_model)
            query_embeddings = _compute_embeddings(extracted_texts, embedding_model)
        except Exception as exc:
            # If embeddings fail, fall back to lexical-only, but say so loudly
            print(f"WARNING: embedding similarity unavailable ({type(exc).__name__}: {exc}). Falling back to lexical matching only.")
            use_embeddings = False
            ref_embeddings = None
            query_embeddings = None

    for idx, ind in enumerate(indicators):
        lex_idx, lex_score = lex_results[idx]
        emb_idx, emb_score = -1, 0.0
        if use_embeddings and query_embeddings is not None and ref_embeddings is not None:
            emb_idx, emb_score = _best_embedding_matches(query_embeddings[idx], ref_embeddings)
        ref_lex = reference_df.iloc[lex_idx] if lex_idx >= 0 else None
        ref_emb = reference_df.iloc[emb_idx] if emb_idx >= 0 else None
        status = classify_match_status(emb_score, lex_score)
        match_rows.append(
            {
                "doc_id": ind.doc_id,
                "indicator_text": ind.indicator_text,
                "level": ind.level,
                "baseline": ind.baseline,
                "target": ind.target,
                "means_of_verification": ind.means_of_verification,
                "sector": ind.sector,
                "disaggregation": ind.disaggregation,
                "unit_or_type": ind.unit_or_type,
                "source_confidence": ind.source_confidence,
                "reference_id_lexical": ref_lex["indicator_id"] if ref_lex is not None else "",
                "reference_text_lexical": ref_lex["indicator_text"] if ref_lex is not None else "",
                "lexical_score": lex_score,
                "reference_id_embedding": ref_emb["indicator_id"] if ref_emb is not None else "",
                "reference_text_embedding": ref_emb["indicator_text"] if ref_emb is not None else "",
                "embedding_score": emb_score,
                "match_status": status,
                "dummy_library": dummy_library,
            }
        )

    return pd.DataFrame(match_rows), dummy_library


# ------------------ Linking proposal indicators to the ZOA reference (embedding shortlist + LLM judgement) ------------------

LINK_PROMPT_VERSION = "link-v1"
LINK_STATUSES = ("Present", "Different", "Missing")


def shortlist_reference_candidates(
    indicator_texts: List[str],
    reference_df,
    k: int = 5,
    embedding_model: Optional[str] = "all-MiniLM-L6-v2",
) -> List[List[Dict[str, object]]]:
    """For each proposal indicator, return the k most similar reference indicators.

    Uses sentence embeddings (cosine similarity) over the whole library; falls back to rapidfuzz
    token_sort_ratio if embeddings are unavailable. Each candidate is a dict with indicator_id,
    indicator_text, sector, score (0-1) and method ("embedding" | "lexical").
    """
    ref_ids = reference_df["indicator_id"].astype(str).tolist()
    ref_texts = reference_df["indicator_text"].astype(str).tolist()
    ref_sectors = reference_df["sector"].astype(str).tolist() if "sector" in reference_df.columns else [""] * len(ref_texts)
    if not indicator_texts or not ref_texts:
        return [[] for _ in indicator_texts]
    k = max(1, min(k, len(ref_texts)))

    def _cand(j: int, score: float, method: str) -> Dict[str, object]:
        return {"indicator_id": ref_ids[j], "indicator_text": ref_texts[j], "sector": ref_sectors[j], "score": round(float(score), 4), "method": method}

    if embedding_model:
        try:
            import numpy as np

            ref_emb = _compute_embeddings(ref_texts, embedding_model)
            q_emb = _compute_embeddings(indicator_texts, embedding_model)
            sims = q_emb @ ref_emb.T
            out = []
            for i in range(len(indicator_texts)):
                top = np.argsort(-sims[i])[:k]
                out.append([_cand(int(j), sims[i][j], "embedding") for j in top])
            return out
        except Exception as exc:
            print(f"WARNING: embedding shortlist unavailable ({type(exc).__name__}: {exc}). Falling back to lexical shortlist.")

    if fuzz is None:
        raise RuntimeError("rapidfuzz is required for the lexical shortlist fallback")
    out = []
    for q in indicator_texts:
        scored = sorted(((fuzz.token_sort_ratio(q, t) / 100.0, j) for j, t in enumerate(ref_texts)), reverse=True)[:k]
        out.append([_cand(j, s, "lexical") for s, j in scored])
    return out


def _build_link_prompt(indicator_text: str, candidates: List[Dict[str, object]]) -> str:
    cand_lines = "\n".join(
        f"  {n}. id={c['indicator_id']} | sector={c['sector']} | {c['indicator_text']}" for n, c in enumerate(candidates, 1)
    )
    return (
        "You are an M&E specialist comparing a project proposal indicator with the ZOA reference indicator library.\n\n"
        f"PROPOSAL INDICATOR:\n\"{indicator_text}\"\n\n"
        f"CLOSEST ZOA REFERENCE INDICATORS (pre-selected by similarity):\n{cand_lines}\n\n"
        "TASK: decide how the proposal indicator relates to the reference library and answer with ONE status:\n"
        "- \"Present\": a candidate measures the same thing. Differences only in target group, location, numbers, or minor wording.\n"
        "- \"Different\": a candidate covers the same topic, but the proposal indicator measures it in a materially different way "
        "(different unit, scope, calculation, or population definition), so it would need adapting to fit the reference.\n"
        "- \"Missing\": none of the candidates cover the topic of the proposal indicator.\n\n"
        "Respond with JSON only, no extra text, using exactly these keys:\n"
        "{\"reference_id\": \"<id of the best candidate, or null when Missing>\", "
        "\"status\": \"Present\" | \"Different\" | \"Missing\", "
        "\"confidence\": <number between 0 and 1>, "
        "\"reason\": \"<one sentence>\"}"
    )


def link_indicator_to_reference(
    indicator_text: str,
    candidates: List[Dict[str, object]],
    model_name: Optional[str] = None,
    prompt_version: str = LINK_PROMPT_VERSION,
    cache_dir: str = CACHE_DIR,
) -> Dict[str, object]:
    """Ask the LLM whether a proposal indicator is Present, Different or Missing relative to the shortlisted candidates.

    Responses are cached on disk keyed by a hash of (prompt, model, prompt_version). Returns a dict with
    reference_id, reference_text, status, confidence, reason, and cached (bool). On API failure status is "Error".
    """
    if not candidates:
        return {"reference_id": None, "reference_text": None, "status": "Missing", "confidence": 1.0, "reason": "No reference candidates available.", "cached": False}

    model_name = model_name or MODEL_NAME
    prompt = _build_link_prompt(indicator_text, candidates)
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(model_name.encode("utf-8"))
    h.update(prompt_version.encode("utf-8"))
    cache_path = os.path.join(cache_dir, f"link_{h.hexdigest()}.json")

    raw = None
    cached = False
    if not FORCE_REFRESH and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh).get("response")
                cached = raw is not None
        except Exception:
            raw = None

    if raw is None:
        try:
            text = _call_openai_with_retries(prompt, model_name, response_format={"type": "json_object"}, temperature=0.0, max_tokens=300)
        except Exception as exc:
            return {"reference_id": None, "reference_text": None, "status": "Error", "confidence": 0.0, "reason": f"{type(exc).__name__}: {exc}", "cached": False}
        raw = _extract_json_from_response(text)
        if not isinstance(raw, dict):
            raw = {"reference_id": None, "status": "Error", "confidence": 0.0, "reason": f"Unparseable response: {text[:200]}"}
        else:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump({"prompt_version": prompt_version, "model": model_name, "indicator_text": indicator_text, "response": raw}, fh, ensure_ascii=False, indent=2)
            except Exception:
                pass

    # Normalise and validate against the candidate list so the model cannot invent a reference id.
    by_id = {str(c["indicator_id"]): c for c in candidates}
    status = str(raw.get("status", "")).strip().capitalize()
    if status not in LINK_STATUSES:
        status = "Error" if status.lower() == "error" else "Missing"
    ref_id = raw.get("reference_id")
    ref_id = str(ref_id).strip() if ref_id not in (None, "", "null") else None
    if ref_id is not None and ref_id not in by_id:
        # tolerate the model echoing the text instead of the id
        text_match = next((c for c in candidates if str(c["indicator_text"]).strip().lower() == str(ref_id).strip().lower()), None)
        ref_id = str(text_match["indicator_id"]) if text_match else None
    if status == "Missing":
        ref_id = None
    if status in ("Present", "Different") and ref_id is None:
        ref_id = str(candidates[0]["indicator_id"])
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "reference_id": ref_id,
        "reference_text": by_id[ref_id]["indicator_text"] if ref_id else None,
        "status": status,
        "confidence": confidence,
        "reason": str(raw.get("reason", "")).strip(),
        "cached": cached,
    }


def simple_parse_docx_text(path: str) -> List[DocumentChunk]:
    return simple_parse_docx(path)


def simple_parse_pdf_text(path: str) -> List[DocumentChunk]:
    return simple_parse_pdf(path)


# End of module
