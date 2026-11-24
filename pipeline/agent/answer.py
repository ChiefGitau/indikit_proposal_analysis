"""LLM answering (paper Section 3.4.2; docs/phase_3.md Task 3) plus the legacy cache helpers of Phase 1.

``answer_pair(doc_id, indicator, ...)`` retrieves the evidence, builds the five-part prompt, calls the provider
(cached per document SHA-256 x IndiKit ID x evidence fingerprint x prompt version x rubric version x model),
validates the ``DisclosureAnswer`` and post-validates every match in code:

- ``evidence_label`` must exist; ``evidence_quote`` and ``document_indicator_text`` must be found in that evidence's
  text (whitespace / case-insensitive, small fuzzy tolerance), otherwise the match is ``unsupported`` and excluded;
- baseline / target are normalised with ``pipeline.eval.metrics.normalise_value`` (raw strings kept);
- the label is mapped back to ``entry_id``, location, table / row / block ids.

``answer_pairs`` runs many pairs with bounded concurrency and a resumable progress file.

The module-level settings (``LLM_PROVIDER``, ``MODEL_NAME``, ``CACHE_DIR``, ``FORCE_REFRESH`` ...) and the
chunk-cache helpers below are the Phase 1 legacy used by ``extraction_utils`` and the old notebook cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Optional, Sequence, Tuple

from pipeline import config
from pipeline.agent.llm import BudgetExceeded, LLMProvider, LLMResult, ResponseCache, UsageTracker, cache_key, model_role, run_parallel
from pipeline.agent.models import DisclosureAnswer, MatchItem, PairAnswer, ValidatedMatch
from pipeline.agent.prompts import build_disclosure_messages, evidence_items
from pipeline.eval.metrics import normalise_value
from pipeline.metadata.models import IndicatorMetadata
from pipeline.models import DocumentChunk

if TYPE_CHECKING:  # avoid the import cycle novel_pass -> answer -> retrieve -> knowledge_base -> logframe -> novel_pass
    from pipeline.agent.retrieve import Evidence, RetrievalConfig, RetrievalResult

# --------------------------------------------------------------------------- #
# Legacy provider settings (Phase 1; forwarded by extraction_utils)
# --------------------------------------------------------------------------- #
SUPPORTED_PROVIDERS = ("ollama", "openai", "azure")

LLM_PROVIDER: str = config.DEFAULT_LLM_PROVIDER
MODEL_NAME: str = config.DEFAULT_MODEL_NAME
OLLAMA_BASE_URL: str = config.OLLAMA_BASE_URL
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
LEGACY_CACHE_MODEL: str = config.LEGACY_CACHE_MODEL
FORCE_REFRESH: bool = False
CACHE_DIR: str = str(config.API_CACHE_DIR)


def configure_llm(provider: Optional[str] = None, model_name: Optional[str] = None, base_url: Optional[str] = None, azure_endpoint: Optional[str] = None, azure_api_version: Optional[str] = None) -> Dict[str, str]:
    """Set the active legacy LLM provider/model (notebook config cell)."""
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
    endpoint = {"ollama": OLLAMA_BASE_URL, "openai": "https://api.openai.com/v1", "azure": AZURE_OPENAI_ENDPOINT or "<AZURE_OPENAI_ENDPOINT not set>"}[LLM_PROVIDER]
    return {"provider": LLM_PROVIDER, "model": MODEL_NAME, "base_url": endpoint}


def _make_client():
    """OpenAI-SDK client for the active legacy provider."""
    from openai import AzureOpenAI, OpenAI

    if LLM_PROVIDER == "ollama":
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key=os.getenv("OLLAMA_API_KEY", "ollama"))
    if LLM_PROVIDER == "azure":
        endpoint = AZURE_OPENAI_ENDPOINT or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not key:
            raise RuntimeError("LLM_PROVIDER is 'azure' but AZURE_OPENAI_ENDPOINT and/or AZURE_OPENAI_API_KEY are not set.")
        return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=AZURE_OPENAI_API_VERSION)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER is 'openai'.")
    return OpenAI(api_key=key)


def llm_available(timeout: float = 3.0) -> Tuple[bool, str]:
    """Cheap readiness check for the active legacy provider."""
    if LLM_PROVIDER == "openai":
        return (True, f"OpenAI configured, model {MODEL_NAME}") if os.getenv("OPENAI_API_KEY") else (False, "OPENAI_API_KEY is not set")
    if LLM_PROVIDER == "azure":
        endpoint = AZURE_OPENAI_ENDPOINT or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        missing = [n for n, v in (("AZURE_OPENAI_ENDPOINT", endpoint), ("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"))) if not v]
        if missing:
            return False, f"Azure OpenAI not configured: missing {', '.join(missing)}"
        return True, f"Azure OpenAI configured at {endpoint}, deployment {MODEL_NAME}"
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
        return False, f"Ollama server not reachable at {OLLAMA_BASE_URL} ({type(exc).__name__})"


class CacheMiss(LookupError):
    """Raised by LLMClient.extract_indicators_from_chunk(cache_only=True) when no cached response exists."""


def _chunk_hash(chunk: DocumentChunk, prompt_version: str = "v1", model_name: Optional[str] = None) -> str:
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


def _call_openai_with_retries(prompt: str, model_name: str, max_retries: int = 3, response_format: Optional[dict] = None, temperature: float = 0.0, max_tokens: int = 1500) -> str:
    """Legacy single-prompt call with exponential backoff (used by the old extraction client)."""
    client = _make_client()
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
            if response_format:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError(f"LLM request to {LLM_PROVIDER} failed after retries")


_call_llm_with_retries = _call_openai_with_retries


# --------------------------------------------------------------------------- #
# Post-validation helpers
# --------------------------------------------------------------------------- #
_NORM_RE = re.compile(r"[^0-9a-zà-ÿ%#]+")


def norm_text(text: str) -> str:
    """Lower-case, strip punctuation / whitespace differences (keeps %, # and digits)."""
    return _NORM_RE.sub(" ", (text or "").lower().replace("’", "'")).strip()


def text_found(needle: str, haystack: str, threshold: int = config.QUOTE_MATCH_THRESHOLD) -> tuple[bool, float]:
    """Is ``needle`` contained in ``haystack`` after normalisation, allowing small fuzzy differences?"""
    n, h = norm_text(needle), norm_text(haystack)
    if not n:
        return False, 0.0
    if n in h:
        return True, 100.0
    try:
        from rapidfuzz import fuzz

        score = float(fuzz.partial_ratio(n, h))
    except Exception:  # pragma: no cover
        score = 0.0
    return score >= threshold, score


def text_similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz

        return float(fuzz.token_set_ratio(norm_text(a), norm_text(b)))
    except Exception:  # pragma: no cover
        return 0.0


def evidence_fingerprint(evidence: Sequence["Evidence"]) -> str:
    h = hashlib.sha256()
    for e in evidence:
        h.update(e.entry_id.encode("utf-8"))
        h.update(b"|")
        h.update(hashlib.sha256(e.display_text.encode("utf-8")).digest())
        h.update(b"\n")
    return h.hexdigest()[:16]


def validate_matches(answer: DisclosureAnswer, evidence: Sequence["Evidence"], indicator: Optional[IndicatorMetadata] = None, threshold: int = config.QUOTE_MATCH_THRESHOLD) -> tuple[list[ValidatedMatch], list[ValidatedMatch]]:
    """Split the answer's matches into (supported, unsupported). A match is supported when its label exists and both
    its quote and its indicator text are found in that evidence item."""
    by_label = {f"E{i}": e for i, e in enumerate(evidence, start=1)}
    supported: list[ValidatedMatch] = []
    unsupported: list[ValidatedMatch] = []
    wording = (indicator.kpi_wording_en or indicator.name) if indicator is not None else ""
    for m in answer.matches:
        ev = by_label.get(m.evidence_label.upper())
        vm = ValidatedMatch(indikit_id=answer.indikit_id, document_indicator_text=m.document_indicator_text, match_type=m.match_type, level=m.level, baseline=m.baseline, target=m.target,
                            baseline_norm=normalise_value(m.baseline), target_norm=normalise_value(m.target), unit=m.unit, evidence_label=m.evidence_label, evidence_quote=m.evidence_quote, reason=m.reason)
        if ev is None:
            vm.supported, vm.unsupported_reason = False, f"evidence label {m.evidence_label!r} does not exist"
            unsupported.append(vm)
            continue
        vm.entry_id, vm.location, vm.kb_type, vm.table_id, vm.row_id, vm.block_ids = ev.entry_id, ev.location, ev.kb_type, ev.table_id, ev.row_id, list(ev.block_ids)
        vm.rerank_score, vm.vector_score = ev.rerank_score, ev.vector_score
        ok_text, s_text = text_found(m.document_indicator_text, ev.display_text, threshold)
        ok_quote, s_quote = (True, 100.0) if not m.evidence_quote else text_found(m.evidence_quote, ev.display_text, threshold)
        if not ok_text or not ok_quote:
            vm.supported = False
            vm.unsupported_reason = ("indicator text not found in evidence" if not ok_text else "quote not found in evidence") + f" (similarity {min(s_text, s_quote):.0f})"
            unsupported.append(vm)
            continue
        vm.text_similarity = round(text_similarity(m.document_indicator_text, wording), 1) if wording else None
        supported.append(vm)
    return supported, unsupported


# --------------------------------------------------------------------------- #
# answer_pair
# --------------------------------------------------------------------------- #
RetrieveFn = Callable[[str, IndicatorMetadata], "RetrievalResult"]


def _doc_sha(doc_id: str, kb_dir: Path) -> str:
    meta = Path(kb_dir) / doc_id / "kb_meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("sha256", doc_id)
        except Exception:
            pass
    return doc_id


def answer_pair(
    doc_id: str,
    indicator: IndicatorMetadata,
    *,
    provider: LLMProvider,
    retrieval_cfg: Optional["RetrievalConfig"] = None,
    retrieve_fn: Optional[RetrieveFn] = None,
    retriever=None,
    use_knowledge: bool = True,
    use_search_terms: bool = True,
    model: Optional[str] = None,
    cache: Optional[ResponseCache] = None,
    force: bool = False,
    doc_sha256: Optional[str] = None,
    kb_dir: Path = config.KB_DIR,
    top_k: Optional[int] = None,
) -> PairAnswer:
    """Retrieve evidence, ask the model (or the cache) and post-validate. ``retrieve_fn(doc_id, indicator)`` overrides
    retrieval (baseline / tests); otherwise ``retriever.retrieve`` or the module-level ``retrieve`` is used."""
    from pipeline.agent.retrieve import RetrievalConfig

    model = model or model_role("ANSWER_MODEL")
    cfg = retrieval_cfg or RetrievalConfig()
    if retrieve_fn is not None:
        res = retrieve_fn(doc_id, indicator)
    else:
        from pipeline.agent.retrieve import retrieve

        kw = dict(top_k=top_k or cfg.top_k, top_n=cfg.top_n, kb_types=cfg.kb_types, use_search_terms=use_search_terms, use_rerank=cfg.use_rerank)
        res = retriever.retrieve(doc_id, indicator, **kw) if retriever is not None else retrieve(doc_id, indicator, config=cfg, **kw)
    evidence = list(res.evidence)[: (top_k or cfg.top_k)]
    fp = evidence_fingerprint(evidence)
    out = PairAnswer(doc_id=doc_id, indikit_id=indicator.indikit_id, evidence_ids=[e.entry_id for e in evidence], evidence_fingerprint=fp, best_rerank_score=res.best_rerank_score,
                     best_vector_score=res.best_vector_score, n_evidence=len(evidence), model=model, prompt_version=config.PROMPT_VERSION, rubric_version=config.RUBRIC_VERSION,
                     use_knowledge=use_knowledge, use_search_terms=use_search_terms)
    if not evidence:
        out.disclosure = "absent"
        out.error = "no evidence retrieved"
        return out
    system, user = build_disclosure_messages(indicator, evidence_items(evidence), use_knowledge=use_knowledge)
    key = cache_key(doc_sha256=doc_sha256 or _doc_sha(doc_id, kb_dir), indikit_id=indicator.indikit_id, evidence_fingerprint=fp, prompt_version=config.PROMPT_VERSION, rubric_version=config.RUBRIC_VERSION,
                    model=model, use_knowledge=use_knowledge)
    answer: Optional[DisclosureAnswer] = None
    cached = cache.get(key) if (cache is not None and not force) else None
    if cached is not None and isinstance(cached.get("response"), dict):
        try:
            answer = DisclosureAnswer.model_validate(cached["response"])
            out.cached = True
        except Exception:
            answer = None
    if answer is None:
        result: LLMResult = provider.complete_json(system, user, DisclosureAnswer, model=model)
        out.prompt_tokens, out.completion_tokens, out.latency_s, out.attempts = result.prompt_tokens, result.completion_tokens, result.latency_s, result.attempts
        if not result.ok:
            out.disclosure = "failed"
            out.error = result.error
            return out
        answer = result.parsed
        if cache is not None:
            cache.put(key, answer.model_dump(mode="json"), doc_id=doc_id, indikit_id=indicator.indikit_id, model=model, prompt_version=config.PROMPT_VERSION, rubric_version=config.RUBRIC_VERSION, evidence_fingerprint=fp)
    out.raw_answer = answer.model_dump(mode="json")
    supported, unsupported = validate_matches(answer, evidence, indicator)
    out.matches, out.unsupported = supported, unsupported
    out.disclosure = "present" if supported else "absent"
    return out


# --------------------------------------------------------------------------- #
# Batch with progress file (resumable)
# --------------------------------------------------------------------------- #
class ProgressFile:
    """Append-only JSONL of finished pairs; a resumed run skips the pairs already present. Pairs whose last
    recorded attempt failed (``disclosure == "failed"``) are *not* treated as done, so a resume retries them."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.done: dict[tuple[str, str], PairAnswer] = {}
        self.retried: int = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        pa = PairAnswer.model_validate_json(line)
                    except Exception:
                        continue
                    if pa.disclosure == "failed":
                        self.done.pop((pa.doc_id, pa.indikit_id), None)
                        self.retried += 1
                    else:
                        self.done[(pa.doc_id, pa.indikit_id)] = pa

    def add(self, pa: PairAnswer) -> None:
        with self._lock:
            self.done[(pa.doc_id, pa.indikit_id)] = pa
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(pa.model_dump_json() + "\n")


def answer_pairs(pairs: Sequence[tuple[str, IndicatorMetadata]], *, provider: LLMProvider, progress: Optional[ProgressFile] = None, workers: int = config.LLM_MAX_WORKERS,
                 on_done: Optional[Callable[[PairAnswer], None]] = None, **kw) -> tuple[list[PairAnswer], Optional[BudgetExceeded]]:
    """Answer many (doc_id, indicator) pairs with bounded concurrency. Returns (answers in input order, budget stop)."""
    todo_idx = [i for i, (d, ind) in enumerate(pairs) if progress is None or (d, ind.indikit_id) not in progress.done]
    results: list[Optional[PairAnswer]] = [progress.done.get((d, ind.indikit_id)) if progress else None for d, ind in pairs]
    stop: Optional[BudgetExceeded] = None

    def _one(i: int) -> PairAnswer:
        d, ind = pairs[i]
        try:
            return answer_pair(d, ind, provider=provider, **kw)
        except BudgetExceeded:
            raise
        except Exception as exc:
            return PairAnswer(doc_id=d, indikit_id=ind.indikit_id, disclosure="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}", model=kw.get("model") or model_role("ANSWER_MODEL"))

    def _record(i: int, r) -> None:
        nonlocal stop
        if isinstance(r, BudgetExceeded):
            stop = stop or r
            return
        results[i] = r
        if progress is not None:
            progress.add(r)
        if on_done is not None:
            on_done(r)

    run_parallel(_one, todo_idx, workers=workers, on_result=_record)
    return [r for r in results if r is not None], stop


def answer_indicator(doc_id: str, indicator: IndicatorMetadata, **kw) -> PairAnswer:
    """Phase 1 name of :func:`answer_pair`."""
    return answer_pair(doc_id, indicator, **kw)
