"""Novel-indicator pass (docs/plan.md Section 4.3, an addition beyond the paper).

The standard-driven disclosure step cannot see document indicators that have no
IndiKit counterpart, so the legacy per-chunk extraction is kept as a secondary
pass. This module holds the reusable parts moved from ``extraction_utils.py``:
chunk scoring and filtering (cheap pre-filter), deduplication, the extraction
prompt and the ``LLMClient`` that calls the model with caching.

Phase 3 adds ``run_novel_pass(doc_kb, disclosure_rows)`` which runs the client
on logframe chunks only and removes indicators already linked to an IndiKit ID.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from pipeline.agent import answer as _answer
from pipeline.models import DocumentChunk, Indicator

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - import guard for environments missing rapidfuzz
    fuzz = None


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
    """Keep chunks scoring at least ``min_score`` plus their neighbours; return (selected, summary)."""
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


# ------------------ Extraction prompt and client ------------------

def _build_extraction_prompt(chunk: DocumentChunk) -> str:
    return (
        "Extract all monitoring and evaluation indicators from the following chunk. "
        "Return only valid JSON with one top-level key, \"indicators\", whose value is an array of objects. "
        "Each object must include exactly these fields: indicator_text, level, baseline, target, means_of_verification, sector, disaggregation, unit_or_type, source_confidence. "
        "Use null when a field is not available. If there are no indicators, return {\"indicators\": []}. "
        "Do not include any markdown syntax in field values. Do not return any extra keys. "
        f"Chunk content:\n\n{chunk.content}\n"
    )


class LLMClient:
    """LLM client that wraps chat-completion calls (OpenAI, Azure or Ollama) and uses the cache helpers in
    ``pipeline.agent.answer``. This is the surviving (second) definition from ``extraction_utils.py``."""

    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        self.model_name = model_name or _answer.MODEL_NAME
        self.cache_dir = cache_dir or _answer.CACHE_DIR

    def extract_indicators_from_chunk(self, chunk: DocumentChunk, prompt_version: str = "v1", cache_only: bool = False) -> List[Indicator]:
        """Extract indicators from one chunk, using the on-disk cache when available.

        With cache_only=True no API call is made: a cache miss raises CacheMiss so the caller can count it.
        This lets the notebook rebuild its dataset from cached responses when the LLM server is down.
        """
        cached = _answer.load_cached_response(chunk, prompt_version, self.model_name)
        if cached is not None:
            raw = cached
        elif cache_only:
            raise _answer.CacheMiss(f"no cached response for chunk {chunk.doc_id}:{chunk.page_or_section} with model {self.model_name}")
        else:
            prompt = _build_extraction_prompt(chunk)
            text = _answer._call_openai_with_retries(prompt, self.model_name, response_format={"type": "json_object"})
            raw = _answer._extract_json_from_response(text)
            if raw is None or not isinstance(raw, dict):
                raw = {"indicators": []}
            # Small local models sometimes return a bare list or wrap it under another key.
            if "indicators" not in raw:
                lists = [v for v in raw.values() if isinstance(v, list)]
                raw = {"indicators": lists[0] if lists else []}
            try:
                _answer.cache_response_for_chunk(chunk, raw, prompt_version, self.model_name)
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


def run_novel_pass(*args, **kwargs):
    """Phase 3: run ``LLMClient`` on logframe chunks only and drop indicators already linked by the disclosure step."""
    raise NotImplementedError("Phase 3")
