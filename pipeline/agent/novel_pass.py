"""Novel-indicator pass (docs/plan.md Section 4.3, docs/phase_3.md Task 5A; an addition beyond the paper).

The standard-driven disclosure step cannot see document indicators that have no IndiKit counterpart, so the
document's own indicators are extracted from its logframe content only (``logframe_content(doc_id)``):

1. logframe **tables with an indicator column**: rows are extracted **in code** (no LLM); cells holding several
   indicators are split on bullets / numbering / line breaks; baseline / target / MoV come from the mapped columns;
2. logframe **text sections** and **tables whose columns could not be mapped**: one schema-enforced LLM call per
   section / table (``EXTRACTION_MODEL``), with the rubric's "not an indicator" flag in the same call;
3. de-duplication within the document (``deduplicate_indicators``) and stable ``doc_indicator_id``s.

The reusable parts moved from ``extraction_utils.py`` (chunk scoring, dedupe, the legacy ``LLMClient``) stay here.
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




# ====================================================================================================== #
# Phase 3: document-driven extraction of the document's own indicators
# ====================================================================================================== #
import hashlib
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path as _Path

from pipeline import config as _config
from pipeline.agent.llm import BudgetExceeded, LLMProvider, ResponseCache, cache_key, model_role
from pipeline.agent.models import DocIndicator, ExtractionAnswer
from pipeline.agent.prompts import build_extraction_messages
from pipeline.preprocess.models import Block, Table, TableRow

_SPLIT_MARK_RE = re.compile(r"(?:^|\n)\s*(?:[-•▪●◦*]|\(?[0-9]{1,2}[.)]|[a-z][.)]|[ivx]{1,4}[.)]|(?:indicator|indicateur)\s*[0-9.]+\s*[:.)-])\s+", re.IGNORECASE)
_INDICATOR_START_RE = re.compile(r"^\s*(?:#|%|no\.?|nr\.?|number of|nombre de|proportion|percentage|pourcentage|taux|share of|average|moyenne|score|ratio|[0-9]{1,3}\s*%|level of|degree of|extent to which|existence of)", re.IGNORECASE)
_ACTIVITY_START_RE = re.compile(r"^\s*(?:to\s+)?(?:train|conduct|organi[sz]e|distribute|establish|provide|support|carry out|implement|facilitate|hold|develop|construct|rehabilitate|procure|recruit|former|organiser|distribuer|mettre en place|r[ée]aliser|appuyer|construire|r[ée]habiliter|fournir)\b", re.IGNORECASE)
_RESULT_PREFIX_RE = re.compile(r"^\s*(?:(?:overall|specific)\s+objective|outcome|output|impact|result|r[ée]sultat|objectif|goal|purpose|so|io|op|r)\s*[0-9.]*\s*[:.)-]\s*", re.IGNORECASE)
_HEADER_WORD_RE = re.compile(r"(?:indicat(?:or|eur)s?|ovi|iov|objectively\s+verifiable\s+indicators?|indicateurs?\s+objectivement\s+v[ée]rifiables?)\s*[:.]?", re.IGNORECASE)
MAX_SECTION_CHARS = 7000


def split_indicator_cell(text: str) -> list[str]:
    """Split a cell holding several indicators (bullets, numbering, line breaks) into separate texts."""
    text = (text or "").replace("\r", "\n").strip()
    if not text:
        return []
    raw_parts = _SPLIT_MARK_RE.split("\n" + text)
    marked = len(raw_parts) > 1
    parts = [re.sub(r"\s+", " ", p.strip(" \t-•▪●◦*;:,.")).strip() for p in raw_parts if p and p.strip()]
    if len(parts) <= 1 and not marked:
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.split("\n") if l.strip()]
        parts = lines if len(lines) > 1 and all(len(l) > 8 for l in lines) else [re.sub(r"\s+", " ", text)]
    out: list[str] = []
    for p in parts:
        p = _RESULT_PREFIX_RE.sub("", p) if _INDICATOR_START_RE.search(_RESULT_PREFIX_RE.sub("", p)) else p
        if len(p) >= 6 and not re.fullmatch(r"[0-9.,% ]+", p):
            out.append(p)
    return out


def looks_like_activity(text: str) -> bool:
    """Cheap rubric proxy for code-extracted rows: verb-first lines are activities, measure-first lines are indicators."""
    t = (text or "").strip()
    if _INDICATOR_START_RE.search(t):
        return False
    return bool(_ACTIVITY_START_RE.search(t))


def _roles(table: Table) -> dict[str, list[int]]:
    roles: dict[str, list[int]] = {}
    for i, r in enumerate(table.column_roles or []):
        if r:
            roles.setdefault(r, []).append(i)
    return roles


def _cell(row: TableRow, table: Table, col: int) -> str:
    """Cell by column index. ``row.cells`` is keyed by the de-duplicated header names (and JSON files store the keys
    sorted), so look the value up by header name rather than by dict position."""
    from pipeline.preprocess.logframe import dedupe_headers

    headers = dedupe_headers(table.header)
    if col >= len(headers):
        return ""
    return (row.cells.get(headers[col]) or "").strip()


def _join_cells(row: TableRow, table: Table, cols: list[int]) -> Optional[str]:
    vals = [v for v in (_cell(row, table, c) for c in cols) if v]
    return " / ".join(dict.fromkeys(vals)) or None


def _level_from_context(context: str) -> str:
    c = (context or "").lower()
    for lvl in ("impact", "outcome", "output"):
        if re.search(rf"\b{lvl}", c):
            return lvl
    if re.search(r"overall objective|objectif g[ée]n[ée]ral|goal", c):
        return "impact"
    if re.search(r"specific objective|objectif sp[ée]cifique|purpose", c):
        return "outcome"
    if re.search(r"\br[ée]sultat|\bresult\b", c):
        return "output"
    return "unclear"


def table_location_label(table: Table) -> str:
    from pipeline.preprocess.knowledge_base import table_location, table_title

    return f"{table_location(table)} | {table_title(table)}"


def extract_rows_in_code(doc_id: str, content: dict) -> tuple[list[DocIndicator], list[dict]]:
    """Code-first extraction from logframe tables with an ``indicator`` column. Returns (indicators, unmapped tables)."""
    out: list[DocIndicator] = []
    unmapped: list[dict] = []
    order = 0
    for item in content.get("tables", []):
        table: Table = item["table"]
        rows: list[TableRow] = item["rows"]
        roles = _roles(table)
        if "indicator" not in roles:
            unmapped.append(item)
            continue
        ind_cols = roles["indicator"]
        header_cells = {h.strip().lower() for h in table.header if h and h.strip()}
        for row in rows:
            cell_text = "\n".join(v for v in (_cell(row, table, c) for c in ind_cols) if v)
            for text in split_indicator_cell(cell_text):
                if text.strip().lower() in header_cells or _HEADER_WORD_RE.fullmatch(text.strip()):
                    continue   # repeated header row inside the table body
                order += 1
                ctx = row.result_context or _join_cells(row, table, roles.get("result", [])) or None
                out.append(DocIndicator(doc_indicator_id="", doc_id=doc_id, indicator_text=text, level=_level_from_context(ctx or ""), baseline=_join_cells(row, table, roles.get("baseline", [])),
                                        target=_join_cells(row, table, roles.get("target", [])), means_of_verification=_join_cells(row, table, roles.get("mov", [])), result_context=ctx,
                                        location=table_location_label(table), table_id=table.table_id, row_id=row.row_id, source="extraction", extraction_method="table_code",
                                        is_indicator=not looks_like_activity(text), order=order))
    return out, unmapped


def text_sections(content: dict, max_chars: int = MAX_SECTION_CHARS) -> list[dict]:
    """Logframe text blocks grouped by their section heading (batched by section, not by paragraph)."""
    sections: list[dict] = []
    cur: Optional[dict] = None
    for b in content.get("blocks", []):
        if b.kind == "table":
            continue
        title = b.text if b.kind == "heading" else (b.heading_path[-1] if b.heading_path else "")
        if b.kind == "heading" or cur is None or cur["title"] != title or len(cur["text"]) + len(b.text) > max_chars:
            cur = {"title": title, "text": "" if b.kind == "heading" else b.text, "block_ids": [b.block_id], "location": f"p.{b.page}" if b.page else (" > ".join(b.heading_path[-2:]) or "document")}
            sections.append(cur)
            continue
        cur["text"] += "\n" + b.text
        cur["block_ids"].append(b.block_id)
    return [s for s in sections if s["text"].strip() and len(s["text"].strip()) > 20]


def table_as_text(table: Table, rows: list[TableRow], max_chars: int = MAX_SECTION_CHARS) -> str:
    lines = [" | ".join(table.header)]
    for r in rows:
        lines.append(" | ".join(v.strip() for v in r.cells.values()))
    text = "\n".join(lines)
    return text[:max_chars]


def legacy_cache_lookup(doc_id: str, text: str, prompt_version: str, model: str) -> Optional[dict]:
    """The pre-Phase-3 ``outputs/api_cache`` may be reused only when chunk content **and** prompt version are identical
    to what produced it (legacy per-chunk prompt v1). The Phase 3 prompt is a different version, so this returns None
    unless the caller is running the legacy prompt."""
    if prompt_version != _config.EXTRACTION_PROMPT_VERSION:
        return None
    chunk = DocumentChunk(doc_id=doc_id, source_file="", chunk_type="text", page_or_section=None, content=text, char_count=len(text))
    return _answer.load_cached_response(chunk, prompt_version, model)


@_dataclass
class NovelPassResult:
    doc_id: str
    indicators: list[DocIndicator] = _field(default_factory=list)
    stats: dict = _field(default_factory=dict)
    errors: list[str] = _field(default_factory=list)
    budget_stop: Optional[str] = None


def extract_with_llm(doc_id: str, units: list[dict], *, provider: LLMProvider, model: str, cache: Optional[ResponseCache], doc_sha256: str, kind: str, force: bool = False) -> tuple[list[DocIndicator], dict, list[str], Optional[str]]:
    """One schema-enforced call per section / table. ``units``: dicts with title, text, location, block_ids / table_id."""
    out: list[DocIndicator] = []
    stats = {"calls": 0, "cached": 0, "failed": 0, "filtered_activities": 0}
    errors: list[str] = []
    budget_stop = None
    for u in units:
        system, user = build_extraction_messages(u["text"], kind=kind, title=u.get("title", ""))
        key = cache_key(doc_sha256=doc_sha256, unit=hashlib.sha256(u["text"].encode("utf-8")).hexdigest()[:16], prompt_version=_config.NOVEL_PROMPT_VERSION, rubric_version=_config.RUBRIC_VERSION, model=model, kind=kind)
        answer: Optional[ExtractionAnswer] = None
        cached = cache.get(key) if (cache is not None and not force) else None
        if cached is not None and isinstance(cached.get("response"), dict):
            try:
                answer = ExtractionAnswer.model_validate(cached["response"])
                stats["cached"] += 1
            except Exception:
                answer = None
        if answer is None:
            try:
                res = provider.complete_json(system, user, ExtractionAnswer, model=model)
            except BudgetExceeded as exc:
                budget_stop = str(exc)
                break
            stats["calls"] += 1
            if not res.ok:
                stats["failed"] += 1
                errors.append(f"{kind} {u.get('location', '')}: {res.error}")
                continue
            answer = res.parsed
            if cache is not None:
                cache.put(key, answer.model_dump(mode="json"), doc_id=doc_id, model=model, prompt_version=_config.NOVEL_PROMPT_VERSION, kind=kind)
        for i, e in enumerate(answer.indicators, start=1):
            if not e.indicator_text:
                continue
            if not e.is_indicator:
                stats["filtered_activities"] += 1
            out.append(DocIndicator(doc_indicator_id="", doc_id=doc_id, indicator_text=e.indicator_text, level=e.level, baseline=e.baseline, target=e.target, unit=e.unit, means_of_verification=e.means_of_verification,
                                    result_context=e.result_context, location=u.get("location", ""), table_id=u.get("table_id"), block_ids=list(u.get("block_ids", [])), source="extraction",
                                    extraction_method=f"{kind}_llm", is_indicator=e.is_indicator, order=u.get("order", 0) * 1000 + i))
    return out, stats, errors, budget_stop


def dedupe_doc_indicators(indicators: list[DocIndicator]) -> tuple[list[DocIndicator], list[tuple[int, int, int]]]:
    """Reuse ``deduplicate_indicators`` (token-sort ratio >= 90). Code-extracted rows win over LLM rows: sort them first."""
    ordered = sorted(indicators, key=lambda d: (0 if d.extraction_method == "table_code" else 1, d.order))
    legacy = [Indicator(indicator_text=d.indicator_text) for d in ordered]
    kept, merged = deduplicate_indicators(legacy)
    kept_texts = {id(k) for k in kept}
    kept_docs = [d for d, l in zip(ordered, legacy) if id(l) in kept_texts]
    # merge values from dropped duplicates into the kept record when the kept one lacks them
    for kidx, ridx, _score in merged:
        k, r = kept_docs[kidx], ordered[ridx]
        for f in ("baseline", "target", "unit", "means_of_verification", "result_context"):
            if getattr(k, f) is None and getattr(r, f) is not None:
                setattr(k, f, getattr(r, f))
    return kept_docs, merged


def assign_ids(indicators: list[DocIndicator]) -> list[DocIndicator]:
    """Stable ids in reading order: D001, D002, ... (tables first in table order, then text sections)."""
    ordered = sorted(indicators, key=lambda d: (0 if d.table_id else 1, d.location, d.order, d.indicator_text))
    for i, d in enumerate(ordered, start=1):
        d.doc_indicator_id = f"D{i:03d}"
    return ordered


def run_novel_pass(doc_id: str, *, provider: Optional[LLMProvider] = None, model: Optional[str] = None, cache: Optional[ResponseCache] = None, kb_dir: _Path = _config.KB_DIR, content: Optional[dict] = None,
                   use_llm: bool = True, force: bool = False, doc_sha256: Optional[str] = None) -> NovelPassResult:
    """Extract the document's own indicators from its logframe content (Task 5A)."""
    from pipeline.preprocess.logframe import logframe_content

    content = content if content is not None else logframe_content(doc_id, kb_dir)
    doc_sha256 = doc_sha256 or _answer._doc_sha(doc_id, kb_dir)
    model = model or model_role("EXTRACTION_MODEL")
    result = NovelPassResult(doc_id=doc_id)
    code_rows, unmapped = extract_rows_in_code(doc_id, content)
    stats = {"table_code": len(code_rows), "table_code_activities": sum(1 for d in code_rows if not d.is_indicator), "unmapped_tables": len(unmapped), "text_sections": 0, "table_llm": 0, "text_llm": 0,
             "llm_calls": 0, "llm_cached": 0, "llm_failed": 0, "filtered_activities": 0, "deduped": 0}
    all_inds = list(code_rows)
    sections = text_sections(content)
    stats["text_sections"] = len(sections)
    if use_llm and (sections or unmapped) and provider is not None:
        units = [{**s, "order": i + 1} for i, s in enumerate(sections)]
        t_inds, t_stats, t_err, stop = extract_with_llm(doc_id, units, provider=provider, model=model, cache=cache, doc_sha256=doc_sha256, kind="text", force=force)
        all_inds += t_inds
        stats["text_llm"] = len(t_inds)
        for k in ("calls", "cached", "failed", "filtered_activities"):
            stats["llm_" + k if k != "filtered_activities" else k] += t_stats[k]
        result.errors += t_err
        if stop:
            result.budget_stop = stop
        else:
            tunits = [{"title": table_location_label(it["table"]), "text": table_as_text(it["table"], it["rows"]), "location": table_location_label(it["table"]), "table_id": it["table"].table_id, "order": 100 + i}
                      for i, it in enumerate(unmapped)]
            u_inds, u_stats, u_err, stop = extract_with_llm(doc_id, tunits, provider=provider, model=model, cache=cache, doc_sha256=doc_sha256, kind="table", force=force)
            all_inds += u_inds
            stats["table_llm"] = len(u_inds)
            for k in ("calls", "cached", "failed", "filtered_activities"):
                stats["llm_" + k if k != "filtered_activities" else k] += u_stats[k]
            result.errors += u_err
            result.budget_stop = stop
    elif (sections or unmapped) and provider is None:
        result.errors.append(f"{len(sections)} text section(s) and {len(unmapped)} unmapped table(s) need the LLM; no provider given")
    kept, merged = dedupe_doc_indicators([d for d in all_inds if d.indicator_text])
    stats["deduped"] = len(merged)
    result.indicators = assign_ids(kept)
    result.stats = stats
    return result
