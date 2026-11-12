"""Consolidation and reverse check (docs/phase_3.md Task 5B / 5C).

B. Link each disclosure match to an extracted document indicator by location (same table row / block) and text
   similarity; a match without counterpart creates a document indicator with source ``disclosure_only``. When a
   document indicator has several IndiKit matches, the primary is verbatim over close, then the higher re-rank
   score; the others are ``alternate_ids``. Document indicators without a match are novel candidates.
C. Reverse check: embed each novel candidate, take the top-N IndiKit indicators by similarity and ask the LLM once
   whether any is a verbatim / close match. Hits become matches with ``source = reverse_check``; what remains is
   novel (``match_type = none``). The number recovered measures what the gate / retrieval missed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from pipeline import config
from pipeline.agent.answer import text_similarity
from pipeline.agent.llm import BudgetExceeded, LLMProvider, ResponseCache, cache_key, model_role
from pipeline.agent.models import ConsolidatedIndicator, DocIndicator, LinkedMatch, PairAnswer, ReverseCheckAnswer, ValidatedMatch
from pipeline.agent.prompts import build_reverse_check_messages
from pipeline.agent.query import build_query_text
from pipeline.metadata.models import IndicatorMetadata
from pipeline.preprocess.embeddings import Embedder, EmbeddingCache, l2_normalise


def _linked(vm: ValidatedMatch, name: str) -> LinkedMatch:
    return LinkedMatch(indikit_id=vm.indikit_id, indikit_name=name, match_type=vm.match_type, source=vm.source, rerank_score=vm.rerank_score, reason=vm.reason, evidence_quote=vm.evidence_quote,
                       baseline=vm.baseline, target=vm.target, unit=vm.unit, text_similarity=vm.text_similarity)


def find_counterpart(vm: ValidatedMatch, indicators: Sequence[DocIndicator], threshold: int = config.LINK_TEXT_SIMILARITY) -> Optional[DocIndicator]:
    """Same table row, else overlapping blocks, else best text similarity >= threshold."""
    best: Optional[DocIndicator] = None
    best_score = -1.0
    for d in indicators:
        same_row = vm.row_id is not None and d.row_id == vm.row_id
        same_block = bool(set(vm.block_ids) & set(d.block_ids)) if vm.block_ids else False
        sim = text_similarity(vm.document_indicator_text, d.indicator_text)
        if same_row or same_block:
            score = 1000 + sim
        elif sim >= threshold:
            score = sim
        else:
            continue
        if score > best_score:
            best, best_score = d, score
    return best


def choose_primary(matches: list[LinkedMatch]) -> tuple[Optional[LinkedMatch], list[str]]:
    if not matches:
        return None, []
    ranked = sorted(matches, key=lambda m: (0 if m.match_type == "verbatim" else 1, -(m.rerank_score if m.rerank_score is not None else -1e9), 0 if m.source == "disclosure" else 1, m.indikit_id))
    return ranked[0], [m.indikit_id for m in ranked[1:]]


def consolidate(doc_id: str, answers: Sequence[PairAnswer], indicators: Sequence[DocIndicator], metadata: dict[str, IndicatorMetadata], *, link_threshold: int = config.LINK_TEXT_SIMILARITY) -> list[ConsolidatedIndicator]:
    """One ``ConsolidatedIndicator`` per document indicator (extracted ones first, then disclosure-only ones)."""
    inds = [d.model_copy(deep=True) for d in indicators]
    by_id: dict[str, ConsolidatedIndicator] = {d.doc_indicator_id: ConsolidatedIndicator(indicator=d) for d in inds}
    extra: list[ConsolidatedIndicator] = []
    next_n = len(inds)
    for pa in answers:
        for vm in pa.matches:
            name = metadata[vm.indikit_id].name if vm.indikit_id in metadata else ""
            lm = _linked(vm, name)
            target = find_counterpart(vm, inds, link_threshold)
            if target is None:
                # maybe an earlier disclosure-only indicator already covers this text
                for c in extra:
                    if text_similarity(vm.document_indicator_text, c.indicator.indicator_text) >= 90:
                        target_c = c
                        break
                else:
                    next_n += 1
                    d = DocIndicator(doc_indicator_id=f"D{next_n:03d}", doc_id=doc_id, indicator_text=vm.document_indicator_text, level=vm.level, baseline=vm.baseline, target=vm.target, unit=vm.unit,
                                     location=vm.location, table_id=vm.table_id, row_id=vm.row_id, block_ids=list(vm.block_ids), source="disclosure_only", extraction_method="", order=10000 + next_n)
                    target_c = ConsolidatedIndicator(indicator=d)
                    extra.append(target_c)
            else:
                target_c = by_id[target.doc_indicator_id]
            if not any(m.indikit_id == lm.indikit_id and m.source == lm.source for m in target_c.matches):
                target_c.matches.append(lm)
                # fill values the extraction did not have
                for f in ("baseline", "target", "unit"):
                    if getattr(target_c.indicator, f) is None and getattr(vm, f) is not None:
                        setattr(target_c.indicator, f, getattr(vm, f))
    out = list(by_id.values()) + extra
    for c in out:
        c.primary, c.alternate_ids = choose_primary(c.matches)
    return out


# --------------------------------------------------------------------------- #
# Reverse check
# --------------------------------------------------------------------------- #
@dataclass
class ReverseCheckStats:
    candidates: int = 0
    checked: int = 0
    recovered: int = 0
    calls: int = 0
    cached: int = 0
    failed: int = 0
    budget_stop: Optional[str] = None
    errors: list[str] = field(default_factory=list)


class IndicatorIndex:
    """Vectors of all IndiKit indicators (query text, with search terms) for the nearest-neighbour step."""

    def __init__(self, metadata: dict[str, IndicatorMetadata], embedder: Embedder, cache: Optional[EmbeddingCache] = None):
        self.ids = sorted(metadata)
        self.metadata = metadata
        self.embedder = embedder
        self.cache = cache
        texts = [build_query_text(metadata[i], use_search_terms=True) for i in self.ids]
        vecs = cache.embed(embedder, texts) if cache is not None else embedder.embed(texts)
        self.vectors = l2_normalise(np.asarray(vecs, dtype=np.float32))

    def nearest(self, texts: Sequence[str], top_n: int = config.REVERSE_CHECK_TOP) -> list[list[tuple[str, float]]]:
        if not texts:
            return []
        q = self.cache.embed(self.embedder, list(texts)) if self.cache is not None else self.embedder.embed(list(texts))
        q = l2_normalise(np.asarray(q, dtype=np.float32))
        sims = q @ self.vectors.T
        out = []
        for row in sims:
            idx = np.argsort(-row)[:top_n]
            out.append([(self.ids[i], round(float(row[i]), 4)) for i in idx])
        return out


def reverse_check(doc_id: str, consolidated: list[ConsolidatedIndicator], index: IndicatorIndex, *, provider: Optional[LLMProvider], model: Optional[str] = None, cache: Optional[ResponseCache] = None,
                  top_n: int = config.REVERSE_CHECK_TOP, doc_sha256: Optional[str] = None, force: bool = False, only_indicators: bool = True) -> ReverseCheckStats:
    """Ask once per novel candidate whether one of its nearest IndiKit indicators matches; mutates ``consolidated``."""
    model = model or model_role("ANSWER_MODEL")
    stats = ReverseCheckStats()
    cands = [c for c in consolidated if not c.matches and (c.indicator.is_indicator or not only_indicators)]
    stats.candidates = len(cands)
    if not cands:
        return stats
    nn = index.nearest([c.indicator.indicator_text for c in cands], top_n)
    for c, neighbours in zip(cands, nn):
        c.nearest_indikit = neighbours
    if provider is None:
        return stats
    for c in cands:
        candidates = [index.metadata[i] for i, _ in c.nearest_indikit]
        system, user = build_reverse_check_messages(c.indicator.indicator_text, candidates, context=c.indicator.result_context or "")
        key = cache_key(doc_sha256=doc_sha256 or doc_id, text=c.indicator.indicator_text, candidates=",".join(i for i, _ in c.nearest_indikit), prompt_version=config.REVERSE_CHECK_PROMPT_VERSION,
                        rubric_version=config.RUBRIC_VERSION, model=model)
        answer: Optional[ReverseCheckAnswer] = None
        cached = cache.get(key) if (cache is not None and not force) else None
        if cached is not None and isinstance(cached.get("response"), dict):
            try:
                answer = ReverseCheckAnswer.model_validate(cached["response"])
                stats.cached += 1
            except Exception:
                answer = None
        if answer is None:
            try:
                res = provider.complete_json(system, user, ReverseCheckAnswer, model=model)
            except BudgetExceeded as exc:
                stats.budget_stop = str(exc)
                break
            stats.calls += 1
            if not res.ok:
                stats.failed += 1
                stats.errors.append(f"{c.indicator.doc_indicator_id}: {res.error}")
                continue
            answer = res.parsed
            if cache is not None:
                cache.put(key, answer.model_dump(mode="json"), doc_id=doc_id, model=model, prompt_version=config.REVERSE_CHECK_PROMPT_VERSION)
        stats.checked += 1
        allowed = {i for i, _ in c.nearest_indikit}
        sims = dict(c.nearest_indikit)
        for m in answer.matches:
            if m.indikit_id not in allowed:
                continue   # the model may only pick from the candidates it was shown
            name = index.metadata[m.indikit_id].name
            c.matches.append(LinkedMatch(indikit_id=m.indikit_id, indikit_name=name, match_type=m.match_type, source="reverse_check", rerank_score=sims.get(m.indikit_id), reason=m.reason,
                                         evidence_quote=c.indicator.indicator_text, baseline=c.indicator.baseline, target=c.indicator.target, unit=c.indicator.unit,
                                         text_similarity=round(text_similarity(c.indicator.indicator_text, index.metadata[m.indikit_id].kpi_wording_en or name), 1)))
        if c.matches:
            stats.recovered += 1
            c.primary, c.alternate_ids = choose_primary(c.matches)
    return stats
