"""Retrieval of knowledge (paper Section 3.4.1): query vector -> cosine top-N from each knowledge base ->
expansion (outline hit -> its section; table keyword / summary hit -> rows) -> de-duplication -> re-ranking ->
top-k evidence. Also the plain-RAG baseline retrieval and the pair-score table for the Phase 3 retrieval gate.

No LLM is involved. Embedders / re-rankers are injected (fakes in tests, real models from the CLI).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pipeline import config
from pipeline.agent.query import QueryStore, build_query_text
from pipeline.agent.rerank import LexicalReranker, Reranker, get_reranker
from pipeline.metadata.models import IndicatorMetadata
from pipeline.preprocess.embeddings import Embedder, EmbeddingCache, get_embedder
from pipeline.preprocess.knowledge_base import KBEntry, KnowledgeBase, load_kb

log = logging.getLogger(__name__)

MAX_RERANK_CANDIDATES = 30


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_n: int = config.RETRIEVAL_TOP_N
    top_k: int = config.RETRIEVAL_TOP_K
    kb_types: tuple[str, ...] = ("text", "outline", "table")
    use_search_terms: bool = True
    use_rerank: bool = True
    logframe_boost: float = config.LOGFRAME_BOOST
    small_table_rows: int = config.SMALL_TABLE_ROWS
    max_rerank_candidates: int = MAX_RERANK_CANDIDATES
    embedder: str = config.KB_EMBEDDING_MODEL
    reranker: str = config.RERANKER_MODEL

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:12]


class Evidence(BaseModel):
    entry_id: str
    kb_type: str
    entry_kind: str = "chunk"
    display_text: str
    location: str
    is_logframe: bool = False
    vector_score: float
    rerank_score: Optional[float] = None
    boost_applied: bool = False
    expanded_from: Optional[str] = None
    table_id: Optional[str] = None
    row_id: Optional[str] = None
    block_ids: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    doc_id: str
    indikit_id: str
    evidence: list[Evidence] = Field(default_factory=list)
    best_vector_score: float = 0.0
    best_rerank_score: Optional[float] = None
    n_candidates: int = 0
    config_fingerprint: str = ""
    query_text: str = ""


# --------------------------------------------------------------------------- #
# Shared model holders (so batch runs load models once)
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self, cfg: Optional[RetrievalConfig] = None, *, embedder: Optional[Embedder] = None, reranker: Optional[Reranker] = None,
                 cache: Optional[EmbeddingCache] = None, kb_dir: Path = config.KB_DIR, queries_dir: Path = config.QUERIES_DIR):
        self.cfg = cfg or RetrievalConfig()
        self.embedder = embedder or get_embedder(self.cfg.embedder)
        self._reranker = reranker
        self.cache = cache
        self.kb_dir = Path(kb_dir)
        self.queries_dir = Path(queries_dir)
        self._kbs: dict[str, KnowledgeBase] = {}
        self._stores: dict[bool, QueryStore] = {}

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = get_reranker(self.cfg.reranker) if self.cfg.use_rerank else LexicalReranker()
        return self._reranker

    def kb(self, doc_id: str) -> KnowledgeBase:
        if doc_id not in self._kbs:
            kb = load_kb(doc_id, self.kb_dir)
            if kb.embedding_model and kb.embedding_model != self.embedder.name:
                raise ValueError(f"KB {doc_id} was built with '{kb.embedding_model}' but the retriever uses '{self.embedder.name}'")
            if len(self._kbs) > 8:
                self._kbs.pop(next(iter(self._kbs)))
            self._kbs[doc_id] = kb
        return self._kbs[doc_id]

    def query_vector(self, indicator: IndicatorMetadata, use_search_terms: bool) -> np.ndarray:
        if use_search_terms not in self._stores:
            self._stores[use_search_terms] = QueryStore(self.embedder, use_search_terms, self.queries_dir, self.cache)
        return self._stores[use_search_terms].vector(indicator)

    def retrieve(self, doc_id: str, indicator: IndicatorMetadata, *, top_n: Optional[int] = None, top_k: Optional[int] = None, kb_types: Optional[tuple[str, ...]] = None,
                 use_search_terms: Optional[bool] = None, use_rerank: Optional[bool] = None) -> RetrievalResult:
        cfg = self.cfg.model_copy(update={k: v for k, v in dict(top_n=top_n, top_k=top_k, kb_types=kb_types, use_search_terms=use_search_terms, use_rerank=use_rerank).items() if v is not None})
        kb = self.kb(doc_id)
        qvec = self.query_vector(indicator, cfg.use_search_terms)
        query_text = build_query_text(indicator, cfg.use_search_terms)

        # 1. top-N per knowledge base
        hits: list[tuple[KBEntry, float]] = []
        for kb_type in cfg.kb_types:
            hits.extend(kb.search(kb_type, qvec, cfg.top_n))
        # 2. expand and de-duplicate by entry id (keep the best vector score)
        cands: dict[str, Evidence] = {}
        for entry, score in hits:
            for ex in kb.expand(entry, cfg.small_table_rows):
                boosted = score + (cfg.logframe_boost if ex.is_logframe and cfg.logframe_boost else 0.0)
                prev = cands.get(ex.entry_id)
                if prev is None or boosted > prev.vector_score:
                    cands[ex.entry_id] = Evidence(
                        entry_id=ex.entry_id, kb_type=ex.kb_type, entry_kind=ex.entry_kind, display_text=ex.display_text, location=ex.location, is_logframe=ex.is_logframe,
                        vector_score=round(boosted, 6), boost_applied=bool(ex.is_logframe and cfg.logframe_boost), expanded_from=None if ex.entry_id == entry.entry_id else entry.entry_id,
                        table_id=ex.payload_ref.get("table_id"), row_id=ex.payload_ref.get("row_id"), block_ids=list(ex.payload_ref.get("block_ids", [])),
                    )
        ordered = sorted(cands.values(), key=lambda e: (-e.vector_score, e.entry_id))
        n_candidates = len(ordered)
        # 3. re-rank (bounded number of candidates) and cut to top-k
        pool = ordered[: cfg.max_rerank_candidates]
        if pool and cfg.use_rerank:
            scores = self.reranker.score(query_text, [e.display_text for e in pool])
            for e, s in zip(pool, scores):
                e.rerank_score = round(float(s), 6)
            pool.sort(key=lambda e: (-(e.rerank_score or 0.0), -e.vector_score, e.entry_id))
        evidence = pool[: cfg.top_k]
        return RetrievalResult(
            doc_id=doc_id, indikit_id=indicator.indikit_id, evidence=evidence,
            best_vector_score=max((e.vector_score for e in ordered), default=0.0),
            best_rerank_score=max((e.rerank_score for e in pool if e.rerank_score is not None), default=None),
            n_candidates=n_candidates, config_fingerprint=cfg.fingerprint() + ":" + self.embedder.name + ":" + (self.reranker.name if cfg.use_rerank else "none"), query_text=query_text,
        )


_DEFAULT_RETRIEVER: Optional[Retriever] = None


def _default_retriever(cfg: Optional[RetrievalConfig]) -> Retriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None or (cfg is not None and cfg.embedder != _DEFAULT_RETRIEVER.cfg.embedder):
        _DEFAULT_RETRIEVER = Retriever(cfg, cache=EmbeddingCache())
    return _DEFAULT_RETRIEVER


def retrieve(
    doc_id: str,
    indicator: IndicatorMetadata,
    *,
    top_n: int = config.RETRIEVAL_TOP_N,
    top_k: int = config.RETRIEVAL_TOP_K,
    kb_types: tuple[str, ...] = ("text", "outline", "table"),
    use_search_terms: bool = True,
    use_rerank: bool = True,
    config: Optional[RetrievalConfig] = None,   # noqa: A002  (name required by the Phase 2 brief)
    retriever: Optional[Retriever] = None,
) -> RetrievalResult:
    """Ranked evidence for one (document, indicator) pair (Phase 3 interface). Loads real models unless a
    ``retriever`` with injected embedder / re-ranker is given."""
    r = retriever or _default_retriever(config)
    return r.retrieve(doc_id, indicator, top_n=top_n, top_k=top_k, kb_types=kb_types, use_search_terms=use_search_terms, use_rerank=use_rerank)


# --------------------------------------------------------------------------- #
# Baseline (plain RAG) retrieval
# --------------------------------------------------------------------------- #
def retrieve_baseline(doc_id: str, indicator: IndicatorMetadata, top_k: int = config.RETRIEVAL_TOP_K, *, embedder: Optional[Embedder] = None, cache: Optional[EmbeddingCache] = None,
                      kb_dir: Path = config.KB_BASELINE_DIR) -> RetrievalResult:
    """Fixed-size chunks, English MiniLM, no search terms, no re-ranking (paper's Benchmark)."""
    from pipeline.preprocess.baseline_kb import load_baseline_kb

    embedder = embedder or get_embedder(config.BASELINE_EMBEDDING_MODEL)
    bkb = load_baseline_kb(doc_id, kb_dir)
    if bkb.embedding_model and bkb.embedding_model != embedder.name:
        raise ValueError(f"baseline KB {doc_id} was built with '{bkb.embedding_model}' but got '{embedder.name}'")
    query_text = build_query_text(indicator, use_search_terms=False)
    qvec = (cache.embed(embedder, [query_text]) if cache is not None else embedder.embed([query_text]))[0]
    hits = bkb.search(qvec, top_k)
    evidence = [Evidence(entry_id=c.entry_id, kb_type="text", entry_kind="chunk", display_text=c.display_text, location=c.location, vector_score=round(s, 6)) for c, s in hits]
    return RetrievalResult(doc_id=doc_id, indikit_id=indicator.indikit_id, evidence=evidence, best_vector_score=max((e.vector_score for e in evidence), default=0.0), n_candidates=len(hits),
                           config_fingerprint=f"baseline:{embedder.name}", query_text=query_text)


# --------------------------------------------------------------------------- #
# Pair scores for the retrieval gate
# --------------------------------------------------------------------------- #
PAIR_SCORE_COLUMNS = ("doc_id", "indikit_id", "best_vector_score", "best_rerank_score", "top_entry_id", "top_is_logframe", "top_kb_type")


def score_pairs(retriever: Retriever, doc_ids: Iterable[str], indicators: Iterable[IndicatorMetadata], *, use_rerank: bool = True, progress: bool = True) -> tuple[list[dict], dict]:
    doc_ids, indicators = list(doc_ids), list(indicators)
    rows: list[dict] = []
    t0 = time.time()
    for i, doc_id in enumerate(doc_ids, start=1):
        td = time.time()
        try:
            retriever.kb(doc_id)
        except Exception as exc:
            log.warning("skip %s: %s", doc_id, exc)
            continue
        for ind in indicators:
            res = retriever.retrieve(doc_id, ind, use_rerank=use_rerank, top_k=1)
            top = res.evidence[0] if res.evidence else None
            rows.append({"doc_id": doc_id, "indikit_id": ind.indikit_id, "best_vector_score": round(res.best_vector_score, 6), "best_rerank_score": None if res.best_rerank_score is None else round(res.best_rerank_score, 6),
                         "top_entry_id": top.entry_id if top else "", "top_is_logframe": bool(top.is_logframe) if top else False, "top_kb_type": top.kb_type if top else ""})
        if progress:
            print(f"[{i}/{len(doc_ids)}] {doc_id} {time.time() - td:.1f}s", flush=True)
    stats = {"documents": len(doc_ids), "indicators": len(indicators), "pairs": len(rows), "seconds": round(time.time() - t0, 1), "use_rerank": use_rerank, "embedder": retriever.embedder.name, "reranker": retriever.reranker.name if use_rerank else "none"}
    return rows, stats


def write_pair_scores(rows: list[dict], path: Path = config.PAIR_SCORES_PATH) -> Path:
    """Parquet when pyarrow is available, else CSV with the same columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        import pandas as pd

        pd.DataFrame(rows, columns=list(PAIR_SCORE_COLUMNS)).to_parquet(path, index=False)
        return path
    except Exception:
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(PAIR_SCORE_COLUMNS))
            w.writeheader()
            w.writerows(rows)
        return csv_path
