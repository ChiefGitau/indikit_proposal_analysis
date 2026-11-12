"""Indicator-side query text and vectors (paper Section 3.4.1: query built from the metadata module).

Query text = name + English wording (template placeholders removed) + search terms (+ French wording when
present). ``use_search_terms=False`` drops the search terms (Phase 3 ablation). Vectors are pre-computed per
model and variant under ``outputs/kb/_queries/<model>/`` and also memoised in the embedding cache.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from pipeline import config
from pipeline.metadata.extensions import strip_placeholders
from pipeline.metadata.models import IndicatorMetadata
from pipeline.preprocess.embeddings import Embedder, EmbeddingCache, l2_normalise, model_slug


def build_query_text(ind: IndicatorMetadata, use_search_terms: bool = True) -> str:
    parts = [ind.name]
    wording = strip_placeholders(ind.kpi_wording_en or "")
    if wording and wording.lower() != ind.name.lower():
        parts.append(wording)
    if use_search_terms and ind.search_terms:
        terms = [t for t in ind.search_terms if t.lower() not in (ind.name.lower(), wording.lower())]
        if terms:
            parts.append("; ".join(terms))
    if ind.kpi_wording_fr:
        parts.append(strip_placeholders(ind.kpi_wording_fr))
    return "\n".join(p for p in parts if p)


def variant_name(use_search_terms: bool) -> str:
    return "with_terms" if use_search_terms else "no_terms"


def queries_path(model_name: str, use_search_terms: bool, out_dir: Path = config.QUERIES_DIR) -> Path:
    return Path(out_dir) / model_slug(model_name) / f"queries_{variant_name(use_search_terms)}.npz"


def precompute_queries(records: Iterable[IndicatorMetadata], embedder: Embedder, use_search_terms: bool = True, out_dir: Path = config.QUERIES_DIR, cache: Optional[EmbeddingCache] = None) -> Path:
    records = list(records)
    texts = [build_query_text(r, use_search_terms) for r in records]
    vectors = cache.embed(embedder, texts) if cache is not None else embedder.embed(texts)
    path = queries_path(embedder.name, use_search_terms, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), ids=np.array([r.indikit_id for r in records]), vectors=vectors)
    path.with_suffix(".json").write_text(json.dumps({"model": embedder.name, "variant": variant_name(use_search_terms), "n": len(records), "texts": dict(zip([r.indikit_id for r in records], texts))}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


class QueryStore:
    """Query vectors per indicator for one embedder; falls back to computing (and caching) on demand."""

    def __init__(self, embedder: Embedder, use_search_terms: bool = True, out_dir: Path = config.QUERIES_DIR, cache: Optional[EmbeddingCache] = None):
        self.embedder, self.use_search_terms, self.cache = embedder, use_search_terms, cache
        self._vectors: dict[str, np.ndarray] = {}
        path = queries_path(embedder.name, use_search_terms, out_dir)
        if path.exists():
            data = np.load(str(path))
            self._vectors = {str(i): v for i, v in zip(data["ids"], data["vectors"])}

    def vector(self, ind: IndicatorMetadata) -> np.ndarray:
        vec = self._vectors.get(ind.indikit_id)
        if vec is None:
            text = build_query_text(ind, self.use_search_terms)
            vec = (self.cache.embed(self.embedder, [text]) if self.cache is not None else self.embedder.embed([text]))[0]
            self._vectors[ind.indikit_id] = vec
        return l2_normalise(vec)[0]
