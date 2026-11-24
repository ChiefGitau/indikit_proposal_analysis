"""Re-rankers (paper Section 3.4.1: cross-encoder re-ranking of the retrieved candidates).

``Reranker.score(query, passages) -> list[float]`` (higher is better). Implementations:

- ``LexicalReranker``: deterministic token / bigram overlap (tests, and the ordering used with ``use_rerank=False``);
- ``FastembedReranker``: ONNX cross-encoders through fastembed (default ``jinaai/jina-reranker-v2-base-multilingual``);
- ``CrossEncoderReranker``: sentence-transformers cross-encoders (e.g. ``BAAI/bge-reranker-v2-m3``, needs PyTorch).
"""
from __future__ import annotations

import math
import re
import threading
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from pipeline import config

_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9%#]+")


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class LexicalReranker(Reranker):
    """Cosine-like overlap of unigrams plus a bigram bonus; stable and dependency-free."""

    name = "lexical"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = self._tokens(query)
        q_set, q_bi = set(q), set(zip(q, q[1:]))
        out: list[float] = []
        for p in passages:
            t = self._tokens(p)
            if not q_set or not t:
                out.append(0.0)
                continue
            t_set, t_bi = set(t), set(zip(t, t[1:]))
            uni = len(q_set & t_set) / math.sqrt(len(q_set) * len(t_set))
            bi = len(q_bi & t_bi) / max(1, len(q_bi))
            out.append(round(uni + 0.5 * bi, 6))
        return out


class FastembedReranker(Reranker):
    def __init__(self, model_name: str = config.RERANKER_MODEL, batch_size: int = config.RERANK_BATCH_SIZE):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self.batch_size = batch_size
        self._lock = threading.Lock()   # one ONNX call at a time: bounds peak memory when several workers retrieve
        # arena off + capped threads: see config.ONNX_CPU_MEM_ARENA / ONNX_THREADS (memory growth in long runs)
        self._model = TextCrossEncoder(model_name, threads=config.ONNX_THREADS, enable_cpu_mem_arena=config.ONNX_CPU_MEM_ARENA)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        cut = config.RERANK_MAX_PASSAGE_CHARS
        passages = [p[:cut] if cut and len(p) > cut else p for p in passages]
        with self._lock:
            return [float(s) for s in self._model.rerank(query, list(passages), batch_size=self.batch_size)]


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.predict([(query, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False)]


def get_reranker(spec: Optional[str] = None) -> Reranker:
    """``"lexical"`` / ``"fastembed:<model>"`` / ``"st:<model>"`` / ``"<model>"`` (fastembed first, then sentence-transformers)."""
    spec = spec or config.RERANKER_MODEL
    if spec == "lexical":
        return LexicalReranker()
    if spec.startswith("fastembed:"):
        return FastembedReranker(spec.split(":", 1)[1])
    if spec.startswith("st:"):
        return CrossEncoderReranker(spec.split(":", 1)[1])
    errors = []
    for cls in (FastembedReranker, CrossEncoderReranker):
        try:
            return cls(spec)
        except Exception as exc:
            errors.append(f"{cls.__name__}: {type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError(f"no re-ranker backend could load '{spec}': " + "; ".join(errors))
