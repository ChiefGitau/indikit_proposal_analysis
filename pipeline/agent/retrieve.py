"""Retrieval of knowledge (paper Section 3.4.1): query construction, top-N search over the
text / outline / table knowledge bases, cross-encoder re-ranking to top-k, retrieval gate.

Filled in Phase 3. Phase 2 provides the knowledge bases this module searches.
"""
from __future__ import annotations

from typing import Any


def build_query(indicator: Any) -> str:
    """KPI wording + search terms (paper: metadata-driven query)."""
    raise NotImplementedError("Phase 3")


def retrieve_candidates(doc_kb: Any, query: str, top_n: int = 5) -> list[Any]:
    """Cosine search over the three knowledge bases; top-N from each."""
    raise NotImplementedError("Phase 3")


def rerank(query: str, candidates: list[Any], top_k: int = 4) -> list[Any]:
    """Cross-encoder re-ranking (paper used coROM)."""
    raise NotImplementedError("Phase 3")


def passes_gate(reranked: list[Any], threshold: float) -> bool:
    """Retrieval gate: skip pairs whose best re-rank score is below the threshold (docs/plan.md 4.3)."""
    raise NotImplementedError("Phase 3")
