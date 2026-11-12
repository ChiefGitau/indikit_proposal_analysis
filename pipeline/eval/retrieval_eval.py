"""Retrieval recall hook (docs/phase_2.md Task 8): recall@k over gold (document, IndiKit indicator) pairs for the
full method and the baseline. A gold pair counts as recalled when its ``indicator_text`` (fuzzy) or its
``location`` appears in the retrieved evidence. Runs cleanly with zero verified gold documents."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from pipeline import config
from pipeline.agent.retrieve import RetrievalResult
from pipeline.eval.gold import GoldDocument, load_gold
from pipeline.metadata.models import IndicatorMetadata


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%#]+", " ", (text or "").lower()).strip()


def evidence_matches(result: RetrievalResult, indicator_text: Optional[str], location: Optional[str], threshold: int = 80) -> bool:
    try:
        from rapidfuzz import fuzz
    except Exception:  # pragma: no cover
        fuzz = None
    want = _norm(indicator_text or "")
    loc = _norm(location or "")
    for ev in result.evidence:
        text = _norm(ev.display_text)
        if want and (want in text or (fuzz is not None and fuzz.partial_ratio(want, text) >= threshold)):
            return True
        if loc and len(loc) >= 4 and loc in _norm(ev.location):
            return True
    return False


def recall_at_k(gold_docs: list[GoldDocument], doc_id_map: dict[str, str], metadata: dict[str, IndicatorMetadata], retrieve_fn: Callable[[str, IndicatorMetadata], RetrievalResult], k: int = config.RETRIEVAL_TOP_K) -> dict:
    """``doc_id_map`` maps gold ``doc_id`` -> KB ``doc_id``; ``retrieve_fn(kb_doc_id, indicator)`` returns top-k evidence."""
    total = hit = 0
    misses: list[tuple[str, str]] = []
    for doc in gold_docs:
        kb_doc = doc_id_map.get(doc.doc_id)
        if kb_doc is None:
            continue
        for row in doc.rows:
            if not row.indikit_id or row.indikit_id not in metadata:
                continue
            total += 1
            res = retrieve_fn(kb_doc, metadata[row.indikit_id])
            res.evidence = res.evidence[:k]
            if evidence_matches(res, row.indicator_text, row.location):
                hit += 1
            else:
                misses.append((doc.doc_id, row.indikit_id))
    return {"k": k, "pairs": total, "recalled": hit, "recall": (hit / total) if total else None, "misses": misses}


def run_retrieval_eval(gold_dir: Path = config.GOLD_DIR, metadata: Optional[dict[str, IndicatorMetadata]] = None, doc_id_map: Optional[dict[str, str]] = None, k: int = config.RETRIEVAL_TOP_K,
                       retrieve_full: Optional[Callable] = None, retrieve_base: Optional[Callable] = None, verified_only: bool = True) -> Optional[dict]:
    """Recall@k for the full method and the baseline over the verified gold documents; None when there are none."""
    docs = [d for d in load_gold(gold_dir) if (d.entry is not None and (d.entry.verified or not verified_only))]
    if not docs:
        print("no verified gold documents")
        return None
    if metadata is None:
        from pipeline.metadata.build_metadata import read_jsonl

        metadata = {r.indikit_id: r for r in read_jsonl()}
    doc_id_map = doc_id_map or {d.doc_id: d.doc_id for d in docs}
    out: dict = {"documents": len(docs)}
    if retrieve_full is None:
        from pipeline.agent.retrieve import retrieve as retrieve_full  # type: ignore[assignment]
    out["full"] = recall_at_k(docs, doc_id_map, metadata, lambda d, i: retrieve_full(d, i, top_k=k), k)
    if retrieve_base is None:
        from pipeline.agent.retrieve import retrieve_baseline as retrieve_base  # type: ignore[assignment]
    out["baseline"] = recall_at_k(docs, doc_id_map, metadata, lambda d, i: retrieve_base(d, i, top_k=k), k)
    return out
