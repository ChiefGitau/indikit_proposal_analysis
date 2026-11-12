"""Plain-RAG baseline index (paper's "Benchmark", docs/phase_2.md Task 7).

Flat text only (legacy ``simple_parse_pdf`` / ``simple_parse_docx`` behaviour; xlsx cells dumped), fixed-size
character chunks with overlap, English ``all-MiniLM-L6-v2``, no outline, no table KB, no search terms, no
re-ranking. Stored under ``outputs/kb_baseline/<doc_id>/``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline import config
from pipeline.preprocess.embeddings import Embedder, EmbeddingCache, VectorIndex, get_embedder
from pipeline.preprocess.models import KBEntry
from pipeline.preprocess.registry import DocumentRef, now_iso


def flat_text(ref: DocumentRef) -> list[tuple[str, str]]:
    """(location, text) units in the legacy style: PDF pages, docx paragraphs then tables, xlsx cell dumps."""
    from extraction_utils import simple_parse_docx, simple_parse_pdf

    if ref.file_type == "pdf":
        return [(c.page_or_section or "", c.content) for c in simple_parse_pdf(str(ref.path)) if c.content.strip()]
    if ref.file_type == "docx":
        return [(c.page_or_section or "", c.content) for c in simple_parse_docx(str(ref.path)) if c.content.strip()]
    import openpyxl

    wb = openpyxl.load_workbook(str(ref.path), data_only=True, read_only=True)
    out = []
    for ws in wb.worksheets:
        lines = [" | ".join(str(c) for c in row if c is not None) for row in ws.iter_rows(values_only=True)]
        text = "\n".join(l for l in lines if l.strip())
        if text:
            out.append((f"sheet {ws.title}", text))
    return out


def fixed_chunks(units: list[tuple[str, str]], size: int = config.BASELINE_CHUNK_CHARS, overlap: int = config.BASELINE_CHUNK_OVERLAP) -> list[tuple[str, str]]:
    text = "\n".join(t for _, t in units)
    offsets: list[tuple[int, str]] = []
    pos = 0
    for loc, t in units:
        offsets.append((pos, loc))
        pos += len(t) + 1
    chunks: list[tuple[str, str]] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size]
        loc = next((l for o, l in reversed(offsets) if o <= start), "")
        chunks.append((loc, piece))
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


class BaselineKB:
    def __init__(self, doc_id: str, meta: dict, chunks: list[KBEntry], index: VectorIndex):
        self.doc_id, self.meta, self.chunks, self.index = doc_id, meta, chunks, index
        self._by_id = {c.entry_id: c for c in chunks}

    @property
    def embedding_model(self) -> str:
        return self.meta.get("embedding_model", "")

    def search(self, qvec: np.ndarray, top_k: int) -> list[tuple[KBEntry, float]]:
        return [(self._by_id[eid], s) for eid, s in self.index.search(qvec, top_k)]


def build_baseline_kb(ref: DocumentRef, embedder: Embedder, cache: Optional[EmbeddingCache] = None, kb_dir: Path = config.KB_BASELINE_DIR) -> dict:
    t0 = time.time()
    folder = Path(kb_dir) / ref.doc_id
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"doc_id": ref.doc_id, "relative_path": ref.relative_path, "sha256": ref.sha256, "embedding_model": embedder.name, "pipeline_version": config.PIPELINE_VERSION, "built_at": now_iso()}
    try:
        units = flat_text(ref)
        pieces = fixed_chunks(units)
        chunks = [KBEntry(entry_id=f"{ref.doc_id}:base:{i:05d}", doc_id=ref.doc_id, kb_type="text", entry_kind="chunk", text_for_embedding=t, display_text=t, location=loc or "document") for i, (loc, t) in enumerate(pieces)]
        vectors = cache.embed(embedder, [c.text_for_embedding for c in chunks]) if cache is not None else embedder.embed([c.text_for_embedding for c in chunks])
        VectorIndex([c.entry_id for c in chunks], vectors).save(folder / "index")
        (folder / "chunks.jsonl").write_text("".join(json.dumps(c.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n" for c in chunks), encoding="utf-8")
        meta.update({"status": "ok", "n_chunks": len(chunks), "n_chars": sum(len(t) for _, t in units)})
    except Exception as exc:
        meta.update({"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "n_chunks": 0})
    meta["build_seconds"] = round(time.time() - t0, 2)
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return meta


def load_baseline_kb(doc_id: str, kb_dir: Path = config.KB_BASELINE_DIR) -> BaselineKB:
    folder = Path(kb_dir) / doc_id
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no baseline KB for {doc_id} under {folder}; run: python -m pipeline.run build-baseline-kb --all")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "ok":
        raise RuntimeError(f"baseline KB for {doc_id} failed: {meta.get('error')}")
    chunks = [KBEntry.model_validate_json(l) for l in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return BaselineKB(doc_id, meta, chunks, VectorIndex.load(folder / "index"))


def default_baseline_embedder() -> Embedder:
    return get_embedder(config.BASELINE_EMBEDDING_MODEL)
