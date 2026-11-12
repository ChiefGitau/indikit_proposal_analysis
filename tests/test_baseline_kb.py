from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.agent.retrieve import retrieve_baseline
from pipeline.preprocess.baseline_kb import build_baseline_kb, fixed_chunks, load_baseline_kb
from pipeline.preprocess.embeddings import EmbeddingCache, HashingEmbedder
from pipeline.preprocess.registry import discover_documents
from tests.fixtures_docs import make_docx, make_pdf, make_xlsx
from tests.test_retrieve import HANDWASH


def test_fixed_chunks_overlap():
    units = [("page_1", "a" * 1500), ("page_2", "b" * 700)]
    chunks = fixed_chunks(units, size=1000, overlap=200)
    assert [len(c) for _, c in chunks] == [1000, 1000, 601]
    assert chunks[0][1][-200:] == chunks[1][1][:200]
    assert chunks[0][0] == "page_1" and chunks[2][0] == "page_2"


def test_build_and_retrieve_baseline(tmp_path: Path):
    root = tmp_path / "proposals" / "r" / "c" / "s"
    root.mkdir(parents=True)
    make_docx(root / "a.docx")
    make_pdf(root / "b.pdf")
    make_xlsx(root / "c.xlsx")
    refs = discover_documents([("proposals", tmp_path / "proposals")])
    emb = HashingEmbedder(256)
    metas = [build_baseline_kb(r, emb, EmbeddingCache(None), kb_dir=tmp_path / "base") for r in refs]
    assert all(m["status"] == "ok" and m["n_chunks"] >= 1 for m in metas)
    doc_id = refs[0].doc_id
    kb = load_baseline_kb(doc_id, tmp_path / "base")
    assert kb.embedding_model == emb.name and all(len(c.display_text) <= 1000 for c in kb.chunks)
    res = retrieve_baseline(doc_id, HANDWASH, top_k=2, embedder=emb, cache=EmbeddingCache(None), kb_dir=tmp_path / "base")
    assert res.config_fingerprint == f"baseline:{emb.name}" and 1 <= len(res.evidence) <= 2 and "handwashing" in res.evidence[0].display_text.lower()
    assert "handwashing station" not in res.query_text  # no search terms in the baseline query
    with pytest.raises(ValueError):
        retrieve_baseline(doc_id, HANDWASH, embedder=HashingEmbedder(32), kb_dir=tmp_path / "base")
    with pytest.raises(FileNotFoundError):
        load_baseline_kb("nope", tmp_path / "base")
