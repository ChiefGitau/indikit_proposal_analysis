from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.preprocess.embeddings import EmbeddingCache, HashingEmbedder, VectorIndex, get_embedder
from pipeline.preprocess.knowledge_base import build_kb, build_text_entries, load_kb
from pipeline.preprocess.logframe import logframe_content
from pipeline.preprocess.models import Block
from pipeline.preprocess.registry import discover_documents
from tests.fixtures_docs import make_docx, make_pdf, make_xlsx


@pytest.fixture
def corpus(tmp_path: Path):
    root = tmp_path / "proposals" / "africa" / "burundi" / "small"
    root.mkdir(parents=True)
    make_docx(root / "proposal.docx")
    make_xlsx(root / "logframe.xlsx")
    make_pdf(root / "proposal.pdf")
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    return discover_documents([("proposals", tmp_path / "proposals")])


def test_hashing_embedder_and_index_roundtrip(tmp_path: Path):
    emb = HashingEmbedder(128)
    v = emb.embed(["handwashing soap water", "handwashing soap water", "budget lines"])
    assert v.shape == (3, 128) and np.allclose(np.linalg.norm(v, axis=1), 1.0) and np.allclose(v[0], v[1])
    assert v[0] @ v[2] < v[0] @ v[1]
    idx = VectorIndex(["a", "b", "c"], v)
    assert idx.search(v[0], 2)[0][0] == "a"
    idx.save(tmp_path / "index_x")
    loaded = VectorIndex.load(tmp_path / "index_x")
    assert loaded.search(v[0], 3) == idx.search(v[0], 3)
    assert VectorIndex([], np.zeros((0, 0))).search(v[0], 3) == []
    assert get_embedder("hashing:64").dim == 64
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    a = cache.embed(emb, ["x", "y"])
    calls = []
    orig = emb.embed
    emb.embed = lambda texts: (calls.append(list(texts)), orig(texts))[1]  # type: ignore[method-assign]
    b = cache.embed(emb, ["y", "x", "z"])
    assert calls == [["z"]] and np.allclose(b[1], a[0])
    cache.close()


def test_embedding_cache_is_usable_from_worker_threads(tmp_path: Path):
    """The answering step shares one cache between threads; sqlite must not raise 'created in a thread'."""
    from concurrent.futures import ThreadPoolExecutor

    emb = HashingEmbedder(64)
    cache = EmbeddingCache(tmp_path / "c.sqlite")       # opened in the main thread
    cache.embed(emb, ["seed"])
    texts = [f"t{i}" for i in range(40)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        vecs = list(pool.map(lambda t: cache.embed(emb, [t, "seed"]), texts))
    assert all(v.shape == (2, 64) for v in vecs) and np.allclose(vecs[0][1], vecs[-1][1])
    cache.close()


def test_text_chunking_is_section_aware():
    def para(i, text, path):
        return Block(block_id=f"d:b{i}", doc_id="d", kind="paragraph", text=text, order=i, heading_path=path, page=1)

    words = " ".join(["word"] * 120)
    blocks = [
        Block(block_id="d:h1", doc_id="d", kind="heading", text="1 Intro", order=0, style_info={"heading_level": 1}),
        para(1, words, ["1 Intro"]), para(2, words, ["1 Intro"]), para(3, words, ["1 Intro"]),
        Block(block_id="d:h2", doc_id="d", kind="heading", text="2 Logframe", order=4, style_info={"heading_level": 1}),
        para(5, "short indicator text", ["2 Logframe"]),
        Block(block_id="d:t", doc_id="d", kind="table", text="Table", order=6, heading_path=["2 Logframe"]),
        para(7, " ".join(["long"] * 800), ["2 Logframe"]),
    ]
    entries, block_map = build_text_entries("d", blocks, target=250, max_words=350)
    assert [e.payload_ref["block_ids"] for e in entries[:3]] == [["d:b1", "d:b2"], ["d:b3"], ["d:b5"]]   # 240 words, then the heading change closes the chunk
    assert entries[0].text_for_embedding.startswith("1 Intro\n") and entries[0].location == "p.1 | 1 Intro"
    assert all(e.entry_id == f"d:text:{i:05d}" for i, e in enumerate(entries))
    assert len(entries) >= 6 and all(len(e.display_text.split()) <= 350 for e in entries)   # the 800-word block is split
    assert block_map["d:b1"] == entries[0].entry_id


def test_build_and_load_kb_end_to_end(corpus, tmp_path: Path):
    kb_dir = tmp_path / "kb"
    emb = HashingEmbedder(256)
    rows = {r.doc_id: build_kb(r, emb, EmbeddingCache(None), kb_dir=kb_dir) for r in corpus}
    by_type = {r.file_type: r for r in corpus}
    assert rows[by_type["pdf"].doc_id].parse_status in ("ok",) if "broken" not in by_type["pdf"].relative_path else True
    broken = next(r for r in corpus if "broken" in r.relative_path)
    assert rows[broken.doc_id].parse_status == "failed" and rows[broken.doc_id].error
    docx_ref = next(r for r in corpus if r.file_type == "docx")
    row = rows[docx_ref.doc_id]
    assert row.parse_status == "ok" and row.n_tables == 1 and row.n_logframe_tables == 1 and row.n_logframe_sections == 1 and row.n_references >= 1
    assert row.indicators_also_elsewhere and not row.indicators_elsewhere and row.language == "en"
    folder = kb_dir / docx_ref.doc_id
    for name in ("blocks.jsonl", "tables.jsonl", "rows.jsonl", "outline.json", "references.json", "kb_text.jsonl", "kb_outline.jsonl", "kb_table.jsonl", "index_text.npy", "index_outline.ids.json", "index_table.npy", "kb_meta.json"):
        assert (folder / name).exists(), name
    kb = load_kb(docx_ref.doc_id, kb_dir)
    assert kb.embedding_model == emb.name and set(kb.entries) == {"text", "outline", "table"}
    kinds = {e.entry_kind for e in kb.entries["table"]}
    assert kinds == {"row", "keyword", "summary"}
    row_entries = [e for e in kb.entries["table"] if e.entry_kind == "row"]
    assert len(row_entries) == 3 and "Result context: Outcome 1: Improved WASH" in row_entries[1].display_text and "Columns: Intervention logic | Indicator | Baseline | Target" in row_entries[1].display_text
    assert all(e.is_logframe for e in row_entries)
    # search before/after save gives the same ranking
    q = emb.embed(["handwashing facilities soap water"])[0]
    fresh = kb.search("table", q, 3)
    again = load_kb(docx_ref.doc_id, kb_dir).search("table", q, 3)
    assert [(e.entry_id, round(s, 6)) for e, s in fresh] == [(e.entry_id, round(s, 6)) for e, s in again]
    assert fresh[0][0].payload_ref.get("row_id", "").endswith("r0000") or "handwashing" in fresh[0][0].display_text.lower()
    # expansion: outline node -> chunks and tables; keyword -> row; summary -> rows (small table)
    node = next(e for e in kb.entries["outline"] if "Logical framework" in e.display_text)
    expanded = kb.expand(node)
    assert any(e.kb_type == "text" for e in expanded) and any(e.entry_kind == "row" for e in expanded)
    kw = next(e for e in kb.entries["table"] if e.entry_kind == "keyword" and e.payload_ref.get("row_id"))
    assert [e.entry_kind for e in kb.expand(kw)] == ["row"]
    summary = next(e for e in kb.entries["table"] if e.entry_kind == "summary")
    assert len(kb.expand(summary)) == 3
    # logframe_content interface
    content = logframe_content(docx_ref.doc_id, kb_dir)
    assert len(content["tables"]) == 1 and len(content["tables"][0]["rows"]) == 3 and content["section_heading_ids"]
    assert any("logframe below" in b.text for b in content["blocks"])
    # determinism of the written KB
    first = (folder / "kb_table.jsonl").read_bytes()
    build_kb(docx_ref, emb, EmbeddingCache(None), kb_dir=kb_dir)
    assert (folder / "kb_table.jsonl").read_bytes() == first
    with pytest.raises(RuntimeError):
        load_kb(broken.doc_id, kb_dir)
    with pytest.raises(FileNotFoundError):
        load_kb("nope", kb_dir)
    xlsx_row = rows[next(r for r in corpus if r.file_type == "xlsx").doc_id]
    assert xlsx_row.n_tables == 3 and xlsx_row.n_logframe_tables == 1 and xlsx_row.language == "fr"
