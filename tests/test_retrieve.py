from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.agent.rerank import LexicalReranker, get_reranker
from pipeline.agent.retrieve import RetrievalConfig, Retriever, retrieve, score_pairs, write_pair_scores
from pipeline.metadata.models import Aspect, IndicatorMetadata
from pipeline.preprocess.embeddings import EmbeddingCache, HashingEmbedder
from pipeline.preprocess.knowledge_base import build_kb
from pipeline.preprocess.registry import discover_documents
from tests.fixtures_docs import make_docx, make_pdf


@pytest.fixture
def built(tmp_path: Path):
    root = tmp_path / "proposals" / "africa" / "x" / "small"
    root.mkdir(parents=True)
    make_docx(root / "a.docx")
    make_pdf(root / "b.pdf")
    refs = discover_documents([("proposals", tmp_path / "proposals")])
    emb = HashingEmbedder(512)
    for r in refs:
        build_kb(r, emb, EmbeddingCache(None), kb_dir=tmp_path / "kb")
    cfg = RetrievalConfig(embedder=emb.name, reranker="lexical")
    retriever = Retriever(cfg, embedder=emb, reranker=LexicalReranker(), cache=EmbeddingCache(None), kb_dir=tmp_path / "kb", queries_dir=tmp_path / "q")
    return refs, retriever


HANDWASH = IndicatorMetadata(indikit_id="5138", name="Handwashing Facilities With Soap and Water", aspect=[Aspect(sector="WASH")], kpi_wording_en="% of households with handwashing facilities with soap and water at home",
                             search_terms=["handwashing facilities", "soap and water", "handwashing station"], knowledge="k")
CHLORINE = IndicatorMetadata(indikit_id="5330", name="Drinking Water Quality (residual chlorine)", kpi_wording_en="% of water samples with free residual chlorine between 0.2 and 0.5 mg/l", search_terms=["free residual chlorine", "frc"])
UNRELATED = IndicatorMetadata(indikit_id="9000", name="Zebra migration distance", kpi_wording_en="average kilometres migrated by zebra herds", search_terms=["zebra", "migration"])


def test_retrieve_top_n_expansion_dedup_and_result_fields(built):
    refs, retriever = built
    docx_id = next(r.doc_id for r in refs if r.file_type == "docx")
    res = retriever.retrieve(docx_id, HANDWASH)
    assert res.doc_id == docx_id and res.indikit_id == "5138" and res.evidence and len(res.evidence) <= 4
    ids = [e.entry_id for e in res.evidence]
    assert len(ids) == len(set(ids))
    top = res.evidence[0]
    assert "handwashing facilities with soap and water" in top.display_text.lower()
    assert top.kb_type == "table" and top.entry_kind == "row" and top.is_logframe and top.boost_applied and top.row_id and top.table_id
    assert "Columns: Intervention logic | Indicator | Baseline | Target" in top.display_text and "Result context: Outcome 1" in top.display_text
    assert res.best_vector_score >= top.vector_score - 1e-9 and res.best_rerank_score is not None and res.n_candidates >= len(res.evidence)
    assert res.config_fingerprint.endswith(":lexical") and res.query_text.startswith("Handwashing")
    # candidates come from all three KBs, outline hits are expanded into chunks / rows
    kinds = {(e.kb_type, e.entry_kind) for e in retriever.retrieve(docx_id, HANDWASH, top_n=10, top_k=30).evidence}
    assert ("text", "chunk") in kinds and ("table", "row") in kinds
    expanded = [e for e in retriever.retrieve(docx_id, HANDWASH, top_n=10, top_k=30).evidence if e.expanded_from]
    assert expanded and all(e.expanded_from.split(":")[1] in ("outline", "table") for e in expanded)


def test_flag_variants_and_determinism(built):
    refs, retriever = built
    pdf_id = next(r.doc_id for r in refs if r.file_type == "pdf")
    a = retriever.retrieve(pdf_id, CHLORINE)
    b = retriever.retrieve(pdf_id, CHLORINE)
    assert a.model_dump() == b.model_dump()
    assert "free residual chlorine" in a.evidence[0].display_text.lower()
    no_rerank = retriever.retrieve(pdf_id, CHLORINE, use_rerank=False)
    assert [e.entry_id for e in no_rerank.evidence] == [e.entry_id for e in sorted(no_rerank.evidence, key=lambda e: (-e.vector_score, e.entry_id))]
    assert not no_rerank.config_fingerprint.endswith(":lexical") and no_rerank.config_fingerprint.endswith(":none")
    no_terms = retriever.retrieve(pdf_id, CHLORINE, use_search_terms=False)
    assert "frc" not in no_terms.query_text and no_terms.evidence
    only_text = retriever.retrieve(pdf_id, CHLORINE, kb_types=("text",))
    assert all(e.kb_type == "text" for e in only_text.evidence)
    cfg_no_boost = retriever.cfg.model_copy(update={"logframe_boost": 0.0})
    r2 = Retriever(cfg_no_boost, embedder=retriever.embedder, reranker=LexicalReranker(), cache=EmbeddingCache(None), kb_dir=retriever.kb_dir, queries_dir=retriever.queries_dir)
    assert all(not e.boost_applied for e in r2.retrieve(pdf_id, CHLORINE).evidence)
    unrelated = retriever.retrieve(pdf_id, UNRELATED)
    assert unrelated.best_vector_score < a.best_vector_score
    # module-level function with an injected retriever
    assert retrieve(pdf_id, CHLORINE, retriever=retriever).evidence[0].entry_id == a.evidence[0].entry_id
    with pytest.raises(ValueError):
        Retriever(RetrievalConfig(embedder="other"), embedder=HashingEmbedder(16), reranker=LexicalReranker(), cache=EmbeddingCache(None), kb_dir=retriever.kb_dir, queries_dir=retriever.queries_dir).kb(pdf_id)


def test_score_pairs_and_writer(built, tmp_path: Path):
    refs, retriever = built
    rows, stats = score_pairs(retriever, [r.doc_id for r in refs] + ["missing-doc"], [HANDWASH, UNRELATED], use_rerank=False, progress=False)
    assert stats["pairs"] == 4 and stats["use_rerank"] is False and stats["reranker"] == "none"
    assert {r["indikit_id"] for r in rows} == {"5138", "9000"} and all(r["best_rerank_score"] is None for r in rows)
    assert all(r["top_entry_id"] for r in rows) and isinstance(rows[0]["top_is_logframe"], bool)
    out = write_pair_scores(rows, tmp_path / "pair_scores.parquet")
    assert out.exists() and out.suffix in (".parquet", ".csv")
    rows_r, _ = score_pairs(retriever, [refs[0].doc_id], [HANDWASH], use_rerank=True, progress=False)
    assert rows_r[0]["best_rerank_score"] is not None


def test_lexical_reranker_and_factory():
    r = LexicalReranker()
    s = r.score("free residual chlorine", ["water samples with free residual chlorine", "budget lines", ""])
    assert s[0] > s[1] >= s[2] == 0.0
    assert get_reranker("lexical").name == "lexical"
