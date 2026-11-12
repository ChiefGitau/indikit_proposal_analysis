from __future__ import annotations

from pathlib import Path

from pipeline import config
from pipeline.agent.retrieve import Evidence, RetrievalResult
from pipeline.eval.gold import load_gold, validate_gold_file
from pipeline.eval.retrieval_eval import evidence_matches, recall_at_k, run_retrieval_eval
from pipeline.metadata.models import IndicatorMetadata


def _result(doc, ind, texts, locs=None):
    locs = locs or ["p.1"] * len(texts)
    return RetrievalResult(doc_id=doc, indikit_id=ind, evidence=[Evidence(entry_id=f"e{i}", kb_type="text", display_text=t, location=l, vector_score=0.5) for i, (t, l) in enumerate(zip(texts, locs))])


def test_evidence_matches_text_fuzzy_and_location():
    r = _result("d", "5138", ["Row: Indicator: % of households with handwashing facilities with soap and water at home | Target: 60%"], ["p.4"])
    assert evidence_matches(r, "% of households with handwashing facilities with soap and water", None)
    assert evidence_matches(r, "households with hand-washing facilities with soap & water at home", None)   # fuzzy
    assert not evidence_matches(r, "Number of VSLA groups established", None)
    assert evidence_matches(_result("d", "x", ["irrelevant"], ["Annex 1 logframe p.2"]), "nothing", "logframe p.2")
    assert not evidence_matches(_result("d", "x", [], []), "x", "p.9")


def test_recall_at_k_on_example_gold_with_synthetic_document(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "index.csv").write_text("doc_id,source_file,language,has_logframe,indicators_in_other_document,labelled_by,verified,split\nexample,proposals/x.docx,en,true,,HM,true,dev\n", encoding="utf-8")
    (gold_dir / "example.csv").write_text((config.GOLD_DIR / "_example.csv").read_text(encoding="utf-8"), encoding="utf-8")
    rows, errors = validate_gold_file(gold_dir / "example.csv", None)
    assert not errors and len(rows) == 4
    docs = load_gold(gold_dir)
    metadata = {i: IndicatorMetadata(indikit_id=i, name=f"ind {i}") for i in ("5138", "5620", "5330")}
    # synthetic "document": the retriever returns the handwashing row and the FCS text, but nothing on chlorine
    synthetic = {
        "5138": ["Table: logframe [p.2]\nRow: Indicator: % of households with handwashing facilities with soap and water at home | Target: 60%"],
        "5620": ["Proportion of target households reporting acceptable food consumption (FCS > 35) rises to 75%"],
        "5330": ["Boreholes will be rehabilitated in all target villages."],
    }

    def fake_retrieve(doc_id, ind):
        return _result(doc_id, ind.indikit_id, synthetic.get(ind.indikit_id, []), ["p.9"] * 3)

    out = recall_at_k(docs, {"example": "kb-doc"}, metadata, fake_retrieve, k=4)
    assert out == {"k": 4, "pairs": 3, "recalled": 2, "recall": 2 / 3, "misses": [("example", "5330")]}
    full = run_retrieval_eval(gold_dir, metadata, {"example": "kb-doc"}, k=4, retrieve_full=lambda d, i, top_k: fake_retrieve(d, i), retrieve_base=lambda d, i, top_k: _result(d, i.indikit_id, []))
    assert full["documents"] == 1 and full["full"]["recall"] == 2 / 3 and full["baseline"]["recall"] == 0.0


def test_run_retrieval_eval_with_no_gold_documents(capsys):
    assert run_retrieval_eval(config.GOLD_DIR, {}, k=4) is None   # the real index is empty
    assert "no verified gold documents" in capsys.readouterr().out
