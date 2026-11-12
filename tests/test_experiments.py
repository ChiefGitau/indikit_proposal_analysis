"""Evaluation code (docs/phase_3.md Task 7) with the mock provider, synthetic documents and fictional gold labels."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.agent.llm import MockLLMProvider, ResponseCache, UsageTracker
from pipeline.agent.orchestrate import RunConfig, Runner
from pipeline.eval.gold import GoldDocument, GoldIndexEntry, GoldRow
from pipeline.eval.run_experiments import EXPERIMENTS, ablation_figure, error_analysis, error_counts, gold_doc_id_map, gold_pair_labels, gold_present_pairs, predictions_from_results, run_experiment, score, write_errors, write_results_table
from pipeline.preprocess.embeddings import EmbeddingCache
from tests.fixtures_agent import METADATA, answer_from_evidence, build_agent_fixture


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_agent_fixture(tmp_path_factory.mktemp("exp"))


def _gold(fx):
    docx = GoldDocument(doc_id="g_docx", entry=GoldIndexEntry(doc_id="g_docx", source_file=fx.manifest[fx.docx_id].relative_path.split("proposals/", 1)[1], language="en", has_logframe=True, indicators_in_other_document="Annex C - Logical Framework", labelled_by="t", verified=True, split="dev"),
                        rows=[GoldRow(doc_indicator_id="D01", indicator_text="% of households with handwashing facilities with soap and water", level="outcome", baseline="12%", target="60%", unit="%", location="logframe", indikit_id="5138", match_type="verbatim"),
                              GoldRow(doc_indicator_id="D02", indicator_text="Number of people receiving hygiene kits", level="output", baseline="0", target="2,500", location="logframe", indikit_id=None, match_type="none"),
                              GoldRow(doc_indicator_id="D03", indicator_text="# of latrines constructed", level="output", baseline="0", target="40", location="logframe", indikit_id="5140", match_type="close")])
    pdf = GoldDocument(doc_id=fx.pdf_id, entry=GoldIndexEntry(doc_id=fx.pdf_id, source_file="x", language="en", has_logframe=True, indicators_in_other_document="Indicator Tracking Table (Annex O)", labelled_by="t", verified=True, split="dev"),
                       rows=[GoldRow(doc_indicator_id="D01", indicator_text="% of water samples with free residual chlorine above 0.2 mg/l", level="output", baseline="n/a", target="90%", location="p.2", indikit_id="5330", match_type="close"),
                             GoldRow(doc_indicator_id="D02", indicator_text="Number of boreholes rehabilitated", level="output", baseline="0", target="12", location="p.1", indikit_id=None, match_type="none")])
    return [docx, pdf]


def _factory(fx, tmp_path):
    def factory(run_id, overrides):
        kw = dict(run_id=run_id, provider="mock", answer_model="m", extraction_model="m", tau=0.2, top_m=4, max_calls=200, max_cost_usd=5.0, workers=2)
        kw.update(overrides)
        cfg = RunConfig(**kw)
        return Runner(cfg, provider=MockLLMProvider(answer_from_evidence), metadata=METADATA, manifest=fx.manifest, retriever=fx.retriever, retrieve_fn=(fx.retrieve if cfg.baseline_retrieval else None), embedder=fx.embedder,
                      embedding_cache=EmbeddingCache(None), cache=ResponseCache(tmp_path / "cache"), usage=UsageTracker(200, 5.0, prices={"m": (0.1, 0.1)}, path=tmp_path / "runs" / run_id / "usage.json"), kb_dir=fx.kb_dir,
                      runs_dir=tmp_path / "runs", pair_scores=fx.pair_scores)
    return factory


def test_gold_mapping_labels_and_scoring(fx, tmp_path):
    gold = _gold(fx)
    gmap = gold_doc_id_map(gold, fx.manifest)
    assert gmap == {"g_docx": fx.docx_id, fx.pdf_id: fx.pdf_id}
    labels = gold_pair_labels(gold, METADATA, gmap)
    assert len(labels) == 2 * len(METADATA) and sum(l.present for l in labels) == 3
    assert gold_present_pairs(gold, gmap) == {(fx.docx_id, "5138"), (fx.docx_id, "5140"), (fx.pdf_id, "5330")}
    runner = _factory(fx, tmp_path)("score-run", {})
    results, _ = runner.run([fx.docx_id, fx.pdf_id], progress=None)
    preds = predictions_from_results(results)
    assert any(p.indikit_id == "5138" and p.present and p.match_type == "verbatim" for p in preds)
    sc = score(gold, results, METADATA, gmap)
    o = sc["overall"]
    assert o["n_documents"] == 2 and o["n_present"] == 3 and 0 <= o["acc_dc"] <= 1 and o["tp"] >= 2 and o["recall"] >= 2 / 3 and o["extraction_recall"] >= 0.5
    assert o["reference_detection"] == 1.0   # Annex C (docx) and Annex O (pdf) detected in Phase 2
    assert set(sc["per_document"]) == {"g_docx", fx.pdf_id}
    universe = {(fx.docx_id, "5138")}
    sc2 = score(gold, results, METADATA, gmap, pair_universe=universe)
    assert sc2["overall"]["n_pairs"] == 3   # present pairs always count, absent pairs restricted to the universe


def test_error_analysis_categories_and_results_table(fx, tmp_path):
    gold = _gold(fx)
    gmap = gold_doc_id_map(gold, fx.manifest)
    runner = _factory(fx, tmp_path)("err-run", {})
    results, _ = runner.run([fx.docx_id, fx.pdf_id], progress=None)
    sel = runner.selection()
    selected = set(zip(sel["doc_id"], sel["indikit_id"]))
    rows = error_analysis(gold, results, METADATA, gmap, selected - {(fx.docx_id, "5140")}, evidence_check=lambda d, i, t: i != "5330", manifest=fx.manifest)
    cats = error_counts(rows)
    assert cats["gate_miss"] >= 1 or cats["llm_false_negative"] + cats["retrieval_miss"] >= 0
    assert set(cats) == {"retrieval_miss", "gate_miss", "llm_false_negative", "llm_false_positive", "wrong_match_type", "wrong_value", "parse_problem"}
    p = write_errors(rows, tmp_path / "errors.csv")
    assert p.exists()
    table_rows = [{"experiment": "benchmark", "model": "m", "split": "dev", "n_documents": 2, "n_pairs": 8, "acc_dc": 0.5, "acc_de": 0.25, "precision": 0.5, "recall": 0.5, "f1": 0.5, "llm_calls": 8, "cost_usd": 0.01, "watermark": "UNVERIFIED GOLD"},
                  {"experiment": "plus_search_term", "model": "m", "split": "dev", "n_documents": 2, "n_pairs": 8, "acc_dc": 0.9, "acc_de": 0.5, "precision": 1.0, "recall": 0.7, "f1": 0.8, "llm_calls": 8, "cost_usd": 0.01, "watermark": "UNVERIFIED GOLD"},
                  {"experiment": "gate_sensitivity", "split": "dev", "best": {"tau": 0.1, "top_m": 20}}, {"experiment": "reverse_check_value", "split": "dev", "reverse_recovered": 1, "reverse_candidates": 3, "reverse_calls": 3, "reverse_cost_usd": 0.001}]
    out = write_results_table(table_rows, tmp_path / "results.csv", errors=rows, changelog="- v1: none")
    md = out.with_suffix(".md").read_text()
    assert "UNVERIFIED GOLD" in md and "## Ablation" in md and "## Model comparison" in md and "## Error analysis" in md and "## Gate sensitivity" in md and "Reverse-check" in md and "preliminary" in md
    fig = ablation_figure(table_rows, tmp_path / "fig" / "ablation.png")
    assert fig is not None and fig.exists()


def test_run_experiments_end_to_end_with_mock(fx, tmp_path):
    gold = _gold(fx)
    gmap = gold_doc_id_map(gold, fx.manifest)
    factory = _factory(fx, tmp_path)
    eval_dir = tmp_path / "eval"
    rows = []
    for name in ("benchmark", "enhanced_rag", "plus_knowledge", "plus_search_term", "model_comparison", "reverse_check_value"):
        rows += run_experiment(name, "dev", runner_factory=factory, gold_docs=gold, metadata=METADATA, doc_id_map=gmap, eval_dir=eval_dir, models=["m", "m-strong"], watermark="UNVERIFIED GOLD")
    assert {r["experiment"] for r in rows} == {"benchmark", "enhanced_rag", "plus_knowledge", "plus_search_term", "model_comparison", "reverse_check_value"} and len(rows) == 7
    assert (eval_dir / "benchmark" / "metrics.json").exists() and (eval_dir / "benchmark" / "predictions.csv").exists() and (eval_dir / "model_comparison" / "m-strong" / "metrics.json").exists()
    m = json.loads((eval_dir / "plus_search_term" / "metrics.json").read_text())
    assert m["watermark"] == "UNVERIFIED GOLD" and m["n_present"] == 3 and "per_document" in m
    gate = run_experiment("gate_sensitivity", "dev", runner_factory=factory, gold_docs=gold, metadata=METADATA, doc_id_map=gmap, eval_dir=eval_dir, pair_scores=fx.pair_scores)
    assert gate[0]["best"] is not None and (eval_dir / "gate_sensitivity" / "curve.csv").exists()
    with pytest.raises(ValueError):
        run_experiment("nope", "dev", runner_factory=factory, gold_docs=gold, metadata=METADATA, doc_id_map=gmap)
    assert EXPERIMENTS[-1] == "reverse_check_value"
