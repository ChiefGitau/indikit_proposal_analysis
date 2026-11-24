"""Output files (docs/phase_3.md Task 6) and the end-to-end synthetic run: schemas, counts computed in code, needs_review,
'indicators elsewhere' note, report --doc table, resumability and budget stop of the orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.agent.llm import BudgetExceeded, MockLLMProvider, ResponseCache, UsageTracker
from pipeline.agent.models import ConsolidatedIndicator, DocIndicator, DocumentResult, LinkedMatch
from pipeline.agent.orchestrate import RunConfig, Runner, pilot_doc_ids
from pipeline.agent.outputs import DOC_IND_COLUMNS, MATRIX_COLUMNS, NOVEL_COLUMNS, SUMMARY_COLUMNS, cluster_texts, copy_latest, document_report_markdown, needs_review, write_run_outputs
from pipeline.preprocess.embeddings import EmbeddingCache
from pipeline.preprocess.models import ManifestRow
from tests.fixtures_agent import METADATA, answer_from_evidence, build_agent_fixture


def _ci(did, text, matches=(), is_indicator=True, nearest=()):
    c = ConsolidatedIndicator(indicator=DocIndicator(doc_indicator_id=did, doc_id="d1", indicator_text=text, is_indicator=is_indicator, location="p.2"), matches=list(matches), nearest_indikit=list(nearest))
    from pipeline.agent.consolidate import choose_primary

    c.primary, c.alternate_ids = choose_primary(c.matches)
    return c


def test_needs_review_rules_and_tables(tmp_path: Path):
    verb = LinkedMatch(indikit_id="5138", indikit_name="Handwashing", match_type="verbatim", rerank_score=0.9, text_similarity=95.0)
    low = LinkedMatch(indikit_id="5138", indikit_name="Handwashing", match_type="verbatim", rerank_score=0.9, text_similarity=20.0)
    close = LinkedMatch(indikit_id="5330", indikit_name="Chlorine", match_type="close", rerank_score=0.5)
    rev = LinkedMatch(indikit_id="5140", indikit_name="Latrine", match_type="close", source="reverse_check", rerank_score=0.4)
    assert needs_review(_ci("D001", "a", [verb])) == (False, "")
    assert needs_review(_ci("D002", "b", [close]))[1] == "close match"
    assert "has alternates" in needs_review(_ci("D003", "c", [verb, close]))[1]
    assert "reverse-check" in needs_review(_ci("D004", "d", [rev]))[1]
    assert "low similarity" in needs_review(_ci("D005", "e", [low]))[1]
    results = [
        DocumentResult(doc_id="d1", country="testland", region="africa", relative_path="p/a.docx", consolidated=[_ci("D001", "% of hh with soap", [verb]), _ci("D002", "chlorine samples", [close]), _ci("D003", "Average km migrated by zebra herds", nearest=[("9000", 0.4)]), _ci("D004", "Train 30 people", is_indicator=False), _ci("D005", "Average km migrated by zebra herds per year", nearest=[("9000", 0.3)])],
                       pairs_selected=10, pairs_asked=8, pairs_cached=2, pairs_present=2, llm_calls=8, cost_usd=0.01, novel_stats={"filtered_activities": 1}, reverse_stats={"recovered": 0, "candidates": 2}),
        DocumentResult(doc_id="d2", country="otherland", region="asia", relative_path="p/b.pdf", indicators_elsewhere=True, referenced_documents=["Annex C: Logical Framework"], skipped_reason="indicators in another document (reported, not queried)"),
    ]
    paths = write_run_outputs(results, METADATA, tmp_path)
    di = pd.read_csv(paths["document_indicators"], dtype={"indikit_id": str})
    assert list(di.columns) == DOC_IND_COLUMNS and len(di) == 5 and di["needs_review"].sum() == 1 and di.loc[di["doc_indicator_id"] == "D001", "match_type"].iloc[0] == "verbatim"
    assert set(di["match_type"]) == {"verbatim", "close", "none"} and di.loc[di["doc_indicator_id"] == "D003", "nearest_indikit"].iloc[0] == "9000 (0.40)"
    mx = pd.read_csv(paths["disclosure_matrix"], dtype={"indikit_id": str})
    assert list(mx.columns) == MATRIX_COLUMNS and set(mx["indikit_id"]) == {"5138", "5330"} and mx["sector"].str.startswith("Water").all()
    nv = pd.read_csv(paths["novel_indicators"])
    assert list(nv.columns) == NOVEL_COLUMNS and set(nv["doc_indicator_id"]) == {"D003", "D005"} and nv["cluster_id"].nunique() == 1   # similar texts cluster together
    sm = pd.read_csv(paths["document_summary"])
    assert list(sm.columns) == SUMMARY_COLUMNS and len(sm) == 2
    d1 = sm.set_index("doc_id").loc["d1"]
    assert d1["total_indicators"] == 4 and d1["verbatim"] == 1 and d1["close"] == 1 and d1["none"] == 2 and d1["pct_matched"] == 50.0 and d1["activities_filtered"] == 1
    d2 = sm.set_index("doc_id").loc["d2"]
    assert d2["note"] == "indicators are in another document: Annex C: Logical Framework" and bool(d2["indicators_elsewhere"])
    md = (tmp_path / "portfolio_summary.md").read_text()
    assert "## By country" in md and "never used" in md and (tmp_path / "portfolio_by_country.csv").exists() and (tmp_path / "portfolio_novel_clusters.csv").exists()
    tot = pd.read_csv(tmp_path / "portfolio_totals.csv").iloc[0]
    assert tot["document_indicators"] == 4 and tot["indikit_indicators_used"] == 2 and tot["indikit_indicators_total"] == 4
    report = document_report_markdown("d1", di, sm)
    assert report.startswith("| Indicator in document | IndiKit indicator (ID) | Match |") and "| % of hh with soap | Handwashing (5138) | verbatim |" in report and "Indicators in document: 4; verbatim: 1; close: 1; total matched: 2" in report
    assert "Referenced document: Annex C: Logical Framework" in document_report_markdown("d2", di, sm)
    latest = copy_latest(tmp_path, tmp_path / "latest")
    assert (latest / "document_summary.csv").exists() and (latest / "LATEST_RUN.txt").exists()
    assert cluster_texts(["a b c", "a b c d", "zzz"], None, 0.8) == [0, 0, 1]


def test_pilot_selection_prefers_gold_then_spreads():
    base = dict(relative_path="x", region="africa", sha256="0" * 64, n_logframe_tables=1)
    m = {f"d{i}": ManifestRow(doc_id=f"d{i}", country=c, file_type=t, **base) for i, (c, t) in enumerate([("a", "docx"), ("a", "pdf"), ("b", "docx"), ("b", "xlsx"), ("c", "pdf"), ("c", "docx"), ("d", "docx")])}
    m["nolf"] = ManifestRow(doc_id="nolf", country="z", file_type="docx", relative_path="y", region="africa", sha256="0" * 64)
    ids = pilot_doc_ids(m, [("d6", True, "dev"), ("d0", False, "dev"), ("d1", True, "test")], size=4)
    assert ids[0] == "d6" and len(ids) == 4 and "nolf" not in ids and len({m[i].country for i in ids}) == 4
    assert pilot_doc_ids(m, None, size=3) == ["d0", "d3", "d4"]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_agent_fixture(tmp_path_factory.mktemp("e2e"))


def _runner(fx, tmp_path, run_id, provider, **cfg):
    cfg = RunConfig(run_id=run_id, provider="mock", answer_model="m", extraction_model="m", tau=0.2, top_m=3, max_calls=cfg.pop("max_calls", 100), max_cost_usd=1.0, workers=2, **cfg)
    return Runner(cfg, provider=provider, metadata=METADATA, manifest=fx.manifest, retriever=fx.retriever, embedder=fx.embedder, embedding_cache=EmbeddingCache(None), cache=ResponseCache(tmp_path / "cache"),
                  usage=UsageTracker(cfg.max_calls, cfg.max_cost_usd, prices={"m": (0.1, 0.1)}, path=tmp_path / "runs" / run_id / "usage.json"), kb_dir=fx.kb_dir, runs_dir=tmp_path / "runs", pair_scores=fx.pair_scores)


def test_end_to_end_synthetic_run_produces_all_outputs(fx, tmp_path: Path):
    provider = MockLLMProvider(answer_from_evidence)
    runner = _runner(fx, tmp_path, "r1", provider)
    results, stop = runner.run([fx.docx_id, fx.pdf_id], progress=None)
    assert stop is None and len(results) == 2 and all(not r.skipped_reason for r in results)
    out = runner.finalize(copy_to_latest=False, extrapolate_to=100)
    run_dir = tmp_path / "runs" / "r1"
    for name in ("selected_pairs.csv", "selected_pairs_per_document.csv", "answers.jsonl", "usage.json", "document_indicators.csv", "disclosure_matrix.csv", "novel_indicators.csv", "document_summary.csv", "portfolio_summary.md", "run_report.md", "run_info.json", "run_config.json"):
        assert (run_dir / name).exists(), name
    di = pd.read_csv(run_dir / "document_indicators.csv", dtype={"indikit_id": str})
    assert (di["match_type"] == "verbatim").any() and (di["indikit_id"] == "5138").any()
    # every reported match carries a quote that is found in the document's knowledge base
    from pipeline.preprocess.knowledge_base import load_kb
    from pipeline.agent.answer import text_found

    for _, r in di[di["match_type"] != "none"].iterrows():
        kb = load_kb(r["doc_id"], fx.kb_dir)
        corpus = "\n".join(e.display_text for es in kb.entries.values() for e in es)
        assert text_found(str(r["evidence_quote"]), corpus)[0], r["evidence_quote"]
    mx = pd.read_csv(run_dir / "disclosure_matrix.csv")
    assert len(mx) >= 2 and set(mx["doc_id"]) == {fx.docx_id, fx.pdf_id}
    sm = pd.read_csv(run_dir / "document_summary.csv")
    assert (sm["pairs_asked"] > 0).all() and (sm["llm_calls"] > 0).all()
    info = out["info"]
    assert info["pairs_asked"] > 0 and info["usage"]["calls"] == provider.calls_made and info["extrapolation"]["documents"] == 100 and info["config_fingerprint"]
    report = (run_dir / "run_report.md").read_text()
    assert "## Extrapolation" in report and "prompt_version" in report
    # resume: a second runner over the same run dir makes no new calls
    provider2 = MockLLMProvider(answer_from_evidence)
    runner2 = _runner(fx, tmp_path, "r1", provider2)
    results2, _ = runner2.run([fx.docx_id, fx.pdf_id], progress=None)
    assert len(results2) == 2 and provider2.calls_made == 0
    # force: cached responses are reused (no live calls) but documents are reprocessed
    runner3 = _runner(fx, tmp_path, "r1", MockLLMProvider(answer_from_evidence), force=False)
    assert len(runner3.load_results()) == 2


def test_budget_stop_is_clean_and_resumable(fx, tmp_path: Path):
    provider = MockLLMProvider(answer_from_evidence)
    runner = _runner(fx, tmp_path, "r2", provider, max_calls=2)
    results, stop = runner.run([fx.docx_id, fx.pdf_id], progress=None, check_estimate=False)
    assert stop and "MAX_CALLS_PER_RUN" in stop and provider.calls_made == 2
    assert (tmp_path / "runs" / "r2" / "answers.jsonl").exists()
    out = runner.finalize()
    assert (tmp_path / "runs" / "r2" / "run_report.md").exists()
    # the estimate guard refuses before any call when the selection exceeds the limits
    with pytest.raises(BudgetExceeded, match="estimate exceeds"):
        _runner(fx, tmp_path, "r3", MockLLMProvider(answer_from_evidence), max_calls=1).run([fx.docx_id, fx.pdf_id], progress=None)
    with pytest.raises(RuntimeError, match="confirm-full-run"):
        _runner(fx, tmp_path, "r4", MockLLMProvider(answer_from_evidence), mode="full").run([fx.docx_id], progress=None)
    # hard-skipped documents are reported, never queried
    fx.manifest[fx.pdf_id] = fx.manifest[fx.pdf_id].model_copy(update={"indicators_elsewhere": True, "n_logframe_tables": 0, "n_logframe_sections": 0})
    try:
        p5 = MockLLMProvider(answer_from_evidence)
        r5 = _runner(fx, tmp_path, "r5", p5)
        res5, _ = r5.run([fx.pdf_id], progress=None)
        assert res5[0].skipped_reason and p5.calls_made == 0 and res5[0].referenced_documents
        r5.finalize()
        sm = pd.read_csv(tmp_path / "runs" / "r5" / "document_summary.csv")
        assert sm.iloc[0]["note"].startswith("indicators are in another document: Annex O")
    finally:
        fx.manifest[fx.pdf_id] = fx.manifest[fx.pdf_id].model_copy(update={"indicators_elsewhere": False, "n_logframe_tables": 1})


def test_run_lock_refuses_second_process(tmp_path: Path):
    import os

    from pipeline.agent.orchestrate import RunLock

    with RunLock(tmp_path):
        assert (tmp_path / "RUNNING.pid").read_text() == str(os.getpid())
        with RunLock(tmp_path):      # same process may re-enter
            pass
    assert not (tmp_path / "RUNNING.pid").exists()
    (tmp_path / "RUNNING.pid").write_text("999999999")   # a dead pid is ignored
    with RunLock(tmp_path):
        pass
    (tmp_path / "RUNNING.pid").write_text(str(os.getppid()))   # a live foreign pid blocks
    with pytest.raises(RuntimeError, match="already running"):
        with RunLock(tmp_path):
            pass


def test_clean_results_and_indikit_table():
    from pipeline.agent.models import DocumentResult
    from pipeline.agent.outputs import INDIKIT_COLUMNS, clean_results, indikit_table

    ok = DocumentResult(doc_id="a", pairs_failed=0)
    failed = DocumentResult(doc_id="b", pairs_failed=3)
    crashed = DocumentResult(doc_id="c", errors=["document failed: boom"])
    skipped = DocumentResult(doc_id="d", skipped_reason="indicators in another document")
    assert [r.doc_id for r in clean_results([ok, failed, crashed, skipped])] == ["a", "d"]
    df = indikit_table(METADATA)
    assert list(df.columns) == INDIKIT_COLUMNS and len(df) == len(METADATA) and df["indikit_id"].is_unique


def test_refresh_metadata_dependent_outputs_rewrites_on_site_flags(tmp_path: Path):
    """After a re-scrape the on_site / export_only columns follow the new metadata; answers and novel clusters are untouched."""
    from pipeline.agent.outputs import refresh_metadata_dependent_outputs

    verb = LinkedMatch(indikit_id="5138", indikit_name="Handwashing", match_type="verbatim", rerank_score=0.9, text_similarity=95.0)
    results = [DocumentResult(doc_id="d1", country="testland", region="africa", relative_path="p/a.docx", consolidated=[_ci("D001", "% of hh with soap", [verb]), _ci("D003", "Average km migrated by zebra herds", nearest=[("9000", 0.4)])], pairs_selected=2, pairs_asked=2, pairs_present=1)]
    run_dir = tmp_path / "run"
    (run_dir / "docs").mkdir(parents=True)
    (run_dir / "docs" / "d1.json").write_text(results[0].model_dump_json(), encoding="utf-8")
    first = write_run_outputs(results, METADATA, run_dir)
    assert pd.read_csv(first["document_indicators"], dtype=str).loc[0, "export_only"] == "False"
    novel_before = (run_dir / "novel_indicators.csv").read_text(encoding="utf-8")
    gone = {k: (v.model_copy(update={"on_site": False}) if k == "5138" else v) for k, v in METADATA.items()}
    paths = refresh_metadata_dependent_outputs(run_dir, gone, note="refreshed")
    di = pd.read_csv(paths["document_indicators"], dtype=str)
    assert di.loc[di["doc_indicator_id"] == "D001", "export_only"].iloc[0] == "True" and di.loc[di["doc_indicator_id"] == "D001", "match_type"].iloc[0] == "verbatim"
    mx = pd.read_csv(paths["disclosure_matrix"], dtype=str)
    assert mx.loc[mx["indikit_id"] == "5138", "on_site"].iloc[0] == "False"
    assert (run_dir / "novel_indicators.csv").read_text(encoding="utf-8") == novel_before
    assert "refreshed" in (run_dir / "portfolio_summary.md").read_text(encoding="utf-8")
