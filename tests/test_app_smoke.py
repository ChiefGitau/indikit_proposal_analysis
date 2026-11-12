"""Streamlit app smoke test (docs/phase_3.md Task 10): every page imports and runs in bare mode, with and without
pipeline results, without any LLM call. Skipped when streamlit / plotly are not installed in the test environment."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

st = pytest.importorskip("streamlit")
pytest.importorskip("plotly")

APP_DIR = Path(__file__).resolve().parents[1] / "MEAL_Proposal-main" / "MEAL_Proposal-main" / "zoa_indikit_app"
PAGES = sorted(p.name for p in (APP_DIR / "pages").glob("*.py"))


def _purge_src(monkeypatch):
    for m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        monkeypatch.delitem(sys.modules, m)


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    """Returns ``prepare(latest_dir)``: points the app at ``latest_dir`` (env var read at import), reloads ``src``,
    and redirects the analysis log into tmp_path so the repository's log files are never touched."""
    monkeypatch.chdir(APP_DIR)
    monkeypatch.syspath_prepend(str(APP_DIR))
    monkeypatch.setattr(st, "stop", lambda: (_ for _ in ()).throw(SystemExit(0)))
    # the pages use width="stretch" (Streamlit >= 1.49); older installs get use_container_width=True instead
    major, minor = (int(x) for x in st.__version__.split(".")[:2])
    if (major, minor) < (1, 49):
        for name in ("dataframe", "plotly_chart", "data_editor", "table", "button", "download_button"):
            orig = getattr(st, name)

            def _compat(*a, _orig=orig, **kw):
                if isinstance(kw.get("width"), str):
                    kw.pop("width")
                    kw["use_container_width"] = True
                return _orig(*a, **kw)

            monkeypatch.setattr(st, name, _compat)

    def prepare(latest_dir: Path):
        monkeypatch.setenv("ZOA_PIPELINE_LATEST", str(latest_dir))
        _purge_src(monkeypatch)
        import src.analysis_log as al

        monkeypatch.setattr(al, "LOG_DIR", tmp_path / "log")
        monkeypatch.setattr(al, "MANIFEST", tmp_path / "log" / "manifest.json")
        return tmp_path

    return prepare


def _run(page: str):
    try:
        runpy.run_path(str(APP_DIR / "pages" / page), run_name="__main__")
    except SystemExit:
        pass


def _fake_latest(tmp_path: Path) -> Path:
    latest = tmp_path / "latest"
    latest.mkdir()
    pd.DataFrame([{"doc_id": "d1", "relative_path": "proposals/africa/burundi/small/a.docx", "country": "burundi", "region": "africa", "language": "en", "file_type": "docx", "parse_status": "ok", "total_indicators": 2, "verbatim": 1, "close": 0, "none": 1, "pct_matched": 50.0,
                    "indicators_elsewhere": False, "indicators_also_elsewhere": False, "referenced_documents": "", "note": "", "pairs_selected": 5, "pairs_asked": 5, "pairs_cached": 0, "pairs_present": 1, "llm_calls": 5, "cost_usd": 0.01, "unsupported_dropped": 0, "reverse_recovered": 0, "activities_filtered": 0, "errors": "", "skipped_reason": ""},
                  {"doc_id": "d2", "relative_path": "proposals/Asia/yemen/small/b.pdf", "country": "yemen", "region": "Asia", "language": "en", "file_type": "pdf", "parse_status": "ok", "total_indicators": 0, "verbatim": 0, "close": 0, "none": 0, "pct_matched": "", "indicators_elsewhere": True, "indicators_also_elsewhere": False,
                   "referenced_documents": "Annex C: Logical Framework", "note": "indicators are in another document: Annex C: Logical Framework", "pairs_selected": 0, "pairs_asked": 0, "pairs_cached": 0, "pairs_present": 0, "llm_calls": 0, "cost_usd": 0, "unsupported_dropped": 0, "reverse_recovered": 0, "activities_filtered": 0, "errors": "", "skipped_reason": "indicators in another document"}]).to_csv(latest / "document_summary.csv", index=False)
    pd.DataFrame([{"doc_id": "d1", "doc_indicator_id": "D001", "indicator_text": "% of HH with soap", "level": "outcome", "baseline": "12%", "target": "60%", "unit": "%", "location": "p.2", "match_type": "verbatim", "indikit_id": "5138", "indikit_name": "Handwashing", "alternate_ids": "", "source": "disclosure", "reason": "same", "evidence_quote": "Row: % of HH with soap", "needs_review": False, "needs_review_reason": "", "country": "burundi", "region": "africa", "is_indicator": True, "extraction_method": "table_code", "text_similarity": 90, "nearest_indikit": "", "export_only": False},
                  {"doc_id": "d1", "doc_indicator_id": "D002", "indicator_text": "zebra km", "level": "output", "baseline": "", "target": "", "unit": "", "location": "p.3", "match_type": "none", "indikit_id": "", "indikit_name": "", "alternate_ids": "", "source": "extraction", "reason": "", "evidence_quote": "", "needs_review": False, "needs_review_reason": "", "country": "burundi", "region": "africa", "is_indicator": True, "extraction_method": "text_llm", "text_similarity": "", "nearest_indikit": "9000 (0.4)", "export_only": False}]).to_csv(latest / "document_indicators.csv", index=False)
    pd.DataFrame([{"doc_id": "d1", "indikit_id": "5138", "indikit_name": "Handwashing", "sector": "WASH", "match_type": "verbatim", "n_document_indicators": 1, "doc_indicator_ids": "D001", "sources": "disclosure", "on_site": True, "country": "burundi", "region": "africa"}]).to_csv(latest / "disclosure_matrix.csv", index=False)
    pd.DataFrame([{"doc_id": "d1", "doc_indicator_id": "D002", "indicator_text": "zebra km", "level": "output", "baseline": "", "target": "", "unit": "", "location": "p.3", "country": "burundi", "region": "africa", "nearest_indikit_ids": "9000", "nearest_similarities": "0.4", "cluster_id": 0}]).to_csv(latest / "novel_indicators.csv", index=False)
    pd.DataFrame([{"cluster_id": 0, "size": 1, "documents": 1, "countries": "burundi", "example": "zebra km", "nearest_indikit": "9000"}]).to_csv(latest / "portfolio_novel_clusters.csv", index=False)
    pd.DataFrame([{"indikit_id": "5330", "indikit_name": "Chlorine", "sector": "WASH", "on_site": True}]).to_csv(latest / "portfolio_never_used_indikit.csv", index=False)
    pd.DataFrame([{"indikit_id": "5138", "indikit_name": "Handwashing", "sector": "WASH", "documents": 1, "verbatim": 1, "close": 0}]).to_csv(latest / "portfolio_most_used_indikit.csv", index=False)
    (latest / "portfolio_summary.md").write_text("# Portfolio summary\n")
    (latest / "run_report.md").write_text("# Run report\n")
    (latest / "LATEST_RUN.txt").write_text("run-test\n")
    return latest


@pytest.mark.parametrize("page", PAGES)
def test_every_page_imports_without_results(app_env, tmp_path, page):
    if page.startswith("9_"):
        pytest.skip("Site Comparison scrapes the live site")
    app_env(tmp_path / "nope")
    _run(page)


def test_proposal_pages_with_pipeline_results(app_env, tmp_path):
    latest = _fake_latest(tmp_path)
    app_env(latest)
    _run("8_Proposal_Overview.py")
    _run("10_Indicator_Matching.py")
    import src.proposals as props
    import src.results as res

    assert res.has_results() and not props.is_demo()
    df = props.load_proposals()
    assert len(df) == 2 and set(df["country_name"]) == {"Burundi", "Yemen"} and bool(df.set_index("id").loc["d2", "indicators_elsewhere"]) and "title" in df.columns
    assert props.covered_countries()["proposals"].sum() == 2 and props.proposal_sites().empty


def test_demo_fallback_keeps_dummy_columns(app_env, tmp_path):
    app_env(tmp_path / "nope")
    import src.proposals as props

    assert props.is_demo()
    df = props.load_proposals()
    assert set(props.OVERVIEW_COLS) <= set(df.columns) and len(df) >= 1 and len(props.proposal_sites()) >= 1


def test_app_makes_no_llm_calls(app_env, tmp_path):
    app_env(tmp_path / "nope")
    import src.analysis_log as al

    assert "indikit_analysis_log" not in str(al.LOG_DIR)   # the smoke tests never write the repository's log
    src = "\n".join(p.read_text(encoding="utf-8") for p in list((APP_DIR / "pages").glob("*.py")) + list((APP_DIR / "src").glob("*.py")))
    assert "openai" not in src.lower() and "pipeline.agent" not in src and "chat.completions" not in src
