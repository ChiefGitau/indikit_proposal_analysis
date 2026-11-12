"""Novel-indicator pass: code-first rows incl. multi-indicator cells, LLM extraction, dedupe, ids (Task 5A)."""
from __future__ import annotations

import pytest

from pipeline.agent.llm import MockLLMProvider, ResponseCache, UsageTracker
from pipeline.agent.novel_pass import assign_ids, dedupe_doc_indicators, extract_rows_in_code, legacy_cache_lookup, looks_like_activity, run_novel_pass, split_indicator_cell, text_sections
from pipeline.preprocess.logframe import build_rows, classify_table, logframe_content
from pipeline.preprocess.models import Block, Table
from tests.fixtures_agent import answer_from_evidence, build_agent_fixture


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_agent_fixture(tmp_path_factory.mktemp("novel"))


def test_split_cells_and_activity_filter():
    assert split_indicator_cell("1. % of HH with soap and water\n2. Number of latrines built\n• 5 VSLAs supported") == ["% of HH with soap and water", "Number of latrines built", "5 VSLAs supported"]
    assert split_indicator_cell("Indicator 1.1: # of people reached; ") == ["# of people reached"]
    assert split_indicator_cell("Output 1.1: # of people reached") == ["# of people reached"]
    assert split_indicator_cell("") == [] and split_indicator_cell("12%") == []
    assert looks_like_activity("Train 30 farmers on CSA") and not looks_like_activity("Number of farmers trained") and not looks_like_activity("% of farmers reporting")


def test_code_first_extraction_with_multi_indicator_cell_and_unmapped_table():
    t = Table(table_id="d:t0", doc_id="d", source="docx", header=["Intervention logic", "Indicators", "Baseline", "Target", "Means of verification"],
              rows=[["Outcome 1: improved WASH", "- % of HH with soap and water\n- Number of people reached", "12% / 0", "60% / 2,500", "survey"], ["Output 1.1", "Train 30 hygiene promoters", "0", "30", "reports"]], n_rows=2, n_cols=5)
    classify_table(t)
    assert t.is_logframe and t.column_roles[1] == "indicator"
    u = Table(table_id="d:t1", doc_id="d", source="docx", header=["A", "B"], rows=[["x", "y"]], n_rows=1, n_cols=2, is_logframe=True)
    content = {"tables": [{"table": t, "rows": build_rows(t)}, {"table": u, "rows": build_rows(u)}], "blocks": []}
    inds, unmapped = extract_rows_in_code("d", content)
    assert [i.indicator_text for i in inds] == ["% of HH with soap and water", "Number of people reached", "Train 30 hygiene promoters"]
    assert inds[0].baseline == "12% / 0" and inds[0].target == "60% / 2,500" and inds[0].means_of_verification == "survey" and inds[0].result_context.startswith("Outcome 1") and inds[0].level == "outcome"
    assert inds[0].row_id == "d:t0:r0000" and inds[0].extraction_method == "table_code" and inds[0].is_indicator and not inds[2].is_indicator and inds[2].level == "output"
    assert len(unmapped) == 1 and unmapped[0]["table"].table_id == "d:t1"


def test_text_sections_batch_by_heading_and_dedupe_ids():
    blocks = [Block(block_id="b1", doc_id="d", kind="heading", text="Key indicators", order=1, style_info={"heading_level": 2}, is_logframe=True),
              Block(block_id="b2", doc_id="d", kind="list_item", text="Number of boreholes rehabilitated (target 12)", order=2, heading_path=["Key indicators"], is_logframe=True),
              Block(block_id="b3", doc_id="d", kind="paragraph", text="Percent of households with soap at the handwashing station is tracked quarterly.", order=3, heading_path=["Key indicators"], is_logframe=True)]
    secs = text_sections({"blocks": blocks, "tables": []})
    assert len(secs) == 1 and secs[0]["title"] == "Key indicators" and secs[0]["block_ids"] == ["b1", "b2", "b3"] and "boreholes" in secs[0]["text"]
    from pipeline.agent.models import DocIndicator

    a = DocIndicator(doc_indicator_id="", doc_id="d", indicator_text="Number of boreholes rehabilitated", extraction_method="table_code", order=2, table_id="t")
    b = DocIndicator(doc_indicator_id="", doc_id="d", indicator_text="number of boreholes  rehabilitated", extraction_method="text_llm", order=5, target="12")
    c = DocIndicator(doc_indicator_id="", doc_id="d", indicator_text="% of HH with soap", extraction_method="text_llm", order=6)
    kept, merged = dedupe_doc_indicators([b, c, a])
    assert [k.indicator_text for k in kept] == ["Number of boreholes rehabilitated", "% of HH with soap"] and kept[0].target == "12" and merged
    ids = [d.doc_indicator_id for d in assign_ids(kept)]
    assert ids == ["D001", "D002"]


def test_run_novel_pass_end_to_end_with_mock(fx, tmp_path):
    content = logframe_content(fx.pdf_id, fx.kb_dir)
    assert content["tables"] and content["blocks"]
    usage = UsageTracker(max_calls=20, max_cost_usd=1, prices={"m": (0, 0)})
    provider = MockLLMProvider(answer_from_evidence, usage=usage)
    res = run_novel_pass(fx.pdf_id, provider=provider, model="m", cache=ResponseCache(tmp_path / "c"), kb_dir=fx.kb_dir)
    texts = [d.indicator_text for d in res.indicators]
    assert any("handwashing" in t.lower() for t in texts) and any("boreholes" in t.lower() for t in texts) and any("chlorine" in t.lower() for t in texts)
    assert res.stats["table_code"] >= 3 and res.stats["llm_calls"] >= 1 and all(d.doc_indicator_id.startswith("D") for d in res.indicators)
    assert all(d.table_id for d in res.indicators if d.extraction_method == "table_code")
    # second run served from the cache, no new calls; without a provider the text sections are reported, not called
    res2 = run_novel_pass(fx.pdf_id, provider=provider, model="m", cache=ResponseCache(tmp_path / "c"), kb_dir=fx.kb_dir)
    assert res2.stats["llm_cached"] >= 1 and usage.calls == res.stats["llm_calls"]
    res3 = run_novel_pass(fx.pdf_id, provider=None, kb_dir=fx.kb_dir)
    assert res3.errors and res3.stats["table_code"] == res.stats["table_code"]
    assert legacy_cache_lookup("d", "text", "extract-v2", "m") is None
