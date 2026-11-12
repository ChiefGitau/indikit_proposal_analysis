"""LLM answering with post-validation and caching (docs/phase_3.md Task 3). Mock provider only."""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import config
from pipeline.agent import prompts
from pipeline.agent.answer import ProgressFile, answer_pair, answer_pairs, evidence_fingerprint, text_found, validate_matches
from pipeline.agent.llm import BudgetExceeded, MockLLMProvider, ResponseCache, UsageTracker
from pipeline.agent.models import DisclosureAnswer
from pipeline.agent.retrieve import Evidence, RetrievalResult
from tests.fixtures_agent import HANDWASH, METADATA, UNRELATED, answer_from_evidence, build_agent_fixture


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_agent_fixture(tmp_path_factory.mktemp("agent"))


def test_prompt_has_five_parts_labels_and_flags(fx):
    res = fx.retrieve(fx.docx_id, HANDWASH)
    system, user = prompts.build_disclosure_messages(HANDWASH, prompts.evidence_items(res.evidence), use_knowledge=True)
    full = system + "\n\n" + user
    for h in prompts.PART_HEADERS:
        assert h in full
    assert full.index(prompts.PART_HEADERS[0]) < full.index(prompts.PART_HEADERS[1]) < full.index(prompts.PART_HEADERS[2]) < full.index(prompts.PART_HEADERS[3]) < full.index(prompts.PART_HEADERS[4])
    assert "[E1] (" in user and "Logical framework" in user and HANDWASH.knowledge in user and "verbatim" in user
    assert "Does the reference content contain one or more indicators" in user and config.PROMPT_VERSION in user and config.RUBRIC_VERSION in user
    _, no_k = prompts.build_disclosure_messages(HANDWASH, prompts.evidence_items(res.evidence), use_knowledge=False)
    assert HANDWASH.knowledge not in no_k and "verbatim" in no_k and "knowledge off" in no_k
    assert "(no reference content retrieved)" in prompts.build_disclosure_prompt(HANDWASH, [])
    # Phase 1 string interface still works
    assert "[E1] (location unknown)\nsome text" in prompts.build_disclosure_prompt(HANDWASH, ["some text"])


def test_validate_matches_drops_unsupported_quotes():
    ev = [Evidence(entry_id="d:table:t0:r1", kb_type="table", entry_kind="row", display_text="Table: x [p.2]\nColumns: Indicator | Baseline\nRow: Indicator: % of households with handwashing facilities with soap and water | Baseline: 12%",
                   location="p.2", vector_score=0.5, rerank_score=0.9, table_id="t0", row_id="t0:r1")]
    ans = DisclosureAnswer(indikit_id="5138", disclosure="present", matches=[
        {"document_indicator_text": "% of households with handwashing facilities with soap and water", "match_type": "verbatim", "evidence_label": "E1", "evidence_quote": "Baseline: 12%", "baseline": "12 %", "target": "60%"},
        {"document_indicator_text": "Number of people trained on hygiene", "match_type": "close", "evidence_label": "E1", "evidence_quote": "people trained"},
        {"document_indicator_text": "% of households with handwashing", "match_type": "close", "evidence_label": "E7", "evidence_quote": "x"},
        {"document_indicator_text": "% of Households  with handwashing facilities with soap and water", "match_type": "verbatim", "evidence_label": "e1", "evidence_quote": ""},
    ])
    ok, bad = validate_matches(ans, ev, HANDWASH)
    assert len(ok) == 2 and len(bad) == 2
    assert ok[0].entry_id == "d:table:t0:r1" and ok[0].row_id == "t0:r1" and ok[0].baseline == "12 %" and ok[0].baseline_norm == "12" and ok[0].target_norm == "60" and ok[0].rerank_score == 0.9
    assert ok[0].text_similarity and ok[0].text_similarity > 80
    assert "not found" in bad[0].unsupported_reason and "does not exist" in bad[1].unsupported_reason
    assert text_found("soap and water", "SOAP   and water!")[0] and not text_found("zebra", "soap and water")[0]


def test_answer_pair_caches_and_counts_unsupported(fx, tmp_path: Path):
    usage = UsageTracker(max_calls=50, max_cost_usd=1.0, prices={"m": (1.0, 1.0)})
    provider = MockLLMProvider(answer_from_evidence, usage=usage)
    cache = ResponseCache(tmp_path / "cache")
    pa = answer_pair(fx.docx_id, HANDWASH, provider=provider, retriever=fx.retriever, cache=cache, model="m", kb_dir=fx.kb_dir)
    assert pa.disclosure == "present" and len(pa.matches) == 1 and pa.matches[0].match_type == "verbatim" and pa.matches[0].baseline == "12%" and pa.matches[0].target == "60%"
    assert pa.matches[0].table_id and pa.matches[0].row_id and pa.evidence_fingerprint and not pa.cached and pa.prompt_tokens > 0 and pa.model == "m"
    assert usage.calls == 1
    again = answer_pair(fx.docx_id, HANDWASH, provider=provider, retriever=fx.retriever, cache=cache, model="m", kb_dir=fx.kb_dir)
    assert again.cached and usage.calls == 1 and again.matches[0].document_indicator_text == pa.matches[0].document_indicator_text
    forced = answer_pair(fx.docx_id, HANDWASH, provider=provider, retriever=fx.retriever, cache=cache, model="m", kb_dir=fx.kb_dir, force=True)
    assert not forced.cached and usage.calls == 2
    # a different model / prompt flag is a different cache key
    answer_pair(fx.docx_id, HANDWASH, provider=provider, retriever=fx.retriever, cache=cache, model="m", kb_dir=fx.kb_dir, use_knowledge=False)
    assert usage.calls == 3
    absent = answer_pair(fx.docx_id, UNRELATED, provider=provider, retriever=fx.retriever, cache=cache, model="m", kb_dir=fx.kb_dir)
    assert absent.disclosure == "absent" and not absent.matches
    # an invented quote is dropped and counted
    liar = MockLLMProvider([{"indikit_id": "5138", "disclosure": "present", "matches": [{"document_indicator_text": "invented indicator text", "match_type": "verbatim", "evidence_label": "E1", "evidence_quote": "nope"}]}])
    pa2 = answer_pair(fx.docx_id, HANDWASH, provider=liar, retriever=fx.retriever, cache=None, model="m", kb_dir=fx.kb_dir)
    assert pa2.disclosure == "absent" and len(pa2.unsupported) == 1 and not pa2.matches
    # invalid JSON -> one retry -> failure recorded
    broken = MockLLMProvider(["{bad", "{still bad"])
    pa3 = answer_pair(fx.docx_id, HANDWASH, provider=broken, retriever=fx.retriever, cache=None, model="m", kb_dir=fx.kb_dir)
    assert pa3.disclosure == "failed" and pa3.attempts == 2 and pa3.error


def test_answer_pairs_progress_file_resumes_and_budget_stops(fx, tmp_path: Path):
    usage = UsageTracker(max_calls=2, max_cost_usd=10.0, prices={"m": (0, 0)})
    provider = MockLLMProvider(answer_from_evidence, usage=usage)
    pairs = [(fx.docx_id, METADATA[i]) for i in ("5138", "5330", "9000", "5140")]
    progress = ProgressFile(tmp_path / "answers.jsonl")
    out, stop = answer_pairs(pairs, provider=provider, progress=progress, workers=1, retriever=fx.retriever, cache=None, model="m", kb_dir=fx.kb_dir)
    assert stop is not None and len(out) == 2 and (tmp_path / "answers.jsonl").exists()
    # resume: the two finished pairs are skipped, a bigger budget finishes the rest
    usage2 = UsageTracker(max_calls=10, max_cost_usd=10.0, prices={"m": (0, 0)})
    provider2 = MockLLMProvider(answer_from_evidence, usage=usage2)
    progress2 = ProgressFile(tmp_path / "answers.jsonl")
    assert len(progress2.done) == 2
    out2, stop2 = answer_pairs(pairs, provider=provider2, progress=progress2, workers=2, retriever=fx.retriever, cache=None, model="m", kb_dir=fx.kb_dir)
    assert stop2 is None and len(out2) == 4 and usage2.calls == 2
    assert evidence_fingerprint([]) == evidence_fingerprint([])
