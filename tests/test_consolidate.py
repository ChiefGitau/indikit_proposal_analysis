"""Consolidation: linking, primary / alternate selection, reverse check (docs/phase_3.md Task 5B / 5C)."""
from __future__ import annotations

import pytest

from pipeline.agent.consolidate import IndicatorIndex, choose_primary, consolidate, find_counterpart, reverse_check
from pipeline.agent.llm import MockLLMProvider
from pipeline.agent.models import DocIndicator, LinkedMatch, PairAnswer, ValidatedMatch
from pipeline.preprocess.embeddings import HashingEmbedder
from tests.fixtures_agent import METADATA


def _vm(iid, text, mt="close", row_id=None, score=0.5, blocks=()):
    return ValidatedMatch(indikit_id=iid, document_indicator_text=text, match_type=mt, row_id=row_id, rerank_score=score, block_ids=list(blocks), reason="r", evidence_quote=text, baseline="1", target="2")


def _doc(did, text, row_id=None, blocks=(), is_indicator=True):
    return DocIndicator(doc_indicator_id=did, doc_id="d", indicator_text=text, row_id=row_id, block_ids=list(blocks), extraction_method="table_code", is_indicator=is_indicator)


def test_linking_by_row_block_and_text():
    inds = [_doc("D001", "% of households with handwashing facilities with soap and water", row_id="r1"), _doc("D002", "Number of boreholes rehabilitated", blocks=["b2"]), _doc("D003", "Totally different", row_id="r9")]
    assert find_counterpart(_vm("5138", "different words", row_id="r1"), inds).doc_indicator_id == "D001"
    assert find_counterpart(_vm("5138", "x", blocks=["b2"]), inds).doc_indicator_id == "D002"
    assert find_counterpart(_vm("5138", "Households with handwashing facilities with soap and water (%)"), inds).doc_indicator_id == "D001"
    assert find_counterpart(_vm("5138", "zebra herds"), inds) is None


def test_consolidate_primary_alternates_and_disclosure_only():
    inds = [_doc("D001", "% of households with handwashing facilities with soap and water", row_id="r1"), _doc("D002", "Number of latrines", row_id="r2")]
    answers = [PairAnswer(doc_id="d", indikit_id="5138", disclosure="present", matches=[_vm("5138", "% of households with handwashing facilities with soap and water", "verbatim", "r1", 0.9)]),
               PairAnswer(doc_id="d", indikit_id="5330", disclosure="present", matches=[_vm("5330", "% of households with handwashing facilities with soap and water", "close", "r1", 0.95)]),
               PairAnswer(doc_id="d", indikit_id="9000", disclosure="present", matches=[_vm("9000", "Average km migrated by zebras", "close", None, 0.3)])]
    out = consolidate("d", answers, inds, METADATA)
    by = {c.indicator.doc_indicator_id: c for c in out}
    assert by["D001"].primary.indikit_id == "5138" and by["D001"].primary.match_type == "verbatim" and by["D001"].alternate_ids == ["5330"] and by["D001"].primary.indikit_name.startswith("Handwashing")
    assert by["D002"].primary is None and by["D002"].match_type == "none"
    extra = by["D003"]
    assert extra.indicator.source == "disclosure_only" and extra.primary.indikit_id == "9000" and extra.indicator.baseline == "1"
    # same type -> higher re-rank score wins
    p, alts = choose_primary([LinkedMatch(indikit_id="a", match_type="close", rerank_score=0.2), LinkedMatch(indikit_id="b", match_type="close", rerank_score=0.8)])
    assert p.indikit_id == "b" and alts == ["a"]


def test_reverse_check_recovers_and_labels(tmp_path):
    inds = [_doc("D001", "Number of latrines constructed in schools", row_id="r1"), _doc("D002", "Average kilometres migrated by zebra herds", row_id="r2"), _doc("D003", "Train 30 people", row_id="r3", is_indicator=False)]
    cons = consolidate("d", [], inds, METADATA)
    index = IndicatorIndex(METADATA, HashingEmbedder(256))
    assert index.nearest(["average kilometres migrated by zebra herds"], 2)[0][0][0] == "9000"
    provider = MockLLMProvider(lambda s, u, schema, m: {"matches": [{"indikit_id": "5140", "match_type": "close", "reason": "counts latrines"}], "reason": "ok"} if 'Document indicator\n"Number of latrines' in u else {"matches": [{"indikit_id": "0000", "match_type": "close", "reason": "not a candidate"}], "reason": "x"})
    st = reverse_check("d", cons, index, provider=provider, model="m", top_n=3)
    assert st.candidates == 2 and st.checked == 2 and st.recovered == 1 and st.calls == 2
    by = {c.indicator.doc_indicator_id: c for c in cons}
    assert by["D001"].primary.indikit_id == "5140" and by["D001"].primary.source == "reverse_check" and by["D001"].primary.rerank_score is not None
    assert by["D002"].primary is None and by["D002"].nearest_indikit and by["D002"].nearest_indikit[0][0] == "9000"   # a non-candidate id is ignored
    assert by["D003"].nearest_indikit == [] and not by["D003"].matches
    # no provider: nearest neighbours only
    cons2 = consolidate("d", [], inds, METADATA)
    st2 = reverse_check("d", cons2, index, provider=None)
    assert st2.calls == 0 and st2.candidates == 2 and cons2[0].nearest_indikit
