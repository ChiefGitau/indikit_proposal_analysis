"""Knowledge / SearchTerm extensions: deterministic baseline, overrides precedence, enrichment without network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.agent import answer as _answer
from pipeline.metadata import extensions as ext
from pipeline.metadata.models import Aspect, IndicatorMetadata


def make_record(**kw) -> IndicatorMetadata:
    base = dict(
        indikit_id="5620",
        name="Food Consumption Score",
        aspect=[Aspect(section="Sectoral", sector="Food Security and Livelihoods", sub_sector="Food Security")],
        kpi_wording_en="% of [specify the target group] households with acceptable Food Consumption Score (FCS) during [specify the period]",
        kpi_wording_fr="% de ménages [préciser le groupe cible] ayant un score de consommation alimentaire acceptable",
        purpose="The Food Consumption Score (FCS) measures the diversity and frequency of food consumption. It is a proxy of food security.",
        how_to_collect="Ask about the previous 7 days. Weight each food group. Then compute the score. Finally classify.",
        important_comments="FCS thresholds differ by country. Do not compare across seasons.",
    )
    base.update(kw)
    return IndicatorMetadata(**base)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Any real LLM call during these tests is a failure."""

    def boom(*a, **k):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(_answer, "_call_openai_with_retries", boom)
    monkeypatch.setattr(_answer, "_make_client", boom)


def test_strip_placeholders_and_key_phrases():
    assert ext.strip_placeholders("% of [specify the target group] households during [specify period]") == "% of households during"
    phrases = ext.key_phrases("% of households with handwashing facilities with soap and water at home", ext._STOPWORDS_EN)
    assert phrases == ["households", "handwashing facilities", "soap", "water", "home"]
    assert ext.find_abbreviations("acceptable Food Consumption Score (FCS)", "uses rCSI too", "no abbr") == ["FCS", "rCSI"]
    assert ext.find_abbreviations("FCSX is not FCS") == ["FCS"]


def test_baseline_search_terms_and_knowledge():
    rec = make_record()
    terms = ext.baseline_search_terms(rec)
    assert terms[0] == "food consumption score"
    assert "fcs" in terms and "food security and livelihoods" in terms and "food security" in terms
    assert all(t == t.lower() for t in terms) and len(terms) == len(set(terms)) and len(terms) <= 15
    assert not any("[" in t or "specify" in t for t in terms)
    knowledge = ext.baseline_knowledge(rec)
    assert knowledge.startswith("The Food Consumption Score (FCS) measures")
    assert "Ask about the previous 7 days. Weight each food group." in knowledge  # first two sentences
    assert "Then compute" not in knowledge
    assert "FCS thresholds differ by country." in knowledge and "Do not compare" not in knowledge
    assert len(knowledge.split()) <= 121


def test_knowledge_truncated_to_120_words_and_fallbacks():
    long_purpose = " ".join(f"w{i}" for i in range(300))
    rec = make_record(purpose=long_purpose, how_to_collect=None, important_comments=None)
    k = ext.baseline_knowledge(rec)
    assert len(k.split()) == 121 and k.endswith("…")
    rec2 = make_record(purpose=None, how_to_collect=None, important_comments=None)
    assert ext.baseline_knowledge(rec2) == "Measures: % of households with acceptable Food Consumption Score (FCS) during."
    rec3 = make_record(purpose=None, how_to_collect=None, important_comments=None, kpi_wording_en="")
    assert ext.baseline_knowledge(rec3) == ""


def test_apply_baseline_reports_exceptions():
    good = make_record()
    thin = make_record(indikit_id="9", name="Thin", aspect=[], kpi_wording_en="", kpi_wording_fr=None, purpose=None, how_to_collect=None, important_comments=None)
    exceptions = ext.apply_baseline([good, thin])
    assert good.search_terms and good.knowledge and good.field_sources["knowledge"] == "derived"
    assert thin.search_terms == ["thin"] and thin.knowledge is None
    assert exceptions == [("9", "only 1 search term(s)"), ("9", "empty knowledge")]


def test_overrides_win_and_name_mismatch_is_skipped(tmp_path: Path):
    rec = make_record()
    other = make_record(indikit_id="5138", name="Handwashing Facilities With Soap and Water", kpi_wording_en="% of households with handwashing facilities with soap and water at home")
    ext.apply_baseline([rec, other])
    rec.knowledge = "llm text"
    rec.field_sources["knowledge"] = "llm"
    yaml_path = tmp_path / "overrides.yaml"
    yaml_path.write_text(
        '"5620":\n  name: Food Consumption Score\n  knowledge: Manual note.\n  search_terms_add: ["FCS > 35", "Acceptable food consumption"]\n  search_terms_remove: ["food security"]\n'
        '"5138":\n  name: Something Else Entirely\n  search_terms_add: ["W8"]\n'
        '"9999":\n  search_terms_add: ["x"]\n',
        encoding="utf-8",
    )
    overrides = ext.load_overrides(yaml_path)
    applied, skipped = ext.apply_overrides([rec, other], overrides)
    assert applied == ["5620"]
    assert [s[0] for s in skipped] == ["5138", "9999"] and "name mismatch" in skipped[0][1]
    assert rec.knowledge == "Manual note." and rec.field_sources["knowledge"] == "override"
    assert "fcs > 35" in rec.search_terms and "acceptable food consumption" in rec.search_terms
    assert "food security" not in rec.search_terms
    assert "w8" not in other.search_terms
    assert ext.load_overrides(tmp_path / "missing.yaml") == {}


def test_names_match_tolerates_small_differences():
    assert ext.names_match("PIP: % of households who are implementing their PIP plans", "PIP of households who are implementing their PIP plans")
    assert ext.names_match(None, "anything")
    assert not ext.names_match("Food Consumption Score", "Handwashing Facilities With Soap and Water")


def test_enrichment_uses_cache_validates_and_never_runs_without_flag(tmp_path: Path):
    rec = make_record()
    ext.apply_baseline([rec])
    baseline_terms = list(rec.search_terms)
    calls: list[str] = []

    def fake_call(prompt, model_name, **kwargs):
        calls.append(prompt)
        assert "5620" in prompt and kwargs.get("response_format") == {"type": "json_object"}
        return 'Sure: {"knowledge": "Measures diet diversity and frequency. Does not count: number of meals per day.", "search_terms": ["FCS", "score de consommation alimentaire", "Acceptable FCS"]}'

    status = ext.enrich_record(rec, "test-model", cache_dir=tmp_path, call_fn=fake_call)
    assert status == "called" and len(calls) == 1
    assert rec.knowledge.startswith("Measures diet diversity") and rec.field_sources["knowledge"] == "llm"
    assert "score de consommation alimentaire" in rec.search_terms and baseline_terms[0] in rec.search_terms
    cached = list(tmp_path.glob("5620__enrich-v1__test-model.json"))
    assert len(cached) == 1 and json.loads(cached[0].read_text())["response"]["search_terms"][0] == "FCS"

    # second run is served from the cache: no call
    rec2 = make_record()
    ext.apply_baseline([rec2])
    assert ext.enrich_record(rec2, "test-model", cache_dir=tmp_path, call_fn=fake_call) == "cached" and len(calls) == 1
    assert rec2.knowledge == rec.knowledge

    # invalid response leaves the record untouched
    rec3 = make_record(indikit_id="5621")
    ext.apply_baseline([rec3])
    before = rec3.knowledge
    assert ext.enrich_record(rec3, "test-model", cache_dir=tmp_path, call_fn=lambda *a, **k: '{"search_terms": []}') == "invalid"
    assert rec3.knowledge == before and rec3.field_sources["knowledge"] == "derived"

    # enrich_records with the default client would hit the (blocked) LLM -> counted as failed, not raised
    stats = ext.enrich_records([make_record(indikit_id="5622")], model_name="m", cache_dir=tmp_path)
    assert stats["failed"] == 1 and stats["called"] == 0
