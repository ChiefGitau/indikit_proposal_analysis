"""Five-part disclosure prompt (paper Section 3.4.2) and the answer schema."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pipeline import config
from pipeline.agent import prompts
from pipeline.agent.models import DisclosureAnswer
from pipeline.metadata.models import Aspect, IndicatorMetadata, Quantity


def test_prompt_contains_all_five_parts_and_indicator_fields():
    ind = IndicatorMetadata(
        indikit_id="5179",
        name="Access to Adequate Shelter",
        aspect=[Aspect(section="Sectoral", sector="Shelter", sub_sector="Shelter")],
        kpi_wording_en="number of people having access to basic, safe and dignified shelter solutions",
        kpi_wording_fr="nombre de personnes ayant accès à un abri",
        levels=["Outcome"],
        topic="by sex and age",
        quantity=Quantity.number,
        knowledge="Counts people with adequate shelter; distribution of NFI kits does not count.",
        search_terms=["adequate shelter", "shelter solutions", "s1"],
    )
    prompt = prompts.build_disclosure_prompt(ind, [("p.12 logframe", "Number of people with access to safe shelter | baseline 0 | target 1,200")], prompts.MATCH_RUBRIC)
    for header in prompts.PART_HEADERS:
        assert header in prompt
    assert prompt.index(prompts.PART_HEADERS[0]) < prompt.index(prompts.PART_HEADERS[1]) < prompt.index(prompts.PART_HEADERS[2]) < prompt.index(prompts.PART_HEADERS[3]) < prompt.index(prompts.PART_HEADERS[4])
    assert "5179" in prompt and ind.kpi_wording_en in prompt and ind.knowledge in prompt
    assert "nombre de personnes" in prompt and "by sex and age" in prompt and "adequate shelter" in prompt
    assert "[E1] (p.12 logframe)" in prompt
    assert "verbatim" in prompt and "close" in prompt  # rubric
    assert '"disclosure": "present | absent"' in prompt
    assert config.PROMPT_VERSION in prompt
    assert prompts.PROMPT_VERSION == config.PROMPT_VERSION


def test_prompt_without_reference_content_says_so():
    ind = IndicatorMetadata(indikit_id="1", name="X")
    prompt = prompts.build_disclosure_prompt(ind, [], "rubric here")
    assert "(no reference content retrieved)" in prompt and "rubric here" in prompt and "(no knowledge note available)" in prompt


def test_disclosure_answer_schema():
    ok = DisclosureAnswer(indikit_id=5179, disclosure="Present", matches=[{"document_indicator_text": "x", "match_type": "Verbatim", "target": 1200, "level": "Outcome", "evidence_label": "[e2]", "reason": "same"}])
    m = ok.matches[0]
    assert ok.indikit_id == "5179" and m.match_type == "verbatim" and m.target == "1200" and m.level == "outcome" and m.evidence_label == "E2"
    absent = DisclosureAnswer(indikit_id="5179", disclosure="absent", matches=[])
    assert absent.disclosure == "absent" and absent.matches == []
    # present <=> matches non-empty (coerced)
    assert DisclosureAnswer(indikit_id="1", disclosure="present").disclosure == "absent"
    assert DisclosureAnswer(indikit_id="1", disclosure="absent", matches=[{"document_indicator_text": "x", "match_type": "close", "evidence_label": "E1"}]).disclosure == "present"
    with pytest.raises(ValidationError):
        DisclosureAnswer(indikit_id="5179", disclosure="present", matches=[{"document_indicator_text": "x", "match_type": "exact", "evidence_label": "E1"}])
    with pytest.raises(ValidationError):
        DisclosureAnswer(indikit_id="5179", disclosure="maybe")
    with pytest.raises(ValidationError):
        DisclosureAnswer(indikit_id="5179", disclosure="absent", extra_field=1)
    schema = json.loads(json.dumps(DisclosureAnswer.model_json_schema()))
    assert set(schema["properties"]) == {"indikit_id", "disclosure", "matches"}
    assert set(schema["$defs"]["MatchItem"]["properties"]) == {"document_indicator_text", "match_type", "level", "baseline", "target", "unit", "evidence_label", "evidence_quote", "reason"}
