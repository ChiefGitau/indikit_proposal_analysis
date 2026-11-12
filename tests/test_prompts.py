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
    prompt = prompts.build_disclosure_prompt(ind, ["[p.12 logframe] Number of people with access to safe shelter | baseline 0 | target 1,200"], prompts.MATCH_RUBRIC)
    for header in prompts.PART_HEADERS:
        assert header in prompt
    assert prompt.index(prompts.PART_HEADERS[0]) < prompt.index(prompts.PART_HEADERS[1]) < prompt.index(prompts.PART_HEADERS[2]) < prompt.index(prompts.PART_HEADERS[3]) < prompt.index(prompts.PART_HEADERS[4])
    assert "5179" in prompt and ind.kpi_wording_en in prompt and ind.knowledge in prompt
    assert "nombre de personnes" in prompt and "by sex and age" in prompt and "adequate shelter" in prompt
    assert "[1] [p.12 logframe]" in prompt
    assert "verbatim" in prompt and "close" in prompt  # rubric
    assert '"disclosure": "present | absent"' in prompt
    assert config.PROMPT_VERSION in prompt
    assert prompts.PROMPT_VERSION == config.PROMPT_VERSION


def test_prompt_without_reference_content_says_so():
    ind = IndicatorMetadata(indikit_id="1", name="X")
    prompt = prompts.build_disclosure_prompt(ind, [], "rubric here")
    assert "(no reference content retrieved)" in prompt and "rubric here" in prompt and "(no knowledge note available)" in prompt


def test_disclosure_answer_schema():
    ok = DisclosureAnswer(indikit_id="5179", disclosure="Present", match_type="Verbatim", target=1200, level="Outcome", reason="same")
    assert ok.match_type == "verbatim" and ok.target == "1200" and ok.level == "outcome"
    absent = DisclosureAnswer(indikit_id="5179", disclosure="absent", match_type="null", reason="not found")
    assert absent.match_type is None
    with pytest.raises(ValidationError):
        DisclosureAnswer(indikit_id="5179", disclosure="present", reason="missing match type")
    with pytest.raises(ValidationError):
        DisclosureAnswer(indikit_id="5179", disclosure="absent", match_type="close", reason="x")
    with pytest.raises(ValidationError):
        DisclosureAnswer(indikit_id="5179", disclosure="maybe", reason="x")
    schema = json.loads(json.dumps(DisclosureAnswer.model_json_schema()))
    assert set(schema["properties"]) == {"indikit_id", "disclosure", "match_type", "document_indicator_text", "level", "baseline", "target", "unit", "location", "evidence", "reason"}
