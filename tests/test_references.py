from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.preprocess.models import Block, Reference
from pipeline.preprocess.references import detect_references, detect_references_in_text, elsewhere_flags, load_references, write_references


def _blocks(*texts: str) -> list[Block]:
    return [Block(block_id=f"d:b{i}", doc_id="d", kind="paragraph", text=t, order=i, page=i + 1) for i, t in enumerate(texts)]


@pytest.mark.parametrize(
    "text, kind, name_part",
    [
        ("The detailed indicators are provided in Annex C - Logical Framework.", "logframe", "Annex C"),
        ("Please refer to Annex 3 (logframe) for baselines and targets.", "logframe", "Annex 3"),
        ("Annex O: Indicator Tracking Table", "tracker", "Annex O: Indicator Tracking Table"),
        ("Progress will be reported in the IPTT every quarter.", "tracker", "IPTT"),
        ("Table 9 Fragment of the indicator database", "indicator_table", "indicator database"),
        ("Le cadre logique est présenté en annexe B.", "logframe", "annexe B"),
        ("Les indicateurs sont joints en annexe.", "indicator_table", "indicateurs"),
        ("The logframe is attached as a separate file.", "logframe", "logframe"),
    ],
)
def test_reference_positives(text, kind, name_part):
    refs = detect_references_in_text(text, "p.1")
    assert refs, text
    assert refs[0].kind == kind and name_part.lower() in refs[0].referenced_document.lower()
    assert 0 < refs[0].confidence <= 1


@pytest.mark.parametrize(
    "text",
    [
        "See Annex 3 for the budget breakdown.",
        "The CVs of key staff are attached in Annex D.",
        "Annex N-2: Design of latrines",
        "This proposal has three outcomes and eight outputs.",
        "Indicators will be measured through household surveys.",
    ],
)
def test_reference_negatives(text):
    assert detect_references_in_text(text, "p.1") == []


def test_detect_references_dedupes_and_flags(tmp_path: Path):
    blocks = _blocks("See Annex C - Logical Framework for the indicators.", "Again, see Annex C - Logical Framework.", "Budget in Annex 3.")
    refs = detect_references(blocks)
    assert len(refs) == 1 and refs[0].location == "p.1"
    assert elsewhere_flags(refs, 0, 0) == (True, False)
    assert elsewhere_flags(refs, 1, 0) == (False, True)
    assert elsewhere_flags([], 1, 0) == (False, False)
    assert elsewhere_flags([Reference(referenced_document="x", quote="q", location="p.1", kind="other", confidence=0.5)], 0, 0) == (False, False)
    folder = tmp_path / "doc"
    write_references(refs, folder / "references.json")
    assert load_references("doc", tmp_path)[0].referenced_document == refs[0].referenced_document
    assert json.loads((folder / "references.json").read_text())[0]["kind"] == "logframe"
    with pytest.raises(FileNotFoundError):
        load_references("missing", tmp_path)
