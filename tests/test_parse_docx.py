from __future__ import annotations

from pathlib import Path

from pipeline.preprocess.parse_docx import fallback_heading_level, heading_level_from_style, parse_docx
from tests.fixtures_docs import make_docx


def test_docx_reading_order_headings_and_tables(tmp_path: Path):
    res = parse_docx(make_docx(tmp_path / "p.docx"), "doc1")
    assert res.status == "ok" and res.warnings == []
    kinds = [b.kind for b in res.blocks]
    assert kinds == ["heading", "paragraph", "heading", "paragraph", "table", "paragraph", "heading", "paragraph"]
    headings = [(b.text, b.style_info["heading_level"]) for b in res.blocks if b.kind == "heading"]
    assert headings == [("Project proposal", 1), ("3.2 Logical framework", 2), ("4 Monitoring", 2)]
    # the table sits under its heading, not appended at the end
    tbl_block = res.blocks[4]
    assert tbl_block.heading_path == ["Project proposal", "3.2 Logical framework"]
    assert res.blocks[5].text.startswith("Assumptions") and res.blocks[5].heading_path[-1] == "3.2 Logical framework"
    assert len(res.tables) == 1
    t = res.tables[0]
    assert t.table_id == "doc1:t000" and t.source == "docx" and t.heading_path[-1] == "3.2 Logical framework"
    assert t.header == ["Intervention logic", "Indicator", "Baseline", "Target"]
    assert t.column_roles == ["result", "indicator", "baseline", "target"] and t.is_logframe
    # vertically merged cell appears once, the continuation cell is blank; nested table flattened into the cell
    assert t.rows[0][0] == "Outcome 1: Improved WASH" and t.rows[1][0] == ""
    assert t.rows[2][3].startswith("40") and "Q1 | 20" in t.rows[2][3]
    assert t.n_rows == 3 and t.n_cols == 4
    ids = [b.block_id for b in res.blocks]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_docx_is_deterministic(tmp_path: Path):
    path = make_docx(tmp_path / "p.docx")
    a = parse_docx(path, "d").model_dump()
    b = parse_docx(path, "d").model_dump()
    assert a == b


def test_heading_detection_rules():
    assert heading_level_from_style("Heading 2") == 2
    assert heading_level_from_style("Titre 3") == 3
    assert heading_level_from_style("Title") == 1
    assert heading_level_from_style("Heading 4a") == 4
    assert heading_level_from_style("Normal") is None
    assert fallback_heading_level("3.2 Logical framework", bold=False, style_name="Normal") == 2
    assert fallback_heading_level("Monitoring and evaluation", bold=True, style_name="Normal") == 2
    assert fallback_heading_level("This is a long bold sentence that is clearly not a heading because it runs on.", bold=True, style_name="Normal") is None
    assert fallback_heading_level("Monitoring plan.", bold=True, style_name="Normal") is None
    assert fallback_heading_level("3.2 Logical framework", bold=True, style_name="List Paragraph") is None
