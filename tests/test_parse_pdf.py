from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.preprocess.parse_pdf import _Line, detect_headers_footers, merge_continuations, parse_pdf
from tests.fixtures_docs import make_pdf


def test_pdf_headings_tables_continuation_and_footer(tmp_path: Path):
    res = parse_pdf(make_pdf(tmp_path / "p.pdf"), "pdf1")
    assert res.status == "ok" and res.has_text_layer and res.n_pages_or_sheets == 3
    headings = [(b.page, b.text) for b in res.blocks if b.kind == "heading"]
    assert ("2. Results framework" in [h for _, h in headings]) and ("3. Monitoring" in [h for _, h in headings]) and ("Annex A: Budget" in [h for _, h in headings])
    texts = " ".join(b.text for b in res.blocks)
    assert "page 1" not in texts and "page 2" not in texts  # repeating footer removed
    assert "handwashing station on premises" in texts
    assert len(res.tables) == 1, [t.model_dump() for t in res.tables]
    t = res.tables[0]
    assert t.page_start == 1 and t.page_end == 2 and t.merged_from == ["p2"]
    assert t.header == ["Result", "Indicator", "Baseline", "Target"] and t.is_logframe
    assert [r[1] for r in t.rows] == ["% of households with handwashing facilities with soap and water", "Number of boreholes rehabilitated", "% of water samples with free residual chlorine above 0.2 mg/l"]
    assert any(w.startswith("continuation table merged") for w in res.warnings)
    # the table block is placed in reading order under the heading of page 1
    tb = next(b for b in res.blocks if b.kind == "table")
    assert tb.page == 1 and tb.heading_path == ["2. Results framework"]
    # a paragraph inside the table region is not duplicated as text
    assert texts.count("Number of boreholes rehabilitated") == 0


def test_pdf_without_continuation(tmp_path: Path):
    res = parse_pdf(make_pdf(tmp_path / "p.pdf", two_page_table=False), "pdf2")
    assert len(res.tables) == 1 and res.tables[0].merged_from == [] and res.tables[0].n_rows == 2


def test_pdf_is_deterministic(tmp_path: Path):
    path = make_pdf(tmp_path / "p.pdf")
    assert parse_pdf(path, "d").model_dump() == parse_pdf(path, "d").model_dump()


def test_header_footer_detection_and_merge_rules():
    pages = [[_Line(page=i, y0=20, y1=30, x0=0, x1=100, text=f"ZOA proposal - page {i + 1}", size=8, bold=False), _Line(page=i, y0=400, y1=410, x0=0, x1=100, text=f"Body text {i}", size=10, bold=False)] for i in range(5)]
    hf = detect_headers_footers(pages, page_height=842)
    assert hf == {"zoa proposal - page #"}
    from pipeline.preprocess.parse_pdf import _RawTable

    header = ["Indicator", "Baseline", "Target"]
    a = _RawTable(page=0, bbox=(50, 700, 500, 800), grid=[header, ["a", "1", "2"]], engine="pymupdf")
    b = _RawTable(page=1, bbox=(50, 50, 500, 120), grid=[header, ["b", "3", "4"]], engine="pymupdf")
    c = _RawTable(page=1, bbox=(50, 500, 500, 600), grid=[["x", "y", "z"], ["1", "2", "3"]], engine="pymupdf")
    warnings: list[str] = []
    out = merge_continuations([a, b, c], 842, warnings)
    assert len(out) == 2 and out[0].grid == [header, ["a", "1", "2"], ["b", "3", "4"]] and out[0].ids == ["p2"]
    assert len(warnings) == 1
    # different column count: not merged
    d = _RawTable(page=2, bbox=(50, 40, 500, 100), grid=[["p", "q"], ["1", "2"]], engine="pymupdf")
    assert len(merge_continuations([c, d], 842, [])) == 2
