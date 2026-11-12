from __future__ import annotations

from pipeline.preprocess import logframe as lf
from pipeline.preprocess.models import Block, Table


def test_header_role_mapping_en_fr():
    assert lf.map_header_roles(["Intervention logic", "Objectively verifiable indicators", "Baseline (value & year)", "Target", "Sources of verification", "Assumptions"]) == ["result", "indicator", "baseline", "target", "mov", "assumption"]
    assert lf.map_header_roles(["Logique d'intervention", "Indicateurs", "Valeur de référence", "Cible (fin du projet)", "Sources et moyens de vérification", "Hypothèses"]) == ["result", "indicator", "baseline", "target", "mov", "assumption"]
    assert lf.map_header_roles(["Result Level", "Baseline", "Target", "Means of Verification", "Risks"]) == ["result", "baseline", "target", "mov", "assumption"]
    assert lf.map_header_roles(["Item", "Unit", "Qty"]) == [None, None, None]
    assert lf.is_logframe_header(["result", "indicator", "baseline"]) and not lf.is_logframe_header(["indicator", None]) and not lf.is_logframe_header(["baseline", "target"])
    assert lf.header_row_index([["Title row", "", ""], ["", "", ""], ["Indicator", "Baseline", "Target"], ["x", "1", "2"]]) == 2
    assert lf.header_row_index([["a", "b", "c"], ["1", "2", "3"]]) is None


def test_classify_table_by_header_and_by_title():
    t = Table(table_id="d:t000", doc_id="d", source="docx", header=["Indicator", "Baseline", "Target"], rows=[["x", "1", "2"]], n_rows=1, n_cols=3)
    assert lf.classify_table(t).is_logframe and t.column_roles == ["indicator", "baseline", "target"]
    t2 = Table(table_id="d:t001", doc_id="d", source="xlsx", sheet_name="Logframe", header=["Result Level", "Value"], rows=[["Outcome 1", "x"]], n_rows=1, n_cols=2)
    assert lf.classify_table(t2).is_logframe  # title rule (sheet name) + a recognised role
    t3 = Table(table_id="d:t002", doc_id="d", source="pdf", caption="Table 4 Budget", header=["Item", "Cost"], rows=[["a", "1"]], n_rows=1, n_cols=2)
    assert not lf.classify_table(t3).is_logframe
    t4 = Table(table_id="d:t003", doc_id="d", source="pdf", caption="Selected indicators per sector:", header=["", "WATER, SANITATION", ""], rows=[["W8", "", "Percent of households"]], n_rows=1, n_cols=3)
    assert lf.classify_table(t4).is_logframe


def test_carry_down_result_context_and_rows():
    t = Table(table_id="d:t000", doc_id="d", source="docx", header=["Intervention logic", "Indicator", "Baseline", "Target"], n_rows=4, n_cols=4,
              rows=[["Outcome 1: Improved WASH", "% of households with soap", "12%", "60%"], ["", "Number of kits", "0", "2500"], ["Output 1.1: Latrines", "# of latrines", "0", "40"], ["", "# of users", "", "100"]])
    lf.classify_table(t)
    assert lf.carry_result_context(t) == ["Outcome 1: Improved WASH", "Outcome 1: Improved WASH", "Output 1.1: Latrines", "Output 1.1: Latrines"]
    rows = lf.build_rows(t)
    assert rows[1].row_id == "d:t000:r0001" and rows[1].result_context == "Outcome 1: Improved WASH"
    assert rows[1].text == "Indicator: Number of kits | Baseline: 0 | Target: 2500"
    assert rows[3].cells == {"Intervention logic": "", "Indicator": "# of users", "Baseline": "", "Target": "100"}
    # without a result column: first-column result rows set the context (HOPE-style logframes)
    t2 = Table(table_id="d:t001", doc_id="d", source="xlsx", header=["Result Level", "Baseline", "Target"], n_rows=3, n_cols=3,
               rows=[["Outcome 1: Vulnerable households are resilient", "", ""], ["Indicator 1.1: % of households with savings", "TBA", "80%"], ["Indicator 1.2: # of VSLAs", "0", "25"]])
    lf.classify_table(t2)
    assert lf.carry_result_context(t2) == ["Outcome 1: Vulnerable households are resilient"] * 3
    assert lf.dedupe_headers(["Sector", "Sector", "", "Target"]) == ["Sector", "Sector (2)", "col3", "Target"]


def test_flag_logframe_sections():
    blocks = [
        Block(block_id="d:b0", doc_id="d", kind="heading", text="1 Context", order=0, style_info={"heading_level": 1}),
        Block(block_id="d:b1", doc_id="d", kind="paragraph", text="Background.", order=1),
        Block(block_id="d:b2", doc_id="d", kind="heading", text="2 Logical framework", order=2, style_info={"heading_level": 1}),
        Block(block_id="d:b3", doc_id="d", kind="list_item", text="W8 Percent of households with soap", order=3),
        Block(block_id="d:b4", doc_id="d", kind="heading", text="2.1 Indicators per sector", order=4, style_info={"heading_level": 2}),
        Block(block_id="d:b5", doc_id="d", kind="paragraph", text="More indicators.", order=5),
        Block(block_id="d:b6", doc_id="d", kind="heading", text="3 Budget", order=6, style_info={"heading_level": 1}),
        Block(block_id="d:b7", doc_id="d", kind="paragraph", text="Costs.", order=7),
    ]
    ids = lf.flag_logframe_sections(blocks, [])
    assert ids == ["d:b2"]
    assert [b.is_logframe for b in blocks] == [False, False, True, True, True, True, False, False]
