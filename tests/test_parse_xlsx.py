from __future__ import annotations

from pathlib import Path

from pipeline.preprocess.parse_xlsx import parse_xlsx, split_regions
from tests.fixtures_docs import make_xlsx


def test_xlsx_regions_header_merged_cells_and_caption(tmp_path: Path):
    res = parse_xlsx(make_xlsx(tmp_path / "lf.xlsx"), "x1")
    assert res.status == "ok" and res.n_pages_or_sheets == 2
    assert [b.kind for b in res.blocks] == ["heading", "table", "heading", "table", "table"]
    lf = res.tables[0]
    assert lf.sheet_name == "Logframe" and lf.source == "xlsx" and lf.heading_path == ["Logframe"]
    assert lf.caption == "Cadre logique de l'action"   # title row above the header
    assert lf.header == ["Logique d'intervention", "Indicateurs objectivement vérifiables", "Valeur de référence", "Cible", "Sources et moyens de vérification"]
    assert lf.column_roles == ["result", "indicator", "baseline", "target", "mov"] and lf.is_logframe
    # merged A4:A5 forward-filled, numbers rendered without ".0"
    assert lf.rows[0][0] == "Objectif spécifique 1" and lf.rows[1][0] == "Objectif spécifique 1"
    assert lf.rows[1][2] == "0" and lf.rows[1][3] == "15"
    assert lf.rows[2][0] == "Résultat 1.1"
    budget, notes = res.tables[1], res.tables[2]
    assert budget.header == ["Item", "Unit", "Qty", "Cost"] and budget.rows == [["Pipes", "m", "100", "2500"]] and not budget.is_logframe
    assert notes.header == ["Notes"] and notes.rows == [["Prices from 2024 quotation"]]


def test_split_regions_on_empty_rows():
    grid = [["a", ""], ["b", ""], ["", ""], ["", ""], ["c", "d"], ["", ""], ["e", ""]]
    regions = split_regions(grid)
    assert [[r[0] for r in reg if r[0]] for reg in regions] == [["a", "b"], ["c", "e"]]
