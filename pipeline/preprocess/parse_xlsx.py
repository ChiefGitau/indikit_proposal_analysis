"""xlsx parsing (paper Section 3.3.1 table reconstruction). One ``Table`` per sheet region.

openpyxl with ``data_only=True`` (cached formula results); merged cells are forward-filled with their value;
regions are separated by two or more fully empty rows; the header row is the first row that maps to logframe
roles, else the first dense row; title rows above the header become the caption.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import openpyxl

from pipeline import config
from pipeline.preprocess.models import Block, ParseResult, Table
from pipeline.preprocess.tables import make_table, table_block_text
from pipeline.preprocess.textnorm import clean_block_text

EMPTY_ROWS_BETWEEN_REGIONS = 2


def _cell_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else repr(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()[:10]
    return clean_block_text(str(value))


def sheet_grid(ws) -> list[list[str]]:
    """Values of a sheet as strings with merged ranges forward-filled from their top-left cell."""
    max_row, max_col = ws.max_row or 0, ws.max_column or 0
    if max_row == 0 or max_col == 0:
        return []
    grid = [[_cell_str(c) for c in row] for row in ws.iter_rows(min_row=1, max_row=max_row, max_col=max_col, values_only=True)]
    for rng in ws.merged_cells.ranges:
        r0, c0, r1, c1 = rng.min_row - 1, rng.min_col - 1, rng.max_row - 1, rng.max_col - 1
        if r0 >= len(grid) or c0 >= len(grid[r0]):
            continue
        value = grid[r0][c0]
        if not value:
            continue
        for r in range(r0, min(r1, len(grid) - 1) + 1):
            for c in range(c0, min(c1, len(grid[r]) - 1) + 1):
                if not grid[r][c]:
                    grid[r][c] = value
    return grid


def split_regions(grid: list[list[str]], gap: int = EMPTY_ROWS_BETWEEN_REGIONS) -> list[list[list[str]]]:
    """Split a sheet into regions separated by ``gap`` or more fully empty rows."""
    regions: list[list[list[str]]] = []
    current: list[list[str]] = []
    empty_run = 0
    for row in grid:
        if any(c.strip() for c in row):
            if empty_run >= gap and current:
                regions.append(current)
                current = []
            current.append(row)
            empty_run = 0
        else:
            empty_run += 1
            if current:
                current.append(row)
    if current:
        regions.append(current)
    return [r for r in regions if any(any(c.strip() for c in row) for row in r)]


def parse_xlsx(path: Path, doc_id: str) -> ParseResult:
    path = Path(path)
    warnings: list[str] = []
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=False)
    blocks: list[Block] = []
    tables: list[Table] = []
    order = 0
    for ws in wb.worksheets:
        title = clean_block_text(ws.title) or "Sheet"
        grid = sheet_grid(ws)
        regions = split_regions(grid)
        blocks.append(Block(block_id=f"{doc_id}:b{order:05d}", doc_id=doc_id, kind="heading", text=title, order=order, heading_path=[], style_info={"heading_level": 1, "sheet": title}))
        order += 1
        if not regions:
            warnings.append(f"sheet '{title}' is empty")
            continue
        pending_caption = ""
        for region in regions:
            non_empty_rows = [r for r in region if any(c.strip() for c in r)]
            if len(non_empty_rows) == 1 and region is not regions[-1]:
                # a lone title row above a gap: caption for the next region, not a table
                pending_caption = " ".join(c for c in non_empty_rows[0] if c.strip())[:300]
                continue
            table = make_table(doc_id, len(tables), region, source="xlsx", order=order, heading_path=[title], sheet_name=title, caption=pending_caption, warnings=warnings)
            pending_caption = ""
            if table is None:
                continue
            tables.append(table)
            blocks.append(Block(block_id=f"{doc_id}:b{order:05d}", doc_id=doc_id, kind="table", text=table_block_text(table), order=order, heading_path=[title], style_info={"table_id": table.table_id, "sheet": title}))
            order += 1
    n_cells = sum(len(c) for t in tables for r in t.rows for c in r)
    if not tables:
        warnings.append("no non-empty sheet regions found")
    return ParseResult(blocks=blocks, tables=tables, warnings=warnings, n_pages_or_sheets=len(wb.worksheets), has_text_layer=True, status="ok" if n_cells else "partial")
