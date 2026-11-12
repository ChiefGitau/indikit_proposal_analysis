"""PDF layout analysis and table reconstruction (paper Section 3.3.1). Replaces ``simple_parse_pdf``.

PyMuPDF gives text spans with font size / weight / position and ``page.find_tables()``; pdfplumber is tried on
pages that look tabular but where PyMuPDF finds no table. Repeating headers / footers and page numbers are
removed, headings are inferred from font size relative to the body text and from numbering, table regions are
excluded from paragraph text, and continuation tables on consecutive pages are merged. Pages without a text
layer are flagged (OCR is optional, behind ``ocr=True`` and only if ``pytesseract`` is installed).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf

from pipeline import config
from pipeline.preprocess.models import Block, ParseResult, Table
from pipeline.preprocess.tables import clean_grid, drop_empty, make_table, table_block_text
from pipeline.preprocess.textnorm import cap_length, clean_block_text, heading_number_prefix, page_number_line

log = logging.getLogger(__name__)

_BULLET_RE = re.compile(r"^\s*([•·▪■●○◦\-–—*]|\d{1,2}[.)]|[a-z][.)]|[ivx]{1,4}[.)])\s+", re.IGNORECASE)
_TABULAR_LINE_RE = re.compile(r"\S\s{3,}\S")


@dataclass
class _Line:
    page: int
    y0: float
    y1: float
    x0: float
    x1: float
    text: str
    size: float
    bold: bool


@dataclass
class _RawTable:
    page: int
    bbox: tuple[float, float, float, float]
    grid: list[list[str]]
    engine: str
    ids: list[str] = field(default_factory=list)


def _inside(line_bbox: tuple[float, float, float, float], boxes: list[tuple[float, float, float, float]], tol: float = 2.0) -> bool:
    x0, y0, x1, y1 = line_bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return any(bx0 - tol <= cx <= bx1 + tol and by0 - tol <= cy <= by1 + tol for bx0, by0, bx1, by1 in boxes)


def _page_lines(page: pymupdf.Page, pno: int, table_boxes: list[tuple[float, float, float, float]]) -> list[_Line]:
    lines: list[_Line] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"])
            bbox = tuple(line["bbox"])
            if _inside(bbox, table_boxes):
                continue
            size = max(s["size"] for s in spans)
            bold = any(("bold" in s.get("font", "").lower()) or (s.get("flags", 0) & 16) for s in spans) and sum(len(s["text"]) for s in spans if ("bold" in s.get("font", "").lower()) or (s.get("flags", 0) & 16)) >= 0.6 * len(text.strip())
            lines.append(_Line(page=pno, y0=bbox[1], y1=bbox[3], x0=bbox[0], x1=bbox[2], text=text, size=round(size, 1), bold=bool(bold)))
    lines.sort(key=lambda l: (round(l.y0 / 3), l.x0))
    return lines


def _norm_hf(text: str) -> str:
    return re.sub(r"\d+", "#", clean_block_text(text).lower())[:80]


def detect_headers_footers(pages: list[list[_Line]], page_height: float) -> set[str]:
    """Normalised line texts that appear in the top or bottom 12% of at least max(3, 40%) of the pages."""
    n_pages = len(pages)
    if n_pages < 3:
        return set()
    counts: Counter = Counter()
    for lines in pages:
        seen = set()
        for l in lines:
            if l.y0 < page_height * 0.12 or l.y1 > page_height * 0.88:
                key = _norm_hf(l.text)
                if key and len(key) > 3 and key not in seen:
                    seen.add(key)
                    counts[key] += 1
    threshold = max(3, int(0.4 * n_pages))
    return {k for k, c in counts.items() if c >= threshold}


def _body_size(pages: list[list[_Line]]) -> float:
    weights: Counter = Counter()
    for lines in pages:
        for l in lines:
            weights[l.size] += len(l.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _heading_level(line: _Line, body: float, size_tiers: list[float]) -> Optional[int]:
    text = line.text.strip()
    if not text or len(text) > 160 or text.endswith((".", ";", ",", ":")) and not heading_number_prefix(text):
        return None
    words = len(text.split())
    if line.size >= body * 1.15 and words <= 20:
        return min(4, size_tiers.index(line.size) + 1) if line.size in size_tiers else 2
    num = heading_number_prefix(text)
    if num and re.match(r"^(\d{1,2}\.)*\d{1,2}\.?$", num) and words <= 16 and (line.bold or line.size > body):
        return min(4, num.rstrip(".").count(".") + 1 + (len(size_tiers) if size_tiers else 0)) if False else min(4, num.rstrip(".").count(".") + 1)
    if num and re.match(r"^(annex|annexe|appendix|chapter|section|part)", num, re.IGNORECASE) and words <= 14:
        return 1
    if line.bold and words <= 10 and line.size >= body and text[0].isupper() and not _BULLET_RE.match(text):
        return min(4, len(size_tiers) + 1) if size_tiers else 2
    return None


def _looks_tabular(lines: list[_Line]) -> bool:
    if len(lines) < 8:
        return False
    tab_like = sum(1 for l in lines if _TABULAR_LINE_RE.search(l.text))
    return tab_like >= max(6, 0.3 * len(lines))


def _find_tables_pymupdf(page: pymupdf.Page) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    out = []
    try:
        finder = page.find_tables()
    except Exception as exc:  # pragma: no cover - library edge cases
        log.debug("find_tables failed on page %s: %s", page.number, exc)
        return out
    for t in finder.tables:
        try:
            grid = [[(c if c is not None else "") for c in row] for row in t.extract()]
        except Exception:
            continue
        if grid and len(grid[0]) >= 2:
            out.append((tuple(t.bbox), grid))
    return out


def _find_tables_pdfplumber(path: Path, pno: int) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    try:
        import pdfplumber
    except Exception:
        return []
    out = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            page = pdf.pages[pno]
            for t in page.find_tables():
                grid = [[(c if c is not None else "") for c in row] for row in t.extract()]
                if grid and len(grid[0]) >= 2 and len(grid) >= 2:
                    out.append((tuple(t.bbox), grid))
    except Exception as exc:
        log.debug("pdfplumber failed on page %s: %s", pno, exc)
    return out


def _header_like(row: list[str], header: list[str]) -> bool:
    a = [clean_block_text(c).lower() for c in row]
    b = [clean_block_text(c).lower() for c in header]
    return a == b or (sum(1 for x, y in zip(a, b) if x and x == y) >= max(1, len(b) // 2) and len(a) == len(b))


def merge_continuations(raw: list[_RawTable], page_height: float, warnings: list[str]) -> list[_RawTable]:
    """Merge a table at the top of page p+1 into the table at the bottom of page p when the column count matches
    and either the header repeats or the first row looks like data."""
    merged: list[_RawTable] = []
    for t in raw:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and t.page == prev.page + 1
            and prev.grid and t.grid and len(prev.grid[0]) == len(t.grid[0])
            and prev.bbox[3] >= page_height * 0.70
            and t.bbox[1] <= page_height * 0.30
        ):
            rows = t.grid[1:] if _header_like(t.grid[0], prev.grid[0]) else t.grid
            prev.grid.extend(rows)
            prev.ids.append(f"p{t.page + 1}")
            warnings.append(f"continuation table merged: page {prev.page + 1} + page {t.page + 1} ({len(rows)} rows)")
            continue
        merged.append(t)
    return merged


def _ocr_page(page: pymupdf.Page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img)


def parse_pdf(path: Path, doc_id: str, *, ocr: bool = False) -> ParseResult:
    path = Path(path)
    warnings: list[str] = []
    doc = pymupdf.open(str(path))
    n_pages = len(doc)
    page_height = float(doc[0].rect.height) if n_pages else 842.0

    raw_tables: list[_RawTable] = []
    pages_lines: list[list[_Line]] = []
    no_text_pages: list[int] = []
    ocr_pages = 0
    for pno in range(n_pages):
        page = doc[pno]
        found = _find_tables_pymupdf(page)
        engine = "pymupdf"
        boxes = [b for b, _ in found]
        lines = _page_lines(page, pno, boxes)
        if not found and _looks_tabular(lines):
            found = _find_tables_pdfplumber(path, pno)
            if found:
                engine = "pdfplumber"
                boxes = [b for b, _ in found]
                lines = _page_lines(page, pno, boxes)
        for bbox, grid in found:
            raw_tables.append(_RawTable(page=pno, bbox=bbox, grid=grid, engine=engine))
        page_text_len = sum(len(l.text.strip()) for l in lines) + sum(len(c) for _, g in found for r in g for c in r if c)
        if page_text_len < 20 and page.get_images():
            no_text_pages.append(pno + 1)
            if ocr:
                text = _ocr_page(page)
                if text.strip():
                    ocr_pages += 1
                    y = 0.0
                    for ln in text.splitlines():
                        if ln.strip():
                            lines.append(_Line(page=pno, y0=y, y1=y + 10, x0=0, x1=100, text=ln, size=10.0, bold=False))
                            y += 12
        pages_lines.append(lines)

    hf = detect_headers_footers(pages_lines, page_height)
    body = _body_size(pages_lines)
    size_counter: Counter = Counter()
    for lines in pages_lines:
        for l in lines:
            if l.size >= body * 1.15 and len(l.text.strip()) <= 160:
                size_counter[l.size] += 1
    size_tiers = sorted((s for s, c in size_counter.items() if c >= 1), reverse=True)[:4]

    raw_tables.sort(key=lambda t: (t.page, t.bbox[1]))
    raw_tables = merge_continuations(raw_tables, page_height, warnings)
    tables_by_page: dict[int, list[_RawTable]] = {}
    for t in raw_tables:
        tables_by_page.setdefault(t.page, []).append(t)

    blocks: list[Block] = []
    tables: list[Table] = []
    heading_stack: list[tuple[int, str]] = []
    order = 0

    def hp() -> list[str]:
        return [t for _, t in heading_stack]

    def flush(par_lines: list[_Line]) -> None:
        nonlocal order
        if not par_lines:
            return
        text = clean_block_text("\n".join(l.text for l in par_lines))
        if not text:
            return
        text = cap_length(text, config.MAX_BLOCK_CHARS, "block", warnings, f"{doc_id}:b{order:05d}")
        kind = "list_item" if _BULLET_RE.match(par_lines[0].text) else "paragraph"
        blocks.append(Block(block_id=f"{doc_id}:b{order:05d}", doc_id=doc_id, kind=kind, text=text, page=par_lines[0].page + 1, order=order, heading_path=hp(), style_info={"font_size": par_lines[0].size, "bold": par_lines[0].bold}))
        order += 1

    for pno, lines in enumerate(pages_lines):
        page_tables = sorted(tables_by_page.get(pno, []), key=lambda t: t.bbox[1])
        items: list[tuple[float, str, object]] = [(l.y0, "line", l) for l in lines if _norm_hf(l.text) not in hf and not page_number_line(l.text)]
        items += [(t.bbox[1], "table", t) for t in page_tables]
        items.sort(key=lambda x: (x[0], 0 if x[1] == "table" else 1))
        par: list[_Line] = []
        last: Optional[_Line] = None
        for _, kind, obj in items:
            if kind == "table":
                flush(par)
                par, last = [], None
                t: _RawTable = obj  # type: ignore[assignment]
                caption = ""
                if blocks and blocks[-1].kind in ("paragraph", "list_item") and blocks[-1].page == pno + 1 and len(blocks[-1].text) <= 140:
                    prev = blocks[-1].text
                    if prev.endswith(":") or re.match(r"^(table|tableau)\s*\d+", prev, re.IGNORECASE) or re.search(r"indicat", prev, re.IGNORECASE):
                        caption = prev
                table = make_table(doc_id, len(tables), t.grid, source="pdf", order=order, heading_path=hp(), caption=caption, page_start=t.page + 1, page_end=t.page + 1 + len(t.ids), warnings=warnings)
                if table is None:
                    continue
                table.merged_from = list(t.ids)
                tables.append(table)
                blocks.append(Block(block_id=f"{doc_id}:b{order:05d}", doc_id=doc_id, kind="table", text=table_block_text(table), page=t.page + 1, order=order, heading_path=hp(), style_info={"table_id": table.table_id, "engine": t.engine}))
                order += 1
                continue
            line: _Line = obj  # type: ignore[assignment]
            level = _heading_level(line, body, size_tiers)
            if level is not None:
                flush(par)
                par, last = [], None
                text = clean_block_text(line.text)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                blocks.append(Block(block_id=f"{doc_id}:b{order:05d}", doc_id=doc_id, kind="heading", text=text, page=pno + 1, order=order, heading_path=hp(), style_info={"heading_level": level, "font_size": line.size, "bold": line.bold}))
                heading_stack.append((level, text))
                order += 1
                continue
            gap = (line.y0 - last.y1) if last is not None else 0.0
            new_par = last is not None and (gap > max(4.0, 0.9 * line.size) or abs(line.size - last.size) > 1.5 or _BULLET_RE.match(line.text))
            if new_par:
                flush(par)
                par = []
            par.append(line)
            last = line
        flush(par)

    status = "ok"
    if no_text_pages:
        warnings.append(f"{len(no_text_pages)} page(s) without a text layer: {no_text_pages[:12]}{' ...' if len(no_text_pages) > 12 else ''}" + (f"; OCR recovered {ocr_pages}" if ocr else "; OCR not run"))
        if ocr_pages < len(no_text_pages):
            status = "partial"
    n_chars = sum(len(b.text) for b in blocks if b.kind != "table") + sum(len(c) for t in tables for r in t.rows for c in r)
    if n_chars == 0:
        warnings.append("no text extracted from PDF")
        status = "partial"
    return ParseResult(blocks=blocks, tables=tables, warnings=warnings, n_pages_or_sheets=n_pages, has_text_layer=not no_text_pages, status=status)
