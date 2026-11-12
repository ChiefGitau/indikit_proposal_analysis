"""PDF layout analysis and table reconstruction (paper Section 3.3.1). Phase 2.

Target: PyMuPDF text blocks with font size/weight and ``find_tables``; OCR fallback for
pages without a text layer. Replaces ``extraction_utils.simple_parse_pdf``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_pdf(path: Path) -> list[Any]:
    """Return ordered blocks (text with font info, tables as rows/columns) for one PDF."""
    raise NotImplementedError("Phase 2")
