"""docx parsing in document order (paper Section 3.3.1). Phase 2.

Target: walk the document body so tables stay next to their headings. Replaces
``extraction_utils.simple_parse_docx`` (which appends all tables after all paragraphs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_docx(path: Path) -> list[Any]:
    """Return ordered blocks (paragraphs with style, tables as rows/columns) for one docx."""
    raise NotImplementedError("Phase 2")
