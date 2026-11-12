"""xlsx parsing (paper Section 3.3.1 table reconstruction). Phase 2.

Target: read every sheet row by row; logframes are often delivered this way
(7 xlsx files in ``proposals/`` are currently skipped).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_xlsx(path: Path) -> list[Any]:
    """Return one table block per sheet."""
    raise NotImplementedError("Phase 2")
