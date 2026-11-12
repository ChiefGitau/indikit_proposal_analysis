"""Reference detection (docs/plan.md Section 4.2): phrases pointing to another document
("see Annex C", "Indicator tracker", "IPTT", "cadre logique en annexe"). Phase 2.
"""
from __future__ import annotations

from typing import Any


def detect_references(blocks: list[Any]) -> list[str]:
    """Return the names of referenced documents found in the text."""
    raise NotImplementedError("Phase 2")
