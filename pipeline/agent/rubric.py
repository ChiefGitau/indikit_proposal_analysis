"""Match rubric for the prompts (docs/phase_3.md Task 2). The text lives in ``docs/rubric.md`` between the
``<!-- prompt-start -->`` / ``<!-- prompt-end -->`` markers; ``RUBRIC_VERSION`` is part of every cache key."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pipeline import config

RUBRIC_VERSION = config.RUBRIC_VERSION
_MARKERS = re.compile(r"<!--\s*prompt-start\s*-->(.*?)<!--\s*prompt-end\s*-->", re.DOTALL)

MATCH_TYPES = ("verbatim", "close")
NON_MATCH = ("none", "not_an_indicator")

# Minimal fallback if docs/rubric.md is missing (keeps the prompt usable; tests assert the file exists)
_FALLBACK = """Match types:
- verbatim: same measurement and (nearly) identical wording; differences limited to targets, placeholders filled in, or word order.
- close: same thing measured, with different wording, unit (number vs percentage), population or scope.
- absent (none): nothing in the evidence measures the same thing. Same topic is not enough.
- not an indicator: activities, deliverables, assumptions, or targets with no measure."""


@lru_cache(maxsize=4)
def load_rubric(path: Path = config.RUBRIC_MD) -> str:
    path = Path(path)
    if not path.exists():
        return _FALLBACK
    text = path.read_text(encoding="utf-8")
    m = _MARKERS.search(text)
    return (m.group(1) if m else text).strip()


def rubric_examples(path: Path = config.RUBRIC_MD) -> list[dict]:
    """The worked examples as records (document_text, indikit, label, why) - used by the tests and the docs."""
    rows = []
    for line in load_rubric(path).splitlines():
        if line.startswith("|") and not line.startswith("|---") and "Document text" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 4:
                rows.append({"document_text": cells[0].strip('"'), "indikit": cells[1], "label": cells[2], "why": cells[3]})
    return rows
