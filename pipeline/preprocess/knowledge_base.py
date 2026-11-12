"""Multi-type knowledge base (paper Section 3.3.2): text, outline and table KBs with a vector
index per document, stored under ``outputs/kb/<doc_id>/``. Phase 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_knowledge_base(doc_path: Path, out_dir: Path) -> Any:
    raise NotImplementedError("Phase 2")


def load_knowledge_base(doc_id: str, kb_dir: Path) -> Any:
    raise NotImplementedError("Phase 2")
