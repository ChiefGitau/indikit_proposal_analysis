"""Experiment runner (paper Section 5.1; docs/plan.md Section 6.3). Phase 3.

Experiments:
- ``benchmark``       plain RAG: fixed-size chunks, MiniLM, no outline/table KBs, no re-rank, no extensions
- ``enhanced_rag``    + layout-aware preprocessing, three KBs, re-ranking
- ``plus_knowledge``  + expert knowledge in the prompt
- ``plus_search_term`` + search terms in the retrieval query (= full method)
- ``model_comparison`` full method with 2-3 models
- ``gate_sensitivity`` recall and cost at different retrieval-gate thresholds

Only the experiment names and the results-table writer exist in Phase 1.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from pipeline import config

EXPERIMENTS: tuple[str, ...] = (
    "benchmark",
    "enhanced_rag",
    "plus_knowledge",
    "plus_search_term",
    "model_comparison",
    "gate_sensitivity",
)

RESULT_COLUMNS: tuple[str, ...] = ("experiment", "model", "split", "acc_dc", "precision", "recall", "f1", "acc_de", "match_type_acc", "reference_detection", "llm_calls", "minutes")


def run_experiment(name: str, split: str = "test") -> Mapping[str, object]:
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment '{name}'; choose from {EXPERIMENTS}")
    raise NotImplementedError("Phase 3")


def run_all(split: str = "test") -> list[Mapping[str, object]]:
    raise NotImplementedError("Phase 3")


def write_results_table(rows: Iterable[Mapping[str, object]], path: Path = config.EVAL_OUTPUT_DIR / "results.csv") -> Path:
    """Write experiment rows as CSV and a Markdown twin (``results.md``) like the paper's Table 4."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESULT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))

    md = ["| " + " | ".join(RESULT_COLUMNS) + " |", "|" + "---|" * len(RESULT_COLUMNS)]
    md += ["| " + " | ".join(fmt(row.get(k)) for k in RESULT_COLUMNS) + " |" for row in rows]
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return path
