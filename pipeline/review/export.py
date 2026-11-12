"""``export-review``: one sheet per country with the rows flagged ``needs_review = true``."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from pipeline import config
from pipeline.metadata.models import IndicatorMetadata

REVIEW_COLUMNS = ["doc_id", "document", "doc_indicator_id", "document_indicator", "level", "baseline", "target", "indikit_id", "indikit_name", "indikit_wording", "indikit_link", "match_type", "source",
                  "alternate_ids", "needs_review_reason", "reason", "evidence_quote", "location", "reviewer_decision", "correct_indikit_id", "comment", "reviewer"]
DECISION_HELP = " | ".join(config.REVIEW_DECISIONS)


def _sheet_name(country: str) -> str:
    s = "".join(c if c.isalnum() or c in " _-" else "_" for c in (country or "unknown"))[:31]
    return s or "unknown"


def review_rows(doc_ind: pd.DataFrame, summary: pd.DataFrame, metadata: dict[str, IndicatorMetadata], *, all_rows: bool = False) -> pd.DataFrame:
    d = doc_ind.copy()
    if "needs_review" in d.columns and not all_rows:
        d = d[d["needs_review"].astype(str).str.lower().isin(("true", "1"))]
    paths = summary.set_index("doc_id")["relative_path"].to_dict() if len(summary) else {}
    rows = []
    for _, r in d.iterrows():
        iid = str(r.get("indikit_id") or "").replace("nan", "")
        md = metadata.get(iid)
        rows.append({"doc_id": r["doc_id"], "document": paths.get(r["doc_id"], ""), "doc_indicator_id": r["doc_indicator_id"], "document_indicator": r["indicator_text"], "level": r.get("level", ""), "baseline": r.get("baseline", ""),
                     "target": r.get("target", ""), "indikit_id": iid, "indikit_name": (md.name if md else r.get("indikit_name", "")), "indikit_wording": (md.kpi_wording_en if md else ""), "indikit_link": (md.url if md and md.url else (f"https://zoa.indikit.net/indicator/{iid}" if iid else "")),
                     "match_type": r.get("match_type", ""), "source": r.get("source", ""), "alternate_ids": r.get("alternate_ids", ""), "needs_review_reason": r.get("needs_review_reason", ""), "reason": r.get("reason", ""),
                     "evidence_quote": r.get("evidence_quote", ""), "location": r.get("location", ""), "reviewer_decision": "", "correct_indikit_id": "", "comment": "", "reviewer": ""})
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def export_review(latest_dir: Path = config.LATEST_DIR, out_path: Optional[Path] = None, metadata: Optional[dict[str, IndicatorMetadata]] = None, *, all_rows: bool = False) -> Path:
    latest_dir = Path(latest_dir)
    doc_ind = pd.read_csv(latest_dir / "document_indicators.csv", dtype=str, keep_default_na=False)
    summary = pd.read_csv(latest_dir / "document_summary.csv", dtype=str, keep_default_na=False)
    if metadata is None:
        from pipeline.metadata.build_metadata import read_jsonl

        metadata = {r.indikit_id: r for r in read_jsonl()}
    rows = review_rows(doc_ind, summary, metadata, all_rows=all_rows)
    out_path = Path(out_path or latest_dir / "review.xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame({"how to review": [f"Fill 'reviewer_decision' with one of: {DECISION_HELP}.", "'other_id': put the right IndiKit ID in 'correct_indikit_id'.", "'reject': the document text is not a match (or not an indicator).",
                                        "Add your name in 'reviewer'. Leave rows you did not check empty.", "Then run: python -m pipeline.run import-review --file review.xlsx"]}).to_excel(xw, sheet_name="README", index=False)
        if len(rows):
            for country, g in rows.groupby("country" if "country" in rows.columns else rows["doc_id"].map(summary.set_index("doc_id")["country"].to_dict() if len(summary) else {}), sort=True):
                g.to_excel(xw, sheet_name=_sheet_name(str(country)), index=False)
        else:
            rows.to_excel(xw, sheet_name="no rows", index=False)
    return out_path
