"""``import-review``: read the filled review file, validate decisions, write ``review_decisions.csv``, apply the decisions
to ``document_indicators_reviewed.csv`` (raw output unchanged), propose overrides as a separate diff file and,
optionally, export reviewed documents as gold files (Phase 1 format, ``verified = true``)."""
from __future__ import annotations

import csv
import difflib
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from pipeline import config
from pipeline.metadata.models import IndicatorMetadata

DECISIONS = set(config.REVIEW_DECISIONS)


def read_review_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        frames = [df for name, df in sheets.items() if name not in ("README", "no rows") and "doc_indicator_id" in df.columns]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.fillna("")


def validate_decisions(df: pd.DataFrame, metadata: dict[str, IndicatorMetadata]) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    rows = []
    for i, r in df.iterrows():
        dec = str(r.get("reviewer_decision", "")).strip().lower()
        if not dec:
            continue
        cid = str(r.get("correct_indikit_id", "")).strip()
        if dec not in DECISIONS:
            errors.append(f"row {i + 2}: unknown decision '{dec}' (use {', '.join(sorted(DECISIONS))})")
            continue
        if dec == "other_id":
            if not cid:
                errors.append(f"row {i + 2}: other_id needs correct_indikit_id")
                continue
            if cid not in metadata:
                errors.append(f"row {i + 2}: correct_indikit_id '{cid}' not in the metadata")
                continue
        rows.append({"doc_id": r["doc_id"], "doc_indicator_id": r["doc_indicator_id"], "document_indicator": r.get("document_indicator", ""), "indikit_id": str(r.get("indikit_id", "")).strip(), "match_type": r.get("match_type", ""),
                     "reviewer_decision": dec, "correct_indikit_id": cid, "comment": r.get("comment", ""), "reviewer": r.get("reviewer", "")})
    return pd.DataFrame(rows, columns=["doc_id", "doc_indicator_id", "document_indicator", "indikit_id", "match_type", "reviewer_decision", "correct_indikit_id", "comment", "reviewer"]), errors


def apply_decisions(doc_ind: pd.DataFrame, decisions: pd.DataFrame, metadata: dict[str, IndicatorMetadata]) -> pd.DataFrame:
    """Reviewer decisions always override the pipeline output. Returns the reviewed copy."""
    out = doc_ind.copy()
    for c in ("reviewed", "reviewer_decision", "reviewer", "review_comment"):
        out[c] = "" if c != "reviewed" else "False"
    if "needs_review" in out.columns:
        out["needs_review"] = out["needs_review"].astype(str)
    idx = {(str(r["doc_id"]), str(r["doc_indicator_id"])): i for i, r in out.iterrows()}
    for _, d in decisions.iterrows():
        i = idx.get((str(d["doc_id"]), str(d["doc_indicator_id"])))
        if i is None:
            continue
        dec = d["reviewer_decision"]
        out.at[i, "reviewed"], out.at[i, "reviewer_decision"], out.at[i, "reviewer"], out.at[i, "review_comment"] = "True", dec, d["reviewer"], d["comment"]
        if dec == "accept":
            pass
        elif dec in ("change_to_verbatim", "change_to_close"):
            out.at[i, "match_type"] = dec.split("_to_")[1]
        elif dec == "reject":
            out.at[i, "match_type"], out.at[i, "indikit_id"], out.at[i, "indikit_name"], out.at[i, "alternate_ids"] = "none", "", "", ""
        elif dec == "other_id":
            cid = d["correct_indikit_id"]
            out.at[i, "indikit_id"], out.at[i, "indikit_name"] = cid, metadata[cid].name if cid in metadata else ""
            if str(out.at[i, "match_type"]) not in ("verbatim", "close"):
                out.at[i, "match_type"] = "close"
        out.at[i, "needs_review"] = "False"
        out.at[i, "source"] = "reviewer"
    return out


def _yaml_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def propose_overrides(decisions: pd.DataFrame, metadata: dict[str, IndicatorMetadata], overrides_path: Path = config.OVERRIDES_YAML) -> tuple[str, str]:
    """YAML text of proposed additions (knowledge notes with reviewer-confirmed decisions, search terms from the document
    wording) and a unified diff against the current overrides file. Nothing is written to overrides.yaml."""
    import yaml

    current = yaml.safe_load(Path(overrides_path).read_text(encoding="utf-8")) if Path(overrides_path).exists() else {}
    current = current or {}
    proposed: dict[str, dict] = {}
    for _, d in decisions.iterrows():
        dec = d["reviewer_decision"]
        iid = d["correct_indikit_id"] if dec == "other_id" else d["indikit_id"]
        if not iid or iid not in metadata:
            continue
        md = metadata[iid]
        entry = proposed.setdefault(iid, {"name": md.name, "knowledge_append": [], "search_terms_add": []})
        text = str(d["document_indicator"]).strip()
        who = d["reviewer"] or "reviewer"
        if dec in ("accept", "change_to_verbatim", "change_to_close", "other_id"):
            mt = {"change_to_verbatim": "verbatim", "change_to_close": "close"}.get(dec, d["match_type"] or "close")
            entry["knowledge_append"].append(f"Confirmed {mt} match ({who}): \"{text}\"")
            term = re.sub(r"\s+", " ", re.sub(r"[^\w%# ]+", " ", text.lower())).strip()
            if 4 <= len(term) <= 80 and term not in (t.lower() for t in md.search_terms):
                entry["search_terms_add"].append(term)
        elif dec == "reject":
            entry["knowledge_append"].append(f"Does NOT count ({who}): \"{text}\"")
    lines = ["# Proposed additions from import-review - NOT applied automatically. Review, then merge into pipeline/metadata/overrides.yaml.", "# 'knowledge_append' lines are suggestions to add to the record's knowledge note (the overrides schema uses 'knowledge' as a full replacement)."]
    merged = dict(current)
    for iid, e in sorted(proposed.items()):
        if not e["knowledge_append"] and not e["search_terms_add"]:
            continue
        lines.append(f'"{iid}":')
        lines.append(f"  name: {_yaml_quote(e['name'])}")
        if e["knowledge_append"]:
            lines.append("  knowledge_append:")
            lines += [f"    - {_yaml_quote(k)}" for k in dict.fromkeys(e["knowledge_append"])]
        if e["search_terms_add"]:
            lines.append("  search_terms_add: [" + ", ".join(_yaml_quote(t) for t in dict.fromkeys(e["search_terms_add"])) + "]")
        cur = dict(merged.get(iid) or {})
        cur.setdefault("name", e["name"])
        cur["search_terms_add"] = list(dict.fromkeys(list(cur.get("search_terms_add", [])) + e["search_terms_add"]))
        if e["knowledge_append"]:
            cur["knowledge"] = ((cur.get("knowledge") or md_knowledge(metadata, iid)) + " " + " ".join(dict.fromkeys(e["knowledge_append"]))).strip()
        merged[iid] = cur
    proposal = "\n".join(lines) + "\n"
    before = Path(overrides_path).read_text(encoding="utf-8").splitlines(keepends=True) if Path(overrides_path).exists() else []
    after = yaml.safe_dump(merged, allow_unicode=True, sort_keys=True).splitlines(keepends=True)
    before_norm = yaml.safe_dump(current, allow_unicode=True, sort_keys=True).splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(before_norm, after, fromfile="overrides.yaml (current, normalised)", tofile="overrides.yaml (proposed)"))
    return proposal, diff


def md_knowledge(metadata: dict[str, IndicatorMetadata], iid: str) -> str:
    return (metadata[iid].knowledge or "") if iid in metadata else ""


def export_gold(reviewed: pd.DataFrame, summary: pd.DataFrame, gold_dir: Path, *, reviewer: str, doc_ids: Optional[list[str]] = None, split: str = "dev") -> list[Path]:
    """Write ``<doc_id>.csv`` gold files (Phase 1 format) for fully reviewed documents and append them to ``index.csv``."""
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index_path = gold_dir / "index.csv"
    existing = pd.read_csv(index_path, dtype=str, keep_default_na=False) if index_path.exists() else pd.DataFrame(columns=["doc_id", "source_file", "language", "has_logframe", "indicators_in_other_document", "labelled_by", "verified", "split"])
    s = summary.set_index("doc_id") if len(summary) else pd.DataFrame()
    for doc_id, g in reviewed.groupby("doc_id"):
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        g = g[g["is_indicator"].astype(str).str.lower().isin(("true", "1"))] if "is_indicator" in g.columns else g
        rows = []
        for _, r in g.iterrows():
            mt = str(r["match_type"]) if str(r["match_type"]) in ("verbatim", "close", "none") else "none"
            iid = str(r["indikit_id"]) if mt != "none" else ""
            rows.append({"doc_indicator_id": r["doc_indicator_id"], "indicator_text": r["indicator_text"], "level": r.get("level", "unclear") or "unclear", "baseline": r.get("baseline", "") or "", "target": r.get("target", "") or "",
                         "unit": r.get("unit", "") or "", "location": r.get("location", "") or "", "indikit_id": iid, "match_type": mt})
        p = gold_dir / f"{doc_id}.csv"
        with p.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["doc_indicator_id", "indicator_text", "level", "baseline", "target", "unit", "location", "indikit_id", "match_type"])
            w.writeheader()
            w.writerows(rows)
        written.append(p)
        rel = str(s.loc[doc_id, "relative_path"]) if len(s) and doc_id in s.index else ""
        entry = {"doc_id": doc_id, "source_file": rel.split("proposals/", 1)[-1] if rel else "", "language": (str(s.loc[doc_id, "language"]) if len(s) and doc_id in s.index else ""), "has_logframe": "true",
                 "indicators_in_other_document": (str(s.loc[doc_id, "referenced_documents"]) if len(s) and doc_id in s.index else ""), "labelled_by": reviewer, "verified": "true", "split": split}
        existing = existing[existing["doc_id"] != doc_id]
        existing = pd.concat([existing, pd.DataFrame([entry])], ignore_index=True)
    existing.to_csv(index_path, index=False)
    return written


def import_review(path: Path, latest_dir: Path = config.LATEST_DIR, metadata: Optional[dict[str, IndicatorMetadata]] = None, *, export_gold_to: Optional[Path] = None, gold_split: str = "dev") -> dict:
    latest_dir = Path(latest_dir)
    if metadata is None:
        from pipeline.metadata.build_metadata import read_jsonl

        metadata = {r.indikit_id: r for r in read_jsonl()}
    df = read_review_file(path)
    decisions, errors = validate_decisions(df, metadata)
    if errors:
        raise ValueError("invalid review file:\n" + "\n".join(errors))
    doc_ind = pd.read_csv(latest_dir / "document_indicators.csv", dtype=str, keep_default_na=False)
    summary = pd.read_csv(latest_dir / "document_summary.csv", dtype=str, keep_default_na=False) if (latest_dir / "document_summary.csv").exists() else pd.DataFrame()
    reviewed = apply_decisions(doc_ind, decisions, metadata)
    out = {"decisions": latest_dir / "review_decisions.csv", "reviewed": latest_dir / "document_indicators_reviewed.csv", "overrides_proposal": latest_dir / "overrides_proposed.yaml", "overrides_diff": latest_dir / "overrides_proposed.diff"}
    decisions.to_csv(out["decisions"], index=False)
    reviewed.to_csv(out["reviewed"], index=False)
    proposal, diff = propose_overrides(decisions, metadata)
    out["overrides_proposal"].write_text(proposal, encoding="utf-8")
    out["overrides_diff"].write_text(diff, encoding="utf-8")
    result = {"paths": out, "n_decisions": int(len(decisions)), "by_decision": decisions["reviewer_decision"].value_counts().to_dict() if len(decisions) else {}, "gold_files": []}
    if export_gold_to is not None and len(decisions):
        reviewers = [r for r in decisions["reviewer"].unique() if r]
        result["gold_files"] = export_gold(reviewed, summary, Path(export_gold_to), reviewer=", ".join(reviewers) or "reviewer", doc_ids=sorted(decisions["doc_id"].unique()), split=gold_split)
    return result
