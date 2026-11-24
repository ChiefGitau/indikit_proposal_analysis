"""Run outputs (docs/phase_3.md Task 6): per-run CSVs / Markdown under ``outputs/runs/<run_id>/`` and, for the
latest full run, a copy in ``outputs/latest/``. All counts are computed in code from the ``DocumentResult``
records; the LLM never summarises anything.
"""
from __future__ import annotations

import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.agent.models import ConsolidatedIndicator, DocumentResult
from pipeline.metadata.models import IndicatorMetadata

DOC_IND_COLUMNS = ["doc_id", "doc_indicator_id", "indicator_text", "level", "baseline", "target", "unit", "location", "match_type", "indikit_id", "indikit_name", "alternate_ids", "source", "reason",
                   "evidence_quote", "needs_review", "needs_review_reason", "country", "region", "is_indicator", "extraction_method", "text_similarity", "nearest_indikit", "export_only"]
MATRIX_COLUMNS = ["doc_id", "indikit_id", "indikit_name", "sector", "match_type", "n_document_indicators", "doc_indicator_ids", "sources", "on_site", "country", "region"]
NOVEL_COLUMNS = ["doc_id", "doc_indicator_id", "indicator_text", "level", "baseline", "target", "unit", "location", "country", "region", "nearest_indikit_ids", "nearest_similarities", "cluster_id"]
SUMMARY_COLUMNS = ["doc_id", "relative_path", "country", "region", "language", "file_type", "parse_status", "total_indicators", "verbatim", "close", "none", "pct_matched", "indicators_elsewhere",
                   "indicators_also_elsewhere", "referenced_documents", "note", "pairs_selected", "pairs_asked", "pairs_cached", "pairs_present", "llm_calls", "cost_usd", "unsupported_dropped",
                   "reverse_recovered", "activities_filtered", "errors", "skipped_reason"]


def needs_review(c: ConsolidatedIndicator, verbatim_warn: int = config.VERBATIM_SIMILARITY_WARN) -> tuple[bool, str]:
    if c.primary is None:
        return False, ""
    reasons = []
    if c.primary.match_type == "close":
        reasons.append("close match")
    if any(m.source == "reverse_check" for m in c.matches):
        reasons.append("reverse-check match")
    if c.alternate_ids:
        reasons.append("has alternates")
    if c.primary.match_type == "verbatim" and c.primary.text_similarity is not None and c.primary.text_similarity < verbatim_warn:
        reasons.append(f"verbatim but low similarity to IndiKit wording ({c.primary.text_similarity:.0f})")
    return bool(reasons), "; ".join(reasons)


def document_indicator_rows(results: Sequence[DocumentResult], metadata: dict[str, IndicatorMetadata]) -> pd.DataFrame:
    rows = []
    for r in results:
        for c in r.consolidated:
            p = c.primary
            nr, why = needs_review(c)
            rows.append({
                "doc_id": r.doc_id, "doc_indicator_id": c.indicator.doc_indicator_id, "indicator_text": c.indicator.indicator_text, "level": c.indicator.level, "baseline": c.indicator.baseline,
                "target": c.indicator.target, "unit": c.indicator.unit, "location": c.indicator.location, "match_type": c.match_type, "indikit_id": p.indikit_id if p else "",
                "indikit_name": (p.indikit_name or metadata.get(p.indikit_id, IndicatorMetadata(indikit_id="x", name="")).name) if p else "", "alternate_ids": "; ".join(c.alternate_ids),
                "source": p.source if p else ("extraction" if c.indicator.source == "extraction" else c.indicator.source), "reason": p.reason if p else "", "evidence_quote": p.evidence_quote if p else "",
                "needs_review": nr, "needs_review_reason": why, "country": r.country, "region": r.region, "is_indicator": c.indicator.is_indicator, "extraction_method": c.indicator.extraction_method,
                "text_similarity": p.text_similarity if p else None, "nearest_indikit": "; ".join(f"{i} ({s:.2f})" for i, s in c.nearest_indikit), "export_only": bool(p and p.indikit_id in metadata and not metadata[p.indikit_id].on_site),
            })
    return pd.DataFrame(rows, columns=DOC_IND_COLUMNS)


def disclosure_matrix_rows(results: Sequence[DocumentResult], metadata: dict[str, IndicatorMetadata]) -> pd.DataFrame:
    rows = []
    for r in results:
        per: dict[str, list[tuple[str, str, str]]] = defaultdict(list)   # indikit_id -> (doc_indicator_id, match_type, source)
        for c in r.consolidated:
            for m in c.matches:
                per[m.indikit_id].append((c.indicator.doc_indicator_id, m.match_type, m.source))
        for iid, items in sorted(per.items()):
            md = metadata.get(iid)
            rows.append({"doc_id": r.doc_id, "indikit_id": iid, "indikit_name": md.name if md else "", "sector": (md.primary_sector if md else "") or "(unknown)",
                         "match_type": "verbatim" if any(mt == "verbatim" for _, mt, _ in items) else "close", "n_document_indicators": len({d for d, _, _ in items}),
                         "doc_indicator_ids": "; ".join(sorted({d for d, _, _ in items})), "sources": "; ".join(sorted({s for _, _, s in items})), "on_site": bool(md.on_site) if md else False,
                         "country": r.country, "region": r.region})
    return pd.DataFrame(rows, columns=MATRIX_COLUMNS)


def cluster_texts(texts: Sequence[str], vectors: Optional[np.ndarray] = None, threshold: float = config.NOVEL_CLUSTER_THRESHOLD) -> list[int]:
    """Greedy clustering in code: each text joins the first cluster whose centroid is within ``threshold`` cosine
    (vectors) or token-set similarity >= threshold*100 (fallback without an embedder). Deterministic."""
    labels: list[int] = []
    if vectors is not None and len(vectors) == len(texts):
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.where(norms == 0, 1, norms)
        centroids: list[np.ndarray] = []
        members: list[list[int]] = []
        for i, x in enumerate(v):
            best, best_sim = -1, -1.0
            for k, cen in enumerate(centroids):
                sim = float(x @ cen / (np.linalg.norm(cen) or 1.0))
                if sim > best_sim:
                    best, best_sim = k, sim
            if best >= 0 and best_sim >= threshold:
                members[best].append(i)
                centroids[best] = v[members[best]].mean(axis=0)
                labels.append(best)
            else:
                centroids.append(x.copy())
                members.append([i])
                labels.append(len(centroids) - 1)
        return labels
    from pipeline.agent.answer import text_similarity

    reps: list[str] = []
    for t in texts:
        k = next((i for i, rep in enumerate(reps) if text_similarity(t, rep) >= threshold * 100), None)
        if k is None:
            reps.append(t)
            k = len(reps) - 1
        labels.append(k)
    return labels


def novel_rows(results: Sequence[DocumentResult], embed_fn=None) -> pd.DataFrame:
    rows = []
    for r in results:
        for c in r.consolidated:
            if c.matches or not c.indicator.is_indicator:
                continue
            rows.append({"doc_id": r.doc_id, "doc_indicator_id": c.indicator.doc_indicator_id, "indicator_text": c.indicator.indicator_text, "level": c.indicator.level, "baseline": c.indicator.baseline,
                         "target": c.indicator.target, "unit": c.indicator.unit, "location": c.indicator.location, "country": r.country, "region": r.region,
                         "nearest_indikit_ids": "; ".join(i for i, _ in c.nearest_indikit), "nearest_similarities": "; ".join(f"{s:.3f}" for _, s in c.nearest_indikit), "cluster_id": -1})
    df = pd.DataFrame(rows, columns=NOVEL_COLUMNS)
    if len(df):
        vectors = embed_fn(df["indicator_text"].tolist()) if embed_fn is not None else None
        df["cluster_id"] = cluster_texts(df["indicator_text"].tolist(), vectors)
    return df


def document_summary_rows(results: Sequence[DocumentResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        real = [c for c in r.consolidated if c.indicator.is_indicator]
        n_v = sum(1 for c in real if c.match_type == "verbatim")
        n_c = sum(1 for c in real if c.match_type == "close")
        n_n = sum(1 for c in real if c.match_type == "none")
        total = len(real)
        note = ""
        if r.indicators_elsewhere:
            note = "indicators are in another document: " + ("; ".join(r.referenced_documents) or "(name not detected)")
        elif r.indicators_also_elsewhere:
            note = "indicators also in another document: " + ("; ".join(r.referenced_documents) or "(name not detected)")
        rows.append({"doc_id": r.doc_id, "relative_path": r.relative_path, "country": r.country, "region": r.region, "language": r.language, "file_type": r.file_type, "parse_status": r.parse_status,
                     "total_indicators": total, "verbatim": n_v, "close": n_c, "none": n_n, "pct_matched": round(100.0 * (n_v + n_c) / total, 1) if total else None, "indicators_elsewhere": r.indicators_elsewhere,
                     "indicators_also_elsewhere": r.indicators_also_elsewhere, "referenced_documents": "; ".join(r.referenced_documents), "note": note, "pairs_selected": r.pairs_selected, "pairs_asked": r.pairs_asked,
                     "pairs_cached": r.pairs_cached, "pairs_present": r.pairs_present, "llm_calls": r.llm_calls, "cost_usd": round(r.cost_usd, 4), "unsupported_dropped": r.unsupported_dropped,
                     "reverse_recovered": int(r.reverse_stats.get("recovered", 0)), "activities_filtered": int(r.novel_stats.get("filtered_activities", 0)) + int(r.novel_stats.get("table_code_activities", 0)),
                     "errors": " || ".join(r.errors)[:1000], "skipped_reason": r.skipped_reason or ""})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or not len(df):
        return "_(none)_\n"
    d = df.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(("" if (isinstance(v, float) and np.isnan(v)) else str(v)).replace("|", "/").replace("\n", " ") for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def portfolio_tables(doc_ind: pd.DataFrame, matrix: pd.DataFrame, novel: pd.DataFrame, summary: pd.DataFrame, metadata: dict[str, IndicatorMetadata]) -> dict[str, pd.DataFrame]:
    real = doc_ind[doc_ind["is_indicator"] == True] if len(doc_ind) else doc_ind  # noqa: E712
    totals = pd.DataFrame([{
        "documents": int(summary["doc_id"].nunique()) if len(summary) else 0, "documents_processed": int((summary["skipped_reason"] == "").sum()) if len(summary) else 0,
        "documents_indicators_elsewhere": int(summary["indicators_elsewhere"].sum()) if len(summary) else 0, "document_indicators": int(len(real)),
        "verbatim": int((real["match_type"] == "verbatim").sum()) if len(real) else 0, "close": int((real["match_type"] == "close").sum()) if len(real) else 0, "none": int((real["match_type"] == "none").sum()) if len(real) else 0,
        "pct_matched": round(100.0 * ((real["match_type"] != "none").sum()) / len(real), 1) if len(real) else None, "indikit_indicators_used": int(matrix["indikit_id"].nunique()) if len(matrix) else 0,
        "indikit_indicators_total": len(metadata), "needs_review": int(real["needs_review"].sum()) if len(real) else 0, "llm_calls": int(summary["llm_calls"].sum()) if len(summary) else 0,
        "cost_usd": round(float(summary["cost_usd"].sum()), 4) if len(summary) else 0.0,
    }])
    by_country = (real.groupby("country").agg(document_indicators=("doc_indicator_id", "size"), verbatim=("match_type", lambda s: int((s == "verbatim").sum())), close=("match_type", lambda s: int((s == "close").sum())),
                                            none=("match_type", lambda s: int((s == "none").sum()))).reset_index() if len(real) else pd.DataFrame(columns=["country", "document_indicators", "verbatim", "close", "none"]))
    if len(by_country):
        by_country["pct_matched"] = ((by_country["verbatim"] + by_country["close"]) / by_country["document_indicators"] * 100).round(1)
        docs_per_country = summary.groupby("country")["doc_id"].nunique() if len(summary) else pd.Series(dtype=int)
        by_country["documents"] = by_country["country"].map(docs_per_country).fillna(0).astype(int)
    by_sector = (matrix.groupby("sector").agg(documents=("doc_id", "nunique"), indikit_indicators=("indikit_id", "nunique"), matches=("indikit_id", "size"), verbatim=("match_type", lambda s: int((s == "verbatim").sum())))
                 .reset_index().sort_values("matches", ascending=False) if len(matrix) else pd.DataFrame(columns=["sector", "documents", "indikit_indicators", "matches", "verbatim"]))
    most_used = (matrix.groupby(["indikit_id", "indikit_name", "sector"]).agg(documents=("doc_id", "nunique"), verbatim=("match_type", lambda s: int((s == "verbatim").sum())), close=("match_type", lambda s: int((s == "close").sum())))
                 .reset_index().sort_values(["documents", "indikit_id"], ascending=[False, True]) if len(matrix) else pd.DataFrame(columns=["indikit_id", "indikit_name", "sector", "documents", "verbatim", "close"]))
    used = set(matrix["indikit_id"]) if len(matrix) else set()
    never = pd.DataFrame([{"indikit_id": i, "indikit_name": m.name, "sector": m.primary_sector, "on_site": m.on_site} for i, m in sorted(metadata.items()) if i not in used])
    clusters = pd.DataFrame(columns=["cluster_id", "size", "documents", "countries", "example", "nearest_indikit"])
    if len(novel):
        g = novel.groupby("cluster_id")
        clusters = pd.DataFrame([{"cluster_id": int(k), "size": len(d), "documents": int(d["doc_id"].nunique()), "countries": "; ".join(sorted(set(d["country"]))), "example": d["indicator_text"].iloc[0],
                                  "nearest_indikit": d["nearest_indikit_ids"].iloc[0]} for k, d in g]).sort_values(["size", "cluster_id"], ascending=[False, True])
    return {"totals": totals, "by_country": by_country, "by_sector": by_sector, "most_used_indikit": most_used, "never_used_indikit": never, "novel_clusters": clusters}


def write_portfolio_summary(tables: dict[str, pd.DataFrame], run_dir: Path, *, note: str = "") -> Path:
    run_dir = Path(run_dir)
    for name, df in tables.items():
        df.to_csv(run_dir / f"portfolio_{name}.csv", index=False)
    t = tables["totals"].iloc[0].to_dict() if len(tables["totals"]) else {}
    md = ["# Portfolio summary", ""]
    if note:
        md += [f"> {note}", ""]
    md += ["## Totals", ""] + [f"- **{k}**: {v}" for k, v in t.items()] + [""]
    md += ["## By country", "", _md_table(tables["by_country"]), "## By IndiKit sector", "", _md_table(tables["by_sector"]), "## Most-used IndiKit indicators", "", _md_table(tables["most_used_indikit"], 30),
           f"## IndiKit indicators never used ({len(tables['never_used_indikit'])})", "", _md_table(tables["never_used_indikit"], 40), "## Clusters of similar novel indicators (candidates to add to IndiKit)", "",
           _md_table(tables["novel_clusters"], 30)]
    p = run_dir / "portfolio_summary.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p


def clean_results(results: Sequence[DocumentResult]) -> list[DocumentResult]:
    """Documents whose answering step completed: skipped documents (reported, not queried) and documents with no failed pair
    and no document-level failure. Used for interim exports while a run is still going."""
    return [r for r in results if r.skipped_reason or (r.pairs_failed == 0 and not any(e.startswith("document failed") for e in r.errors))]


INDIKIT_COLUMNS = ["indikit_id", "name", "sector", "sub_sector", "section", "levels", "quantity", "topic", "kpi_wording_en", "kpi_wording_fr", "on_site", "in_export", "url"]


def indikit_table(metadata: dict[str, IndicatorMetadata]) -> pd.DataFrame:
    """One row per IndiKit indicator (dimension table for BI tools); the first aspect gives sector / sub-sector / section."""
    rows = []
    for iid, m in sorted(metadata.items()):
        a = m.aspect[0] if m.aspect else None
        rows.append({"indikit_id": iid, "name": m.name, "sector": m.primary_sector or "(unknown)", "sub_sector": a.sub_sector if a else "", "section": a.section if a else "",
                     "levels": "; ".join(m.levels), "quantity": getattr(m.quantity, "value", str(m.quantity)), "topic": m.topic or "", "kpi_wording_en": m.kpi_wording_en,
                     "kpi_wording_fr": m.kpi_wording_fr or "", "on_site": m.on_site, "in_export": m.in_export, "url": m.url or ""})
    return pd.DataFrame(rows, columns=INDIKIT_COLUMNS)


def write_run_outputs(results: Sequence[DocumentResult], metadata: dict[str, IndicatorMetadata], run_dir: Path, *, embed_fn=None, note: str = "") -> dict[str, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    doc_ind = document_indicator_rows(results, metadata)
    matrix = disclosure_matrix_rows(results, metadata)
    novel = novel_rows(results, embed_fn)
    summary = document_summary_rows(results)
    paths = {}
    for name, df in (("document_indicators", doc_ind), ("disclosure_matrix", matrix), ("novel_indicators", novel), ("document_summary", summary)):
        p = run_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    tables = portfolio_tables(doc_ind, matrix, novel, summary, metadata)
    paths["portfolio_summary"] = write_portfolio_summary(tables, run_dir, note=note)
    return paths


def write_run_report(run_dir: Path, info: dict) -> Path:
    """``info`` holds config fingerprint, models, versions, counts, usage, runtime, extrapolation (all computed by the caller)."""
    run_dir = Path(run_dir)
    md = [f"# Run report - {info.get('run_id', run_dir.name)}", "", f"Written {time.strftime('%Y-%m-%d %H:%M:%S')}.", ""]
    if info.get("watermark"):
        md += [f"**{info['watermark']}**", ""]
    md += ["## Configuration", ""]
    for k in ("config_fingerprint", "provider", "answer_model", "extraction_model", "prompt_version", "rubric_version", "novel_prompt_version", "reverse_prompt_version", "retrieval", "gate", "route_sectors", "include_export_only", "mode", "documents_requested"):
        if k in info:
            md.append(f"- **{k}**: {info[k]}")
    md += ["", "## Counts", ""]
    for k in ("documents", "documents_processed", "documents_skipped", "documents_failed", "pairs_selected", "pairs_asked", "pairs_cached", "pairs_present", "pairs_failed", "unsupported_dropped", "novel_llm_calls", "reverse_check_candidates", "reverse_check_recovered", "export_only_matches"):
        if k in info:
            md.append(f"- **{k}**: {info[k]}")
    md += ["", "## Usage", ""]
    for k, v in (info.get("usage") or {}).items():
        if k != "per_model":
            md.append(f"- **{k}**: {v}")
    if (info.get("usage") or {}).get("per_model"):
        md += ["", "Per model:", "", _md_table(pd.DataFrame([{"model": m, **d} for m, d in info["usage"]["per_model"].items()]))]
    md += ["", f"- **runtime_seconds**: {info.get('runtime_seconds', '')}", f"- **seconds_per_pair**: {info.get('seconds_per_pair', '')}", f"- **cost_per_pair_usd**: {info.get('cost_per_pair_usd', '')}", f"- **cost_per_document_usd**: {info.get('cost_per_document_usd', '')}", ""]
    if info.get("extrapolation"):
        md += ["## Extrapolation to all documents", ""] + [f"- **{k}**: {v}" for k, v in info["extrapolation"].items()] + [""]
    if info.get("recommendation"):
        md += ["## Recommendation", "", info["recommendation"], ""]
    if info.get("errors"):
        md += ["## Errors", ""] + [f"- {e}" for e in info["errors"][:100]] + [""]
    p = run_dir / "run_report.md"
    p.write_text("\n".join(md), encoding="utf-8")
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return p


def refresh_metadata_dependent_outputs(run_dir: Path, metadata: dict[str, IndicatorMetadata], *, note: str = "") -> dict[str, Path]:
    """Rewrite the output files of a finished run that depend on the IndiKit metadata (``on_site`` / ``export_only``,
    names, sectors) from the per-document results in ``run_dir/docs``, without any LLM call and without touching the
    answers. ``novel_indicators.csv`` is kept as it is (its clusters do not depend on the metadata) and only re-created
    when it is missing. Used after a re-scrape of the IndiKit site."""
    run_dir = Path(run_dir)
    results = []
    for p in sorted((run_dir / "docs").glob("*.json")):
        try:
            results.append(DocumentResult.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not results:
        raise FileNotFoundError(f"no document results in {run_dir / 'docs'}")
    doc_ind = document_indicator_rows(results, metadata)
    matrix = disclosure_matrix_rows(results, metadata)
    summary = document_summary_rows(results)
    novel_path = run_dir / "novel_indicators.csv"
    novel = pd.read_csv(novel_path, dtype={"doc_id": str, "doc_indicator_id": str, "nearest_indikit_ids": str, "nearest_similarities": str}, keep_default_na=False) if novel_path.exists() else novel_rows(results, None)
    if "cluster_id" in novel.columns:
        novel["cluster_id"] = pd.to_numeric(novel["cluster_id"], errors="coerce").fillna(-1).astype(int)
    paths = {}
    for name, df in (("document_indicators", doc_ind), ("disclosure_matrix", matrix), ("novel_indicators", novel), ("document_summary", summary)):
        p = run_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    tables = portfolio_tables(doc_ind, matrix, novel, summary, metadata)
    paths["portfolio_summary"] = write_portfolio_summary(tables, run_dir, note=note)
    return paths


def copy_latest(run_dir: Path, latest_dir: Path = config.LATEST_DIR) -> Path:
    run_dir, latest_dir = Path(run_dir), Path(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for p in run_dir.iterdir():
        if p.is_file() and p.suffix in (".csv", ".md", ".json"):
            shutil.copy2(p, latest_dir / p.name)
    (latest_dir / "LATEST_RUN.txt").write_text(f"{run_dir.name}\n{run_dir}\n", encoding="utf-8")
    return latest_dir


# --------------------------------------------------------------------------- #
# report --doc
# --------------------------------------------------------------------------- #
def document_report_markdown(doc_id: str, doc_ind: pd.DataFrame, summary: pd.DataFrame) -> str:
    """The three-column table MEAL staff use plus the summary line."""
    doc_ind = doc_ind.fillna("")
    summary = summary.fillna("")
    is_ind = doc_ind["is_indicator"].astype(str).str.lower().isin(("true", "1")) if len(doc_ind) else doc_ind.get("is_indicator", [])
    d = doc_ind[(doc_ind["doc_id"] == doc_id) & is_ind].copy() if len(doc_ind) else doc_ind
    if len(d):
        d["indikit_id"] = d["indikit_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    s = summary[summary["doc_id"] == doc_id]
    lines = ["| Indicator in document | IndiKit indicator (ID) | Match |", "|---|---|---|"]
    for _, r in d.iterrows():
        ik = f"{r['indikit_name']} ({r['indikit_id']})" if r["indikit_id"] else "-"
        lines.append(f"| {str(r['indicator_text']).replace('|', '/')} | {ik.replace('|', '/')} | {r['match_type']} |")
    if not len(d):
        lines.append("| _(no indicators found in this document)_ | | |")
    n_v = int((d["match_type"] == "verbatim").sum()) if len(d) else 0
    n_c = int((d["match_type"] == "close").sum()) if len(d) else 0
    total = int(len(d))
    tail = f"Indicators in document: {total}; verbatim: {n_v}; close: {n_c}; total matched: {n_v + n_c}"
    if len(s):
        row = s.iloc[0]
        if row.get("note"):
            tail += f". Referenced document: {row['referenced_documents'] or '(name not detected)'}"
        if row.get("skipped_reason"):
            tail += f". Note: {row['skipped_reason']}"
    return "\n".join(lines) + "\n\n" + tail + "\n"
