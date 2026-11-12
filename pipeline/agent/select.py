"""Pair selection before answering (docs/phase_3.md Task 4): hard skips, retrieval gate, optional sector routing.

Asking every indicator of every document costs documents x ~350 calls. From ``outputs/kb/pair_scores.*`` a pair
(document, IndiKit indicator) is kept when its best score is >= ``tau`` **or** the indicator is among the
document's top-``M`` by score. ``tau`` defaults to the ``GATE_TAU_PERCENTILE``-th percentile of all pair scores
and is tuned on the dev split when verified gold exists (``tune_gate``). The gate uses ``best_rerank_score`` when
the scores were computed with the re-ranker, else ``best_vector_score``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.metadata.models import IndicatorMetadata
from pipeline.preprocess.models import ManifestRow

SELECTED_COLUMNS = ("doc_id", "indikit_id", "score", "score_kind", "rank_in_doc", "reason", "sector", "on_site", "routed_sectors")


def load_pair_scores(path: Path = config.PAIR_SCORES_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype={"doc_id": str, "indikit_id": str})
    if path.exists():
        return pd.read_csv(path, dtype={"doc_id": str, "indikit_id": str})
    raise FileNotFoundError(f"{path} (or .csv) not found; run: python -m pipeline.run score-pairs --all")


def score_kind(df: pd.DataFrame) -> str:
    """'rerank' when re-rank scores are present for (nearly) all rows, else 'vector'."""
    if "best_rerank_score" in df.columns and df["best_rerank_score"].notna().mean() >= 0.95 and len(df):
        return "rerank"
    return "vector"


def with_gate_score(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    kind = score_kind(df)
    out = df.copy()
    out["doc_id"] = out["doc_id"].astype(str)
    out["indikit_id"] = out["indikit_id"].astype(str)
    col = "best_rerank_score" if kind == "rerank" else "best_vector_score"
    out["score"] = pd.to_numeric(out[col], errors="coerce").fillna(pd.to_numeric(out["best_vector_score"], errors="coerce")).fillna(-1e9)
    return out, kind


def resolve_tau(scores: Iterable[float], tau: Optional[float] = None, percentile: float = config.GATE_TAU_PERCENTILE) -> float:
    if tau is not None:
        return float(tau)
    arr = np.asarray(list(scores), dtype=float)
    return float(np.percentile(arr, percentile)) if arr.size else 0.0


def document_sectors(df: pd.DataFrame, metadata: dict[str, IndicatorMetadata], tau: float, top_n: int = config.SECTOR_ROUTE_TOP_N) -> dict[str, set[str]]:
    """Sectors a document covers, without an LLM: a sector is kept when the mean of its top-N pair scores >= tau."""
    sector_of = {i: (m.primary_sector or "(unknown)") for i, m in metadata.items()}
    d = df[["doc_id", "indikit_id", "score"]].copy()
    d["sector"] = d["indikit_id"].map(sector_of).fillna("(unknown)")
    out: dict[str, set[str]] = {}
    for (doc, sec), g in d.groupby(["doc_id", "sector"]):
        top = g["score"].nlargest(top_n)
        if float(top.mean()) >= tau:
            out.setdefault(doc, set()).add(sec)
    return out


@dataclass
class SelectionResult:
    selected: pd.DataFrame
    tau: float
    top_m: int
    score_kind: str
    route_sectors: bool
    include_export_only: bool
    skipped_documents: list[tuple[str, str]] = field(default_factory=list)
    per_document: pd.DataFrame = field(default_factory=pd.DataFrame)
    routed_sectors: dict[str, list[str]] = field(default_factory=dict)
    n_candidates: int = 0
    sector_tau: float = 0.0

    def summary(self) -> dict:
        return {"tau": round(self.tau, 6), "top_m": self.top_m, "score_kind": self.score_kind, "route_sectors": self.route_sectors, "sector_tau": round(self.sector_tau, 6), "include_export_only": self.include_export_only,
                "documents": int(self.selected["doc_id"].nunique()) if len(self.selected) else 0, "pairs_selected": int(len(self.selected)), "pairs_candidates": self.n_candidates,
                "skipped_documents": len(self.skipped_documents), "export_only_pairs": int((~self.selected["on_site"]).sum()) if len(self.selected) else 0,
                "mean_pairs_per_document": round(float(len(self.selected)) / max(1, self.selected["doc_id"].nunique()), 1) if len(self.selected) else 0.0}


def hard_skip_reason(row: ManifestRow) -> Optional[str]:
    if row.parse_status == "failed":
        return "parse failed"
    if row.indicators_elsewhere and (row.n_logframe_tables + row.n_logframe_sections) == 0:
        return "indicators in another document (reported, not queried)"
    return None


def select_pairs(scores: pd.DataFrame, manifest: dict[str, ManifestRow], metadata: dict[str, IndicatorMetadata], *, tau: Optional[float] = config.GATE_TAU, tau_percentile: float = config.GATE_TAU_PERCENTILE,
                 top_m: int = config.GATE_TOP_M, route_sectors: bool = config.ROUTE_SECTORS, include_export_only: bool = config.INCLUDE_EXPORT_ONLY, doc_ids: Optional[Iterable[str]] = None) -> SelectionResult:
    df, kind = with_gate_score(scores)
    scores_all = df["score"].tolist()
    tau_used = resolve_tau(scores_all, tau, tau_percentile)   # percentile over ALL pair scores (before document filters), so it is stable per run
    wanted = set(doc_ids) if doc_ids is not None else set(df["doc_id"].unique())
    skipped: list[tuple[str, str]] = []
    keep_docs: set[str] = set()
    for d in sorted(wanted):
        row = manifest.get(d)
        if row is None:
            skipped.append((d, "not in manifest"))
            continue
        why = hard_skip_reason(row)
        if why:
            skipped.append((d, why))
        else:
            keep_docs.add(d)
    df = df[df["doc_id"].isin(keep_docs) & df["indikit_id"].isin(set(metadata))].copy()
    df["sector"] = df["indikit_id"].map(lambda i: metadata[i].primary_sector or "(unknown)")
    df["on_site"] = df["indikit_id"].map(lambda i: bool(metadata[i].on_site))
    if not include_export_only:
        df = df[df["on_site"]]
    n_candidates = len(df)
    routed: dict[str, list[str]] = {}
    sector_tau = resolve_tau(scores_all, None, config.SECTOR_ROUTE_PERCENTILE)
    if route_sectors:
        secs = document_sectors(df, metadata, sector_tau)
        routed = {d: sorted(s) for d, s in secs.items()}
    df = df.sort_values(["doc_id", "score", "indikit_id"], ascending=[True, False, True])
    df["rank_in_doc"] = df.groupby("doc_id").cumcount() + 1
    above = df["score"] >= tau_used
    if route_sectors:
        in_sector = df.apply(lambda r: r["sector"] in set(routed.get(r["doc_id"], [])), axis=1) if len(df) else pd.Series([], dtype=bool)
        sub = df[in_sector].copy()
        sub["rank_in_sector_docs"] = sub.groupby("doc_id").cumcount() + 1
        topm = pd.Series(False, index=df.index)
        topm.loc[sub.index[sub["rank_in_sector_docs"] <= top_m]] = True
    else:
        topm = df["rank_in_doc"] <= top_m
    df["reason"] = np.select([above & topm, above, topm], ["score>=tau;top_m", "score>=tau", "top_m"], default="")
    df["score_kind"] = kind
    df["routed_sectors"] = df["doc_id"].map(lambda d: "; ".join(routed.get(d, [])) if route_sectors else "")
    selected = df[df["reason"] != ""].copy()
    selected["score"] = selected["score"].round(6)
    selected = selected[list(SELECTED_COLUMNS)].reset_index(drop=True)
    per_doc = selected.groupby("doc_id").agg(pairs=("indikit_id", "size"), by_tau=("reason", lambda s: int(s.str.contains("tau").sum())), by_top_m=("reason", lambda s: int(s.str.contains("top_m").sum())),
                                            export_only=("on_site", lambda s: int((~s).sum()))).reset_index() if len(selected) else pd.DataFrame(columns=["doc_id", "pairs", "by_tau", "by_top_m", "export_only"])
    for d, why in skipped:
        per_doc = pd.concat([per_doc, pd.DataFrame([{"doc_id": d, "pairs": 0, "by_tau": 0, "by_top_m": 0, "export_only": 0, "skipped": why}])], ignore_index=True)
    if "skipped" not in per_doc.columns:
        per_doc["skipped"] = ""
    per_doc["skipped"] = per_doc["skipped"].fillna("")
    res = SelectionResult(selected=selected, tau=tau_used, top_m=top_m, score_kind=kind, route_sectors=route_sectors, include_export_only=include_export_only, skipped_documents=skipped,
                          per_document=per_doc.sort_values("doc_id").reset_index(drop=True), routed_sectors=routed, n_candidates=n_candidates)
    res.sector_tau = sector_tau
    return res


def write_selection(result: SelectionResult, run_dir: Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    p1 = run_dir / "selected_pairs.csv"
    p2 = run_dir / "selected_pairs_per_document.csv"
    p3 = run_dir / "selection.json"
    result.selected.to_csv(p1, index=False)
    result.per_document.to_csv(p2, index=False)
    p3.write_text(json.dumps({**result.summary(), "skipped": result.skipped_documents, "routed_sectors": result.routed_sectors}, indent=1), encoding="utf-8")
    return {"selected_pairs": p1, "per_document": p2, "selection": p3}


def read_selection(run_dir: Path) -> pd.DataFrame:
    p = Path(run_dir) / "selected_pairs.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run: python -m pipeline.run select-pairs --run-id <id>")
    return pd.read_csv(p, dtype={"doc_id": str, "indikit_id": str})


# --------------------------------------------------------------------------- #
# Gate evaluation and tuning against gold (dev split only)
# --------------------------------------------------------------------------- #
def gate_recall(scores: pd.DataFrame, gold_pairs: set[tuple[str, str]], *, tau: float, top_m: int, metadata: Optional[dict[str, IndicatorMetadata]] = None, route_sectors: bool = False) -> dict:
    """Share of gold-present pairs that pass the gate, and the number of pairs asked, for one (tau, M) setting."""
    df, kind = with_gate_score(scores)
    docs = {d for d, _ in gold_pairs}
    df = df[df["doc_id"].isin(docs)].copy()
    df = df.sort_values(["doc_id", "score", "indikit_id"], ascending=[True, False, True])
    df["rank_in_doc"] = df.groupby("doc_id").cumcount() + 1
    passed = (df["score"] >= tau) | (df["rank_in_doc"] <= top_m)
    if route_sectors and metadata:
        secs = document_sectors(with_gate_score(scores)[0], metadata, resolve_tau(with_gate_score(scores)[0]["score"], None, config.SECTOR_ROUTE_PERCENTILE))
        sector_of = {i: (m.primary_sector or "(unknown)") for i, m in metadata.items()}
        in_sec = df.apply(lambda r: sector_of.get(r["indikit_id"], "(unknown)") in secs.get(r["doc_id"], set()), axis=1)
        passed = (df["score"] >= tau) | ((df["rank_in_doc"] <= top_m) & in_sec)
    kept = set(zip(df.loc[passed, "doc_id"], df.loc[passed, "indikit_id"]))
    hit = sum(1 for p in gold_pairs if p in kept)
    return {"tau": round(float(tau), 6), "top_m": int(top_m), "route_sectors": bool(route_sectors), "gold_pairs": len(gold_pairs), "recalled": hit, "recall": (hit / len(gold_pairs)) if gold_pairs else None,
            "pairs_asked": int(passed.sum()), "pairs_per_document": round(float(passed.sum()) / max(1, len(docs)), 1), "score_kind": kind, "missed": sorted(p for p in gold_pairs if p not in kept)}


def tune_gate(scores: pd.DataFrame, gold_pairs: set[tuple[str, str]], *, target_recall: float = config.GATE_TARGET_RECALL, percentiles: Iterable[float] = (50, 60, 70, 80, 85, 90, 95, 97, 99),
              ms: Iterable[int] = (0, 20, 40, 60, 80, 100, 150, 200), metadata: Optional[dict[str, IndicatorMetadata]] = None, with_routing: bool = True) -> tuple[pd.DataFrame, Optional[dict]]:
    """Trade-off curve over (tau percentile, M[, routing]); best = cheapest setting with recall >= target."""
    df, _ = with_gate_score(scores)
    rows = []
    for pct in percentiles:
        tau = resolve_tau(df["score"], None, pct)
        for m in ms:
            for route in ((False, True) if (with_routing and metadata) else (False,)):
                r = gate_recall(scores, gold_pairs, tau=tau, top_m=m, metadata=metadata, route_sectors=route)
                r.pop("missed", None)
                r["tau_percentile"] = pct
                rows.append(r)
    curve = pd.DataFrame(rows)
    ok = curve[(curve["recall"].fillna(0) >= target_recall)]
    best = ok.sort_values(["pairs_asked", "tau_percentile"], ascending=[True, False]).iloc[0].to_dict() if len(ok) else None
    return curve, best
