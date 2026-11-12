"""Pair selection: hard skips, retrieval gate, sector routing, tuning (docs/phase_3.md Task 4)."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.agent.select import document_sectors, gate_recall, hard_skip_reason, load_pair_scores, resolve_tau, select_pairs, tune_gate, with_gate_score, write_selection
from pipeline.preprocess.models import ManifestRow
from tests.fixtures_agent import METADATA


def _manifest():
    base = dict(relative_path="x", region="africa", country="t", file_type="docx", sha256="0" * 64)
    return {
        "ok": ManifestRow(doc_id="ok", n_logframe_tables=1, **base),
        "failed": ManifestRow(doc_id="failed", parse_status="failed", **base),
        "elsewhere": ManifestRow(doc_id="elsewhere", indicators_elsewhere=True, n_logframe_tables=0, n_logframe_sections=0, **base),
        "both": ManifestRow(doc_id="both", indicators_also_elsewhere=True, n_logframe_tables=1, **base),
    }


def _scores():
    rows = []
    for d in ("ok", "failed", "elsewhere", "both"):
        for i, s in (("5138", 0.9), ("5330", 0.5), ("5140", 0.2), ("9000", 0.1)):
            rows.append({"doc_id": d, "indikit_id": i, "best_vector_score": s, "best_rerank_score": s if d != "both" else None, "top_entry_id": "e", "top_is_logframe": True, "top_kb_type": "table"})
    return pd.DataFrame(rows)


def test_hard_skips_gate_and_reasons(tmp_path):
    m = _manifest()
    assert hard_skip_reason(m["failed"]) == "parse failed" and "another document" in hard_skip_reason(m["elsewhere"]) and hard_skip_reason(m["both"]) is None and hard_skip_reason(m["ok"]) is None
    res = select_pairs(_scores(), m, METADATA, tau=0.6, top_m=2, include_export_only=True)
    assert set(res.selected["doc_id"]) == {"ok", "both"} and {d for d, _ in res.skipped_documents} == {"failed", "elsewhere"}
    ok = res.selected[res.selected["doc_id"] == "ok"].set_index("indikit_id")
    assert ok.loc["5138", "reason"] == "score>=tau;top_m" and ok.loc["5330", "reason"] == "top_m" and "5140" not in ok.index
    assert bool(ok.loc["5138", "on_site"]) is True
    res2 = select_pairs(_scores(), m, METADATA, tau=0.15, top_m=0, include_export_only=False)
    assert "5140" not in set(res2.selected["indikit_id"]) and set(res2.selected["indikit_id"]) == {"5138", "5330"}
    paths = write_selection(res, tmp_path)
    assert paths["selected_pairs"].exists() and paths["per_document"].exists() and paths["selection"].exists()
    per = pd.read_csv(paths["per_document"])
    assert set(per["doc_id"]) == {"ok", "both", "failed", "elsewhere"} and int(per.set_index("doc_id").loc["ok", "pairs"]) == 2
    assert res.summary()["pairs_selected"] == 4 and res.summary()["export_only_pairs"] == 0


def test_percentile_tau_score_kind_and_routing():
    df, kind = with_gate_score(_scores())
    assert kind == "vector"   # one document has no re-rank scores
    assert resolve_tau([0, 1, 2, 3, 4], None, 50) == 2.0 and resolve_tau([0, 1], 0.7) == 0.7
    secs = document_sectors(df, METADATA, tau=0.4, top_n=1)
    assert secs["ok"] == {"Water, Sanitation and Hygiene"}
    res = select_pairs(_scores(), _manifest(), METADATA, tau=0.95, top_m=1, route_sectors=True)
    assert set(res.selected["indikit_id"]) == {"5138"} and res.routed_sectors["ok"] == ["Water, Sanitation and Hygiene"] and (res.selected["routed_sectors"] != "").all()


def test_gate_recall_and_tuning():
    gold = {("ok", "5138"), ("ok", "5330"), ("both", "5140")}
    r = gate_recall(_scores(), gold, tau=0.6, top_m=2)
    assert r["recalled"] == 2 and r["missed"] == [("both", "5140")] and r["pairs_asked"] == 4
    curve, best = tune_gate(_scores(), gold, target_recall=0.95, percentiles=(10, 50, 90), ms=(0, 2, 3), metadata=METADATA)
    assert len(curve) == 3 * 3 * 2 and best is not None and best["recall"] >= 0.95
    assert best["pairs_asked"] == curve[curve["recall"] >= 0.95]["pairs_asked"].min()


def test_load_pair_scores_csv_fallback(tmp_path):
    _scores().to_csv(tmp_path / "pair_scores.csv", index=False)
    df = load_pair_scores(tmp_path / "pair_scores.parquet")
    assert len(df) == 16 and df["indikit_id"].iloc[0] == "5138" and isinstance(df["indikit_id"].iloc[0], str)
    with pytest.raises(FileNotFoundError):
        load_pair_scores(tmp_path / "missing.parquet")
