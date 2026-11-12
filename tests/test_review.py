"""Review workflow round trip on a small synthetic file (docs/phase_3.md Task 9)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.eval.gold import validate_gold
from pipeline.review.export import REVIEW_COLUMNS, export_review, review_rows
from pipeline.review.import_ import apply_decisions, import_review, propose_overrides, read_review_file, validate_decisions
from tests.fixtures_agent import METADATA


def _latest(tmp_path: Path) -> Path:
    latest = tmp_path / "latest"
    latest.mkdir()
    doc_ind = pd.DataFrame([
        {"doc_id": "d1", "doc_indicator_id": "D001", "indicator_text": "% of HH with soap", "level": "outcome", "baseline": "12%", "target": "60%", "unit": "%", "location": "p.2", "match_type": "close", "indikit_id": "5138", "indikit_name": "Handwashing Facilities With Soap and Water", "alternate_ids": "", "source": "disclosure", "reason": "r", "evidence_quote": "q", "needs_review": True, "needs_review_reason": "close match", "country": "testland", "region": "africa", "is_indicator": True, "extraction_method": "table_code", "text_similarity": 70, "nearest_indikit": "", "export_only": False},
        {"doc_id": "d1", "doc_indicator_id": "D002", "indicator_text": "zebra herds km", "level": "output", "baseline": "", "target": "", "unit": "", "location": "p.3", "match_type": "close", "indikit_id": "5330", "indikit_name": "Drinking Water Quality (residual chlorine)", "alternate_ids": "", "source": "reverse_check", "reason": "r", "evidence_quote": "q", "needs_review": True, "needs_review_reason": "reverse-check match", "country": "testland", "region": "africa", "is_indicator": True, "extraction_method": "text_llm", "text_similarity": 10, "nearest_indikit": "9000 (0.5)", "export_only": False},
        {"doc_id": "d1", "doc_indicator_id": "D003", "indicator_text": "latrines built", "level": "output", "baseline": "0", "target": "40", "unit": "", "location": "p.2", "match_type": "none", "indikit_id": "", "indikit_name": "", "alternate_ids": "", "source": "extraction", "reason": "", "evidence_quote": "", "needs_review": False, "needs_review_reason": "", "country": "testland", "region": "africa", "is_indicator": True, "extraction_method": "table_code", "text_similarity": "", "nearest_indikit": "5140 (0.6)", "export_only": False},
        {"doc_id": "d2", "doc_indicator_id": "D001", "indicator_text": "% hw", "level": "outcome", "baseline": "", "target": "", "unit": "", "location": "p.1", "match_type": "verbatim", "indikit_id": "5138", "indikit_name": "Handwashing Facilities With Soap and Water", "alternate_ids": "5330", "source": "disclosure", "reason": "r", "evidence_quote": "q", "needs_review": True, "needs_review_reason": "has alternates", "country": "otherland", "region": "asia", "is_indicator": True, "extraction_method": "table_code", "text_similarity": 90, "nearest_indikit": "", "export_only": False},
    ])
    doc_ind.to_csv(latest / "document_indicators.csv", index=False)
    pd.DataFrame([{"doc_id": "d1", "relative_path": "proposals/africa/testland/small/a.docx", "country": "testland", "region": "africa", "language": "en", "referenced_documents": ""}, {"doc_id": "d2", "relative_path": "proposals/Asia/otherland/small/b.pdf", "country": "otherland", "region": "asia", "language": "fr", "referenced_documents": "Annex C"}]).to_csv(latest / "document_summary.csv", index=False)
    return latest


def test_export_import_round_trip(tmp_path: Path):
    latest = _latest(tmp_path)
    xlsx = export_review(latest, metadata=METADATA)
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype=str)
    assert set(sheets) == {"README", "testland", "otherland"} and list(sheets["testland"].columns) == REVIEW_COLUMNS and len(sheets["testland"]) == 2
    assert sheets["testland"].iloc[0]["indikit_link"].startswith("https://zoa.indikit.net/indicator/5138") and sheets["testland"]["reviewer_decision"].fillna("").eq("").all()
    # fill the file like a reviewer would
    t = sheets["testland"]
    t.loc[t["doc_indicator_id"] == "D001", ["reviewer_decision", "reviewer", "comment"]] = ["change_to_verbatim", "AB", "same measure"]
    t.loc[t["doc_indicator_id"] == "D002", ["reviewer_decision", "reviewer"]] = ["reject", "AB"]
    o = sheets["otherland"]
    o.loc[o["doc_indicator_id"] == "D001", ["reviewer_decision", "correct_indikit_id", "reviewer"]] = ["other_id", "5330", "CD"]
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        sheets["README"].to_excel(xw, sheet_name="README", index=False)
        t.to_excel(xw, sheet_name="testland", index=False)
        o.to_excel(xw, sheet_name="otherland", index=False)
    res = import_review(xlsx, latest, METADATA, export_gold_to=tmp_path / "gold")
    assert res["n_decisions"] == 3 and res["by_decision"] == {"change_to_verbatim": 1, "reject": 1, "other_id": 1}
    reviewed = pd.read_csv(res["paths"]["reviewed"], dtype=str, keep_default_na=False)
    r = reviewed.set_index(["doc_id", "doc_indicator_id"])
    assert r.loc[("d1", "D001"), "match_type"] == "verbatim" and r.loc[("d1", "D001"), "reviewer"] == "AB" and r.loc[("d1", "D001"), "source"] == "reviewer"
    assert r.loc[("d1", "D002"), "match_type"] == "none" and r.loc[("d1", "D002"), "indikit_id"] == ""
    assert r.loc[("d2", "D001"), "indikit_id"] == "5330" and r.loc[("d2", "D001"), "indikit_name"].startswith("Drinking") and r.loc[("d2", "D001"), "match_type"] == "verbatim"
    assert r.loc[("d1", "D003"), "reviewed"] == "False" and r.loc[("d1", "D003"), "match_type"] == "none"
    # raw output unchanged; decisions and proposal written; overrides.yaml untouched
    raw = pd.read_csv(latest / "document_indicators.csv", dtype=str, keep_default_na=False)
    assert raw.set_index(["doc_id", "doc_indicator_id"]).loc[("d1", "D001"), "match_type"] == "close"
    dec = pd.read_csv(res["paths"]["decisions"], dtype=str)
    assert len(dec) == 3
    prop = res["paths"]["overrides_proposal"].read_text()
    assert '"5138":' in prop and "Confirmed verbatim match (AB)" in prop and "Does NOT count (AB)" in prop and "NOT applied" in prop
    assert res["paths"]["overrides_diff"].read_text().startswith("---")
    # gold export in Phase 1 format, verified = true, validates
    gold_dir = tmp_path / "gold"
    assert {p.name for p in res["gold_files"]} == {"d1.csv", "d2.csv"}
    idx = pd.read_csv(gold_dir / "index.csv", dtype=str)
    assert set(idx["doc_id"]) == {"d1", "d2"} and (idx["verified"] == "true").all() and idx.set_index("doc_id").loc["d2", "labelled_by"] == "AB, CD" and idx.set_index("doc_id").loc["d1", "source_file"] == "africa/testland/small/a.docx"
    report = validate_gold(gold_dir, set(METADATA), include_templates=False)
    assert report.ok, report.errors
    g = pd.read_csv(gold_dir / "d1.csv", dtype=str, keep_default_na=False).set_index("doc_indicator_id")
    assert g.loc["D001", "match_type"] == "verbatim" and g.loc["D002", "match_type"] == "none" and g.loc["D002", "indikit_id"] == ""


def test_validation_rejects_bad_decisions(tmp_path: Path):
    df = pd.DataFrame([{"doc_id": "d", "doc_indicator_id": "D1", "reviewer_decision": "maybe"}, {"doc_id": "d", "doc_indicator_id": "D2", "reviewer_decision": "other_id", "correct_indikit_id": ""},
                       {"doc_id": "d", "doc_indicator_id": "D3", "reviewer_decision": "other_id", "correct_indikit_id": "1"}, {"doc_id": "d", "doc_indicator_id": "D4", "reviewer_decision": ""}, {"doc_id": "d", "doc_indicator_id": "D5", "reviewer_decision": "Accept"}])
    dec, errors = validate_decisions(df.fillna(""), METADATA)
    assert len(errors) == 3 and len(dec) == 1 and dec.iloc[0]["reviewer_decision"] == "accept"
    latest = _latest(tmp_path)
    (tmp_path / "bad.csv").write_text("doc_id,doc_indicator_id,reviewer_decision\nd1,D001,maybe\n")
    with pytest.raises(ValueError, match="unknown decision"):
        import_review(tmp_path / "bad.csv", latest, METADATA)
    assert read_review_file(tmp_path / "bad.csv").shape[0] == 1
