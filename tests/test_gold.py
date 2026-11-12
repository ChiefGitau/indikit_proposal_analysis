"""Gold-set loader / validator."""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import config
from pipeline.eval import gold

HEADER = "doc_indicator_id,indicator_text,level,baseline,target,unit,location,indikit_id,match_type\n"
IDS = {"5138", "5620", "5330", "5179"}


def test_example_template_validates():
    rows, errors = gold.validate_gold_file(config.GOLD_DIR / "_example.csv", IDS)
    assert errors == [] and len(rows) == 4
    assert rows[2].indikit_id is None and rows[2].match_type == "none"
    assert rows[0].baseline == "12%" and rows[3].baseline is None


def test_whole_gold_dir_validates():
    report = gold.validate_gold(config.GOLD_DIR, metadata_ids=None)
    assert report.ok, report.errors
    assert "index.csv" in report.files_checked and "_example.csv" in report.files_checked
    assert report.rows == 4
    assert gold.load_gold(config.GOLD_DIR) == []  # index is empty: no real labels yet


def test_broken_fixture_gives_clear_messages(tmp_path: Path):
    bad = tmp_path / "doc1.csv"
    bad.write_text(
        HEADER
        + "D01,ok row,outcome,1,2,%,p1,5138,verbatim\n"
        + "D01,duplicate id,output,,,,p1,5620,close\n"
        + "D02,bad level,result,,,,p1,5620,close\n"
        + "D03,none with id,output,,,,p1,5620,none\n"
        + "D04,id without match,output,,,,p1,,close\n"
        + "D05,unknown id,output,,,,p1,1234,verbatim\n"
        + "D06,bad match type,output,,,,p1,5138,exact\n"
        + ",empty id,output,,,,p1,5138,verbatim\n",
        encoding="utf-8",
    )
    rows, errors = gold.validate_gold_file(bad, IDS)
    assert len(rows) == 3  # D01 twice + D05 parsed (D05 is flagged separately)
    joined = "\n".join(errors)
    assert "line 3: duplicate doc_indicator_id 'D01'" in joined
    assert "line 4" in joined and "level" in joined
    assert "line 5" in joined and "match_type must be 'none' exactly when indikit_id is empty" in joined
    assert "line 6" in joined
    assert "line 7: indikit_id '1234' not in the metadata library" in joined
    assert "line 8" in joined and "match_type" in joined
    assert "line 9" in joined and "doc_indicator_id" in joined
    assert len(errors) == 7

    missing_cols = tmp_path / "doc2.csv"
    missing_cols.write_text("doc_indicator_id,indicator_text\nD1,x\n", encoding="utf-8")
    _, errors = gold.validate_gold_file(missing_cols, IDS)
    assert errors == ["doc2.csv: missing columns ['level', 'baseline', 'target', 'unit', 'location', 'indikit_id', 'match_type']"]
    assert gold.validate_gold_file(tmp_path / "nope.csv", IDS)[1] == ["nope.csv: file not found"]


def test_index_validation_and_load_gold(tmp_path: Path):
    (tmp_path / "index.csv").write_text(
        "doc_id,source_file,language,has_logframe,indicators_in_other_document,labelled_by,verified,split\n"
        "docA,africa/x.docx,en,true,,HM,false,dev\n"
        "docB,africa/y.pdf,fr,false,Annex C - Logical Framework,HM,true,holdout\n"
        "docA,africa/z.pdf,en,true,,HM,true,test\n",
        encoding="utf-8",
    )
    (tmp_path / "docA.csv").write_text(HEADER + "D01,ok,outcome,,,,p1,5138,verbatim\n", encoding="utf-8")
    entries, errors = gold.load_index(tmp_path / "index.csv")
    assert [e.doc_id for e in entries] == ["docA", "docA"]
    assert any("line 3" in e and "split" in e for e in errors)
    assert any("duplicate doc_id 'docA'" in e for e in errors)
    report = gold.validate_gold(tmp_path, IDS)
    assert not report.ok and report.rows == 2
    with pytest.raises(gold.GoldValidationError) as excinfo:
        gold.load_gold(tmp_path, IDS)
    assert "split" in str(excinfo.value)

    # a clean index loads
    (tmp_path / "index.csv").write_text(
        "doc_id,source_file,language,has_logframe,indicators_in_other_document,labelled_by,verified,split\n"
        "docA,africa/x.docx,en,true,,HM,false,dev\n",
        encoding="utf-8",
    )
    docs = gold.load_gold(tmp_path, IDS)
    assert len(docs) == 1 and docs[0].entry.split == "dev" and docs[0].rows[0].indikit_id == "5138"


def test_cli_validate_gold(capsys, tmp_path: Path):
    from pipeline.run import main

    assert main(["validate-gold", "--no-metadata-check"]) == 0
    assert "gold set OK" in capsys.readouterr().out
    broken = tmp_path / "x.csv"
    broken.write_text(HEADER + "D01,x,output,,,,p1,5138,none\n", encoding="utf-8")
    assert main(["validate-gold", "--file", str(broken), "--no-metadata-check"]) == 1
    out = capsys.readouterr().out
    assert "1 problem(s)" in out and "match_type must be 'none'" in out
