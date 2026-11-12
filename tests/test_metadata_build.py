"""Merge logic of the metadata module on small synthetic fixtures (docs/phase_1.md Task 3)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.metadata import build_metadata as bm
from pipeline.metadata.models import Aspect, IndicatorMetadata, Quantity


# --------------------------------------------------------------------------- #
# helpers / fixtures
# --------------------------------------------------------------------------- #
EXPORT_HEADER = "Section,Sector,Sub-Sector,Indicator Level,Indicator Name,Indicator Wording - ENG,Indicator Wording - FRA,Guidance,Unique ID,What is its Purpose?,,,\n"


def write_export(path: Path, rows: list[dict]) -> Path:
    """Write an export-like CSV in cp1252 (what the real file really is), with trailing empty columns."""
    lines = [EXPORT_HEADER]
    for r in rows:
        vals = [r.get(k, "") for k in ("Section", "Sector", "Sub-Sector", "Indicator Level", "Indicator Name", "Indicator Wording - ENG", "Indicator Wording - FRA", "Guidance", "Unique ID", "What is its Purpose?")]
        lines.append(",".join('"' + v.replace('"', '""') + '"' for v in vals) + ",,,\n")
    path.write_text("".join(lines), encoding="cp1252")
    return path


def write_scraped(path: Path, records: dict, levels: dict | None = None) -> Path:
    path.write_text(json.dumps({"scraped_at": "2026-07-06T09:14:24", "records": records, "live_ids": sorted(records), "levels": levels or {}, "levels_at": None}, ensure_ascii=False), encoding="utf-8")
    return path


def site_record(uid: str, **kw) -> dict:
    base = {"ok": True, "url": f"https://zoa.indikit.net/indicator/{uid}-x", "unique_id": uid, "name": f"Name {uid}", "section": "Sectoral", "sector": "Water, Sanitation and Hygiene", "sub_sector": "Water", "wording_en": f"% of households with thing {uid}", "wording_fr": f"% de ménages avec {uid}", "purpose": "Purpose text.", "how_to_collect": "Collect it. Then compute.", "disaggregate": "by sex", "important_comments": "Comment one. Comment two."}
    base.update(kw)
    return base


@pytest.fixture
def fixture_dir(tmp_path: Path) -> dict:
    export = write_export(
        tmp_path / "export.csv",
        [
            {"Section": "Sectoral", "Sector": "Water, Sanitation and Hygiene", "Sub-Sector": "Water", "Indicator Level": "Output | Outcome", "Indicator Name": "Shared Indicator", "Indicator Wording - ENG": "% of households with thing 1", "Indicator Wording - FRA": "% de ménages avec 1", "Guidance": "https://zoa.indikit.net/indicator/1-x", "Unique ID": "1", "What is its Purpose?": "Purpose text."},
            {"Section": "Sectoral", "Sector": "Water, Sanitation and Hygiene", "Sub-Sector": "Hygiene", "Indicator Level": "Output | Outcome", "Indicator Name": "Shared Indicator", "Indicator Wording - ENG": "% of households with thing 1", "Indicator Wording - FRA": "% de ménages avec 1", "Guidance": "https://zoa.indikit.net/indicator/1-x", "Unique ID": "1", "What is its Purpose?": "Purpose text."},
            {"Section": "Sectoral", "Sector": "Shelter", "Sub-Sector": "", "Indicator Level": "Impact", "Indicator Name": "Export Only One", "Indicator Wording - ENG": "number of people with shelter", "Indicator Wording - FRA": "nombre de personnes", "Guidance": "https://zoa.indikit.net/indicator/2-export-only", "Unique ID": "2", "What is its Purpose?": "Export purpose ’quoted’."},
            {"Section": "Sectoral", "Sector": "Food Security and Livelihoods", "Sub-Sector": "Agriculture", "Indicator Level": "", "Indicator Name": "Conflicting One", "Indicator Wording - ENG": "export wording", "Indicator Wording - FRA": "", "Guidance": "https://zoa.indikit.net/indicator/3-x", "Unique ID": "3", "What is its Purpose?": "Same purpose, different punctuation!"},
        ],
    )
    scraped = write_scraped(
        tmp_path / "scraped.json",
        {
            "1": site_record("1", name="Shared Indicator", wording_en="% of households with thing 1", wording_fr="% de ménages avec 1"),
            "3": site_record("3", name="Conflicting One", section="Sectoral", sector="Food Security and Livelihoods", sub_sector="Agriculture", wording_en="site wording", wording_fr="", purpose="Same purpose different punctuation"),
            "4": site_record("4", name="Site Only Four", sector="Shelter", sub_sector="", wording_en="average score of shelters"),
        },
        levels={"1": ["Outcome"], "3": ["Output"], "4": ["Outcome", "Impact"]},
    )
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text('"1":\n  name: Shared Indicator\n  search_terms_add: ["W8"]\n"3":\n  name: Totally Different Name\n  search_terms_add: ["nope"]\n', encoding="utf-8")
    return {"export": export, "scraped": scraped, "overrides": overrides, "tmp": tmp_path}


def run_build(fx: dict, out_name: str = "meta.jsonl", **kw):
    tmp = fx["tmp"]
    return bm.build(
        export_csv=fx["export"],
        scraped_json=fx["scraped"],
        site_csv=None,
        supplement_json=None,
        overrides_yaml=fx["overrides"],
        out_jsonl=tmp / out_name,
        reference_csv=tmp / "reference_library.csv",
        report_md=tmp / "build_report.md",
        conflicts_csv=tmp / "merge_conflicts.csv",
        seed_urls=["https://zoa.indikit.net/indicator/9001-pip-of-households-who-plan"],
        **kw,
    )


# --------------------------------------------------------------------------- #
# text cleaning and field rules
# --------------------------------------------------------------------------- #
def test_fix_mojibake_repairs_cp1252_read_as_latin1():
    raw = "quantit\xe9 de l\x92unit\xe9 \x96 \x93ok\x94\x85"
    assert bm.fix_mojibake(raw) == "quantité de l’unité – “ok”…"
    assert bm.clean_text("  a \n b\xa0c ") == "a b c"
    assert bm.clean_text(None) == "" and bm.clean_text(float("nan")) == ""


def test_normalise_for_compare_ignores_punctuation_and_case():
    assert bm.normalise_for_compare("Same purpose, different punctuation!") == bm.normalise_for_compare("same purpose different punctuation")
    assert bm.normalise_for_compare("l’unité") == bm.normalise_for_compare("l'unité")
    assert bm.normalise_for_compare("export wording") != bm.normalise_for_compare("site wording")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Output | Outcome", ["Output", "Outcome"]),
        ("Outcome | Impact", ["Outcome", "Impact"]),
        (["Impact", "Outcome"], ["Outcome", "Impact"]),
        ("organisational level", ["Organisational level"]),
        ("Organizational level", ["Organisational level"]),
        ("(missing)", []),
        ("", []),
        (None, []),
        ("Output | Output", ["Output"]),
    ],
)
def test_normalise_levels(raw, expected):
    assert bm.normalise_levels(raw) == expected


@pytest.mark.parametrize(
    "wording, name, expected",
    [
        ("% of households with handwashing facilities", "Handwashing", Quantity.percentage),
        ("percentage of farmers adopting practices", "", Quantity.percentage),
        ("proportion of children under five", "", Quantity.percentage),
        ("# of people trained", "", Quantity.number),
        ("number of people with access to safe water", "", Quantity.number),
        ("total amount of CO2 equivalent (in tonnes) reduced", "Amount of Greenhouse Gas Emissions Saved", Quantity.number),
        ("total USD value of cash transferred", "Total CVA Value Distributed", Quantity.number),
        ("average number of different food groups consumed", "Individual Dietary Diversity Score", Quantity.score),
        ("average Food Consumption Score of households", "Food Consumption Score", Quantity.score),
        ("life satisfaction index value", "Life Satisfaction Index", Quantity.score),
        ("ratio of girls to boys enrolled", "Gender Parity", Quantity.ratio),
        ("average yield per hectare", "Agricultural Productivity", Quantity.number),
        ("whether the organisation has a policy in place", "", Quantity.yes_no),
        ("existence of a functional water committee", "", Quantity.yes_no),
        ("level of satisfaction with the service", "", Quantity.other),
        ("", "", Quantity.other),
        ("", "% of households with latrines", Quantity.percentage),
        ("farmers report increased yields (rate of change)", "", Quantity.percentage),
        ("", "PIP: # of households who are implementing their PIP plans", Quantity.number),
        ("the average duration of a period in the past 12 months when households had enough food", "Months of Adequate Household Food Provisioning", Quantity.number),
        ("Food Insecurity Experience Scale", "Food Insecurity Experience Scale (FIES)", Quantity.score),
        ("hectares of land managed using environmentally sustainable practices", "Area of Land Under Improved Management", Quantity.number),
        ("# land certificates or title deeds issued through the support of the intervention", "", Quantity.number),
        ("a platform for holding a regular dialogue is institutionalized between public and private sector actors", "Public-Private Platforms", Quantity.yes_no),
        ("gender and age marks scored by the project", "Gender and Age marks", Quantity.score),
        ("increased budgetary commitment by [specify the stakeholder] to address the advocated issue", "Budgetary Commitment", Quantity.other),
    ],
)
def test_derive_quantity(wording, name, expected):
    assert bm.derive_quantity(wording, name) is expected


def test_split_indikit_wording():
    block = "English\nFrench\nNumber of people with access to \r\nsafe water\nNombre de personnes ayant accès à l'eau potable"
    en, fr = bm.split_indikit_wording(block)
    assert en == "Number of people with access to safe water"
    assert fr.startswith("Nombre de personnes")
    assert bm.split_indikit_wording("Just one line") == ("Just one line", "")
    assert bm.split_indikit_wording(None) == ("", "")


def test_name_from_slug_and_id_from_url():
    url = "https://zoa.indikit.net/indicator/5889-pip-of-households-who-are-implementing-their-pip-plans"
    assert bm.id_from_url(url) == "5889"
    assert bm.name_from_slug(url) == "PIP of households who are implementing their PIP plans"


# --------------------------------------------------------------------------- #
# export collapse
# --------------------------------------------------------------------------- #
def test_collapse_export_rows_merges_repeated_ids(fixture_dir):
    df = bm.read_export(fixture_dir["export"])
    assert list(df.columns)[-1] == "What is its Purpose?"  # empty trailing columns dropped
    by_id, issues = bm.collapse_export_rows(df)
    assert list(by_id) == ["1", "2", "3"]
    assert by_id["1"]["rows"] == 2
    assert by_id["1"]["aspects"] == [Aspect(section="Sectoral", sector="Water, Sanitation and Hygiene", sub_sector="Water"), Aspect(section="Sectoral", sector="Water, Sanitation and Hygiene", sub_sector="Hygiene")]
    assert by_id["1"]["levels"] == ["Output", "Outcome"]
    assert by_id["2"]["purpose"] == "Export purpose ’quoted’."  # mojibake repaired
    assert issues == []


def test_collapse_export_rows_logs_inconsistent_repeats():
    df = pd.DataFrame(
        [
            {"Section": "S", "Sector": "A", "Sub-Sector": "x", "Indicator Level": "Output", "Indicator Name": "One", "Indicator Wording - ENG": "w", "Indicator Wording - FRA": "", "Guidance": "", "Unique ID": "7", "What is its Purpose?": "p"},
            {"Section": "S", "Sector": "A", "Sub-Sector": "y", "Indicator Level": "Outcome", "Indicator Name": "One changed", "Indicator Wording - ENG": "w", "Indicator Wording - FRA": "", "Guidance": "", "Unique ID": "7", "What is its Purpose?": "p"},
        ]
    )
    by_id, issues = bm.collapse_export_rows(df)
    assert by_id["7"]["levels"] == ["Output", "Outcome"]
    assert issues == [{"indikit_id": "7", "field": "name", "first_value": "One", "other_value": "One changed"}]


# --------------------------------------------------------------------------- #
# merge rules
# --------------------------------------------------------------------------- #
def test_merge_conflict_rule_site_wins_and_is_recorded(fixture_dir):
    res = run_build(fixture_dir)
    rec = {r.indikit_id: r for r in res.records}["3"]
    assert rec.kpi_wording_en == "site wording"
    assert rec.field_sources["kpi_wording_en"] == "site"
    wording_conflicts = [c for c in res.conflicts if c["indikit_id"] == "3" and c["field"] == "kpi_wording_en"]
    assert wording_conflicts == [{"indikit_id": "3", "field": "kpi_wording_en", "export_value": "export wording", "site_value": "site wording", "resolved": "site"}]
    # purpose differs only in punctuation -> no conflict
    assert not [c for c in res.conflicts if c["indikit_id"] == "3" and c["field"] == "purpose"]
    # levels: export had none, site supplies Output
    assert rec.levels == ["Output"] and rec.field_sources["levels"] == "site"


def test_merge_export_only_and_site_only_flags(fixture_dir):
    res = run_build(fixture_dir)
    by = {r.indikit_id: r for r in res.records}
    assert by["2"].in_export and not by["2"].on_site
    assert by["2"].url == "https://zoa.indikit.net/indicator/2-export-only" and by["2"].field_sources["url"] == "export"
    assert by["2"].guidance == by["2"].url
    assert by["2"].levels == ["Impact"] and by["2"].quantity is Quantity.number
    assert by["4"].on_site and not by["4"].in_export
    assert by["4"].levels == ["Outcome", "Impact"] and by["4"].quantity is Quantity.score
    assert by["1"].on_site and by["1"].in_export
    assert len(by["1"].aspect) == 2  # repeated export rows collapsed, nothing added by the identical site aspect
    assert by["1"].levels == ["Output", "Outcome"]
    assert by["1"].topic == "by sex" and by["1"].how_to_collect.startswith("Collect it.")


def test_seeded_records_when_no_fetch(fixture_dir):
    res = run_build(fixture_dir)
    seed = {r.indikit_id: r for r in res.records}["9001"]
    assert seed.needs_fetch and not seed.on_site and not seed.in_export
    assert seed.name == "PIP of households who plan"
    assert seed.kpi_wording_en == ""
    assert seed.aspect == [Aspect(section="Sectoral", sector="Food Security and Livelihoods", sub_sector="")]
    assert seed.field_sources["name"] == "seed" and seed.field_sources["aspect"] == "seed"
    assert res.stats.seed_only == 1


def test_build_is_deterministic_and_writes_outputs(fixture_dir):
    tmp = fixture_dir["tmp"]
    res1 = run_build(fixture_dir, out_name="a.jsonl")
    res2 = run_build(fixture_dir, out_name="b.jsonl")
    assert (tmp / "a.jsonl").read_bytes() == (tmp / "b.jsonl").read_bytes()
    ids = [r.indikit_id for r in res1.records]
    assert ids == ["1", "2", "3", "4", "9001"] and len(set(ids)) == len(ids)
    for r in res1.records:
        assert r.name
        IndicatorMetadata.model_validate(r.model_dump())
    # outputs
    ref = pd.read_csv(tmp / "reference_library.csv", dtype=str).fillna("")
    assert {"indicator_id", "indicator_text", "sector", "level"}.issubset(ref.columns)
    assert ref.loc[ref["indicator_id"] == "1", "level"].iloc[0] == "Output | Outcome"
    assert ref.loc[ref["indicator_id"] == "9001", "indicator_text"].iloc[0] == "PIP of households who plan"  # name fallback
    conflicts = pd.read_csv(tmp / "merge_conflicts.csv", dtype=str)
    assert list(conflicts.columns) == ["indikit_id", "field", "export_value", "site_value", "resolved"]
    report = (tmp / "build_report.md").read_text(encoding="utf-8")
    assert "Export-only (not on site) | 1" in report and "Site-only (not in export) | 1" in report
    assert "9001" in report  # seed listed under missing wording
    # overrides: 1 applied, 3 skipped on name mismatch
    assert res1.stats.overrides_applied == ["1"]
    assert res1.stats.overrides_skipped and res1.stats.overrides_skipped[0][0] == "3"
    assert "w8" in {r.indikit_id: r for r in res1.records}["1"].search_terms
    # round trip
    loaded = bm.read_jsonl(tmp / "a.jsonl")
    assert [r.indikit_id for r in loaded] == ids and loaded[0] == res1.records[0]


def test_read_jsonl_missing_file_points_to_cli(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-metadata"):
        bm.read_jsonl(tmp_path / "nope.jsonl")


def test_model_rejects_empty_name_and_unknown_level():
    with pytest.raises(ValueError):
        IndicatorMetadata(indikit_id="1", name="  ")
    with pytest.raises(ValueError):
        IndicatorMetadata(indikit_id="1", name="x", levels=["Activity"])
    rec = IndicatorMetadata(indikit_id="1", name="x", levels=["Impact", "Output", "Impact"])
    assert rec.levels == ["Output", "Impact"]
