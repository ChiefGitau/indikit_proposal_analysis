"""Checks against the real IndiKit files (skipped when they are not present)."""
from __future__ import annotations

from collections import Counter

import pytest

from pipeline import config
from pipeline.metadata import build_metadata as bm

pytestmark = pytest.mark.skipif(not (config.EXPORT_CSV.exists() and config.SCRAPED_JSON.exists()), reason="raw IndiKit files not present")


@pytest.fixture(scope="module")
def real_build(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("real")
    return bm.build(out_jsonl=tmp / "meta.jsonl", reference_csv=tmp / "ref.csv", report_md=tmp / "report.md", conflicts_csv=tmp / "conflicts.csv"), tmp


def test_counts_and_unique_ids(real_build):
    res, _ = real_build
    ids = [r.indikit_id for r in res.records]
    assert len(ids) == len(set(ids))
    assert res.stats.export_unique_ids == 344
    assert res.stats.export_rows == 386
    assert res.stats.site_records == 303
    assert res.stats.export_only == 41
    assert res.stats.site_only + res.stats.seed_only == 7  # PIP indicators, fetched or seeded
    assert len(ids) == 344 + 7


def test_required_fields(real_build):
    res, _ = real_build
    for r in res.records:
        assert r.name
        assert r.aspect and r.aspect[0].sector
        assert r.search_terms
    missing_wording = [r.indikit_id for r in res.records if not r.kpi_wording_en]
    assert set(missing_wording) == set(res.stats.missing_wording_ids)
    # every record without wording is listed in the report
    for uid in missing_wording:
        assert uid in res.report
    assert set(res.stats.per_sector) >= {"Food Security and Livelihoods", "Water, Sanitation and Hygiene", "Shelter"}
    assert Counter(r.quantity.value for r in res.records)["percentage"] > 100


def test_overrides_from_yaml_are_reflected(real_build):
    res, _ = real_build
    by = {r.indikit_id: r for r in res.records}
    assert "w8" in by["5138"].search_terms
    assert "frc" in by["5330"].search_terms
    assert "k1" in by["5639"].search_terms
    assert "fcs > 35" in by["5620"].search_terms
    assert "plan intégré du paysan" in by["5889"].search_terms
    assert not res.stats.overrides_skipped


def test_committed_jsonl_is_up_to_date(real_build):
    """The tracked indikit_metadata.jsonl must equal a fresh build (determinism + no stale file)."""
    res, tmp = real_build
    if not config.METADATA_JSONL.exists():
        pytest.skip("indikit_metadata.jsonl not built yet")
    assert (tmp / "meta.jsonl").read_bytes() == config.METADATA_JSONL.read_bytes()
