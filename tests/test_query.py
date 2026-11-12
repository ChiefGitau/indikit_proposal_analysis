from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.agent.query import QueryStore, build_query_text, precompute_queries, queries_path
from pipeline.metadata.models import Aspect, IndicatorMetadata
from pipeline.preprocess.embeddings import EmbeddingCache, HashingEmbedder


def ind(**kw) -> IndicatorMetadata:
    base = dict(indikit_id="5138", name="Handwashing Facilities With Soap and Water", aspect=[Aspect(sector="WASH", sub_sector="Hygiene")],
                kpi_wording_en="% of [specify the target group] households with handwashing facilities with soap and water at home", kpi_wording_fr="% de ménages ayant un lieu de lavage des mains",
                search_terms=["handwashing facilities with soap and water", "soap and water", "w8", "handwashing station"])
    base.update(kw)
    return IndicatorMetadata(**base)


def test_query_text_variants():
    full = build_query_text(ind(), use_search_terms=True)
    assert full.splitlines()[0] == "Handwashing Facilities With Soap and Water"
    assert "[specify" not in full and "% of households with handwashing facilities with soap and water at home" in full
    assert "w8; handwashing station" in full.replace("soap and water; ", "") or "w8" in full
    assert "% de ménages" in full
    no_terms = build_query_text(ind(), use_search_terms=False)
    assert "w8" not in no_terms and "% de ménages" in no_terms
    assert build_query_text(ind(kpi_wording_en="", kpi_wording_fr=None, search_terms=[])) == "Handwashing Facilities With Soap and Water"


def test_precompute_and_store(tmp_path: Path):
    emb = HashingEmbedder(64)
    recs = [ind(), ind(indikit_id="5330", name="Drinking Water Quality", search_terms=["frc"])]
    path = precompute_queries(recs, emb, True, tmp_path, cache=EmbeddingCache(None))
    assert path == queries_path(emb.name, True, tmp_path) and path.exists() and path.with_suffix(".json").exists()
    store = QueryStore(emb, True, tmp_path)
    v = store.vector(recs[0])
    assert v.shape == (64,) and np.isclose(np.linalg.norm(v), 1.0)
    assert np.allclose(v, emb.embed([build_query_text(recs[0], True)])[0])
    # unknown indicator: computed on demand
    other = ind(indikit_id="9999", name="Other")
    assert QueryStore(emb, False, tmp_path).vector(other).shape == (64,)
