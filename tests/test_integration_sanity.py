"""Sanity checks on the real template documents with the real embedder (docs/phase_2.md Task 8).

Run with ``pytest -m integration``; skipped automatically when the knowledge bases or the cached models are missing.
"""
from __future__ import annotations

import pytest

from pipeline import config

pytestmark = pytest.mark.integration

CHECKS = [
    ("Tigray Emergency Response", "5138", ["soap and water at a handwashing station"]),
    ("Tigray Emergency Response", "5330", ["free residual chlorine", "frc"]),
    ("ZOA-we-are-able-full-proposal-v6", "5620", ["food consumption score"]),
    ("ZOA-we-are-able-full-proposal-v6", "5643", ["secure tenure rights to land", "tenure rights"]),
]
EXPECTED_NAMES = {"5138": "Handwashing Facilities With Soap and Water", "5330": "Drinking Water Quality (residual chlorine)", "5620": "Food Consumption Score", "5643": "Land Tenure Security"}


@pytest.fixture(scope="module")
def setup():
    from pipeline.metadata.build_metadata import read_jsonl
    from pipeline.preprocess.registry import read_manifest

    manifest = read_manifest()
    template = {r.relative_path: r for r in manifest.values() if r.region == "template" and r.parse_status != "failed"}
    if len(template) < 2:
        pytest.skip("template knowledge bases not built (python -m pipeline.run build-kb --all)")
    try:
        from pipeline.agent.retrieve import Retriever, RetrievalConfig
        from pipeline.preprocess.embeddings import EmbeddingCache

        retriever = Retriever(RetrievalConfig(), cache=EmbeddingCache())
        retriever.embedder.embed(["probe"])
    except Exception as exc:  # model not cached / no backend
        pytest.skip(f"real embedder unavailable: {exc}")
    meta = {r.indikit_id: r for r in read_jsonl()}
    return template, retriever, meta


@pytest.mark.parametrize("doc_fragment, indikit_id, needles", CHECKS)
def test_expected_passage_in_top_k(setup, doc_fragment, indikit_id, needles):
    template, retriever, meta = setup
    ind = meta[indikit_id]
    assert EXPECTED_NAMES[indikit_id].lower() in ind.name.lower(), f"metadata name for {indikit_id} is '{ind.name}'"
    doc = next(r for p, r in template.items() if doc_fragment in p)
    res = retriever.retrieve(doc.doc_id, ind, top_k=config.RETRIEVAL_TOP_K)
    texts = [e.display_text.lower() for e in res.evidence]
    assert any(any(n in t for n in needles) for t in texts), "\n---\n".join(e.display_text[:300] for e in res.evidence)


def test_reference_detection_on_template_documents(setup):
    from pipeline.preprocess.references import load_references

    template, _, _ = setup
    bha = next(r for p, r in template.items() if "Tigray Emergency Response" in p)
    waa = next(r for p, r in template.items() if "we-are-able" in p)
    assert any("indicator tracking table" in r.referenced_document.lower() for r in load_references(bha.doc_id))
    assert any("indicator database" in r.referenced_document.lower() for r in load_references(waa.doc_id))
