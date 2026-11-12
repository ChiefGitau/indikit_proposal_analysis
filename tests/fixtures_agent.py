"""Shared synthetic set-up for the Phase 3 tests: two synthetic documents (docx + pdf) built into a tmp knowledge base
with the hashing embedder, a tiny metadata library, a lexical retriever, pair scores and a manifest. No model
downloads, no network, no real documents."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.agent.rerank import LexicalReranker
from pipeline.agent.retrieve import RetrievalConfig, Retriever, score_pairs
from pipeline.metadata.models import Aspect, IndicatorMetadata
from pipeline.preprocess.embeddings import EmbeddingCache, HashingEmbedder
from pipeline.preprocess.knowledge_base import build_kb
from pipeline.preprocess.registry import discover_documents
from tests.fixtures_docs import make_docx, make_pdf

HANDWASH = IndicatorMetadata(indikit_id="5138", name="Handwashing Facilities With Soap and Water", aspect=[Aspect(section="Sectoral", sector="Water, Sanitation and Hygiene", sub_sector="Hygiene")],
                             kpi_wording_en="% of households with handwashing facilities with soap and water at home", levels=["Output"], search_terms=["handwashing facilities", "soap and water", "handwashing station"],
                             knowledge="Counts households whose handwashing station has both soap and water at the time of the visit.", on_site=True, quantity="percentage")
CHLORINE = IndicatorMetadata(indikit_id="5330", name="Drinking Water Quality (residual chlorine)", aspect=[Aspect(section="Sectoral", sector="Water, Sanitation and Hygiene", sub_sector="Water")],
                             kpi_wording_en="% of water samples with free residual chlorine between 0.2 and 0.5 mg/l", search_terms=["free residual chlorine", "frc"], knowledge="Share of samples in the FRC range.", on_site=True)
LATRINE = IndicatorMetadata(indikit_id="5140", name="Latrine Construction", aspect=[Aspect(section="Sectoral", sector="Water, Sanitation and Hygiene", sub_sector="Sanitation")],
                            kpi_wording_en="number of latrines constructed or rehabilitated", search_terms=["latrines constructed", "latrine"], knowledge="Counts latrines.", on_site=False)
UNRELATED = IndicatorMetadata(indikit_id="9000", name="Zebra migration distance", aspect=[Aspect(section="Sectoral", sector="Wildlife")], kpi_wording_en="average kilometres migrated by zebra herds",
                              search_terms=["zebra", "migration"], knowledge="Zebras.", on_site=True)
METADATA = {m.indikit_id: m for m in (HANDWASH, CHLORINE, LATRINE, UNRELATED)}


@dataclass
class AgentFixture:
    root: Path
    kb_dir: Path
    refs: list
    manifest: dict
    retriever: Retriever
    embedder: HashingEmbedder
    pair_scores: pd.DataFrame
    docx_id: str
    pdf_id: str
    metadata: dict

    def retrieve(self, doc_id, indicator, **kw):
        return self.retriever.retrieve(doc_id, indicator, **kw)


def build_agent_fixture(tmp_path: Path) -> AgentFixture:
    root = tmp_path / "proposals" / "africa" / "testland" / "small"
    root.mkdir(parents=True)
    make_docx(root / "a.docx")
    make_pdf(root / "b.pdf")
    refs = discover_documents([("proposals", tmp_path / "proposals")])
    emb = HashingEmbedder(512)
    kb_dir = tmp_path / "kb"
    manifest = {}
    for r in refs:
        row = build_kb(r, emb, EmbeddingCache(None), kb_dir=kb_dir)
        manifest[row.doc_id] = row
    retriever = Retriever(RetrievalConfig(embedder=emb.name, reranker="lexical"), embedder=emb, reranker=LexicalReranker(), cache=EmbeddingCache(None), kb_dir=kb_dir, queries_dir=tmp_path / "q")
    rows, _ = score_pairs(retriever, [r.doc_id for r in refs], list(METADATA.values()), use_rerank=True, progress=False)
    scores = pd.DataFrame(rows)
    docx_id = next(r.doc_id for r in refs if r.file_type == "docx")
    pdf_id = next(r.doc_id for r in refs if r.file_type == "pdf")
    return AgentFixture(root=tmp_path, kb_dir=kb_dir, refs=refs, manifest=manifest, retriever=retriever, embedder=emb, pair_scores=scores, docx_id=docx_id, pdf_id=pdf_id, metadata=METADATA)


def answer_from_evidence(system: str, user: str, schema, model: str):
    """Scripted 'model': finds the evidence item that contains the indicator wording and copies text from it. Used by the
    mock provider so that answers are supported by real evidence quotes."""
    import re

    if schema.__name__ == "ExtractionAnswer":
        inds = []
        section = user.split("## Rubric", 1)[0]
        for line in section.splitlines():
            if re.match(r"^\s*(?:#|%|Number of|Percent)", line) and not line.startswith("##"):
                inds.append({"indicator_text": line.strip(), "is_indicator": True, "level": "output", "baseline": None, "target": None, "unit": None, "means_of_verification": None, "result_context": None, "source_quote": line.strip()})
        return {"indicators": inds}
    if schema.__name__ == "ReverseCheckAnswer":
        return {"matches": [], "reason": "none of the candidates measures this"}
    m = re.search(r"IndiKit ID: (\d+)", system)
    iid = m.group(1) if m else "0"
    wording = {"5138": "handwashing facilities with soap and water", "5330": "free residual chlorine", "5140": "latrines constructed"}.get(iid)
    items = re.findall(r"\[(E\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[E\d+\]|\n\n## 3\.)", user, flags=re.DOTALL)
    for label, text in items:
        if wording and wording in text.lower():
            row_line = next((l for l in text.splitlines() if "Row:" in l and wording in l.lower()), text.splitlines()[-1])
            cell = next((c.split(":", 1)[1].strip() for c in row_line.replace("Row: ", "").split(" | ") if c.lower().startswith("indicator") and wording in c.lower()), None)
            doc_text = cell or next((l.strip() for l in text.splitlines() if wording in l.lower()), text[:80])
            bl = re.search(r"Baseline: ([^|]+)", row_line)
            tg = re.search(r"Target: ([^|]+)", row_line)
            return {"indikit_id": iid, "disclosure": "present", "matches": [{"document_indicator_text": doc_text, "match_type": "verbatim" if iid == "5138" else "close", "level": "outcome",
                    "baseline": bl.group(1).strip() if bl else None, "target": tg.group(1).strip() if tg else None, "unit": "%" if "%" in doc_text else None, "evidence_label": label,
                    "evidence_quote": doc_text, "reason": "same measure"}]}
    return {"indikit_id": iid, "disclosure": "absent", "matches": []}
