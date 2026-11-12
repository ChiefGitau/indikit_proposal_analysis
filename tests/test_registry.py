from __future__ import annotations

from pathlib import Path

from pipeline.preprocess import registry
from pipeline.preprocess.models import ManifestRow
from pipeline.preprocess.textnorm import detect_language, fix_hyphenation, normalise_whitespace, slugify
from tests.fixtures_docs import make_docx, make_xlsx


def test_discover_documents_ids_and_folders(tmp_path: Path):
    root = tmp_path / "proposals"
    (root / "africa" / "burundi" / "large").mkdir(parents=True)
    (root / "Asia" / "yemen" / "small").mkdir(parents=True)
    make_docx(root / "africa" / "burundi" / "large" / "Proposal.docx")
    make_docx(root / "Asia" / "yemen" / "small" / "Proposal.docx")  # same file name in another country
    make_xlsx(root / "africa" / "burundi" / "large" / "logframe.xlsx")
    (root / "africa" / "burundi" / "large" / "~$Proposal.docx").write_bytes(b"lock")
    (root / "africa" / "burundi" / "large" / "notes.txt").write_text("x")
    refs = registry.discover_documents([("proposals", root)])
    assert [r.relative_path for r in refs] == ["proposals/Asia/yemen/small/Proposal.docx", "proposals/africa/burundi/large/Proposal.docx", "proposals/africa/burundi/large/logframe.xlsx"]
    assert len({r.doc_id for r in refs}) == 3
    assert refs[1].region == "africa" and refs[1].country == "burundi" and refs[1].file_type == "docx"
    assert refs[1].doc_id.startswith("proposals-africa-burundi-large-proposal-") and len(refs[1].doc_id.rsplit("-", 1)[1]) == 8
    # deterministic: same inputs -> same ids
    assert [r.doc_id for r in registry.discover_documents([("proposals", root)])] == [r.doc_id for r in refs]
    assert registry.find_ref(refs[2].doc_id, refs) is refs[2]
    assert registry.find_ref(str(refs[2].path), refs) is refs[2]


def test_manifest_roundtrip_and_incremental(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    row = ManifestRow(doc_id="d-1", relative_path="proposals/x.docx", region="africa", country="x", file_type="docx", sha256="abc", pipeline_version="kb-v1", parse_status="ok", n_tables=2, indicators_elsewhere=True)
    registry.upsert_manifest(row, path)
    registry.upsert_manifest(row.model_copy(update={"doc_id": "d-2", "relative_path": "proposals/a.pdf", "file_type": "pdf", "parse_status": "failed", "error": "boom"}), path)
    rows = registry.read_manifest(path)
    assert set(rows) == {"d-1", "d-2"} and rows["d-1"].indicators_elsewhere is True and rows["d-1"].n_tables == 2
    assert registry.list_documents(path)[0].doc_id == "d-2"  # sorted by relative path
    assert registry.get_document("d-1", path).sha256 == "abc"
    ref = registry.DocumentRef(doc_id="d-1", path=tmp_path / "x.docx", relative_path="proposals/x.docx", region="africa", country="x", file_type="docx", sha256="abc")
    assert registry.needs_processing(ref, rows, pipeline_version="kb-v1") is False
    assert registry.needs_processing(ref, rows, force=True, pipeline_version="kb-v1") is True
    assert registry.needs_processing(ref, rows, pipeline_version="kb-v2") is True
    assert registry.needs_processing(ref._replace(sha256="changed") if hasattr(ref, "_replace") else registry.DocumentRef(**{**ref.__dict__, "sha256": "changed"}), rows, pipeline_version="kb-v1") is True
    failed = registry.DocumentRef(doc_id="d-2", path=tmp_path / "a.pdf", relative_path="proposals/a.pdf", region="", country="", file_type="pdf", sha256="abc")
    assert registry.needs_processing(failed, rows, pipeline_version="kb-v1") is True  # failed docs are retried


def test_language_heuristic_and_text_utils():
    en = "The project will improve access to water and sanitation for the displaced households in the region of the country."
    fr = "Le projet vise à améliorer l'accès à l'eau et à l'assainissement pour les ménages déplacés dans la région du pays."
    assert detect_language(en * 2) == "en"
    assert detect_language(fr * 2) == "fr"
    assert detect_language(en + " " + fr) == "mixed"
    assert detect_language("short") == ""
    assert normalise_whitespace("a  b\n c d") == "a b c d"
    assert fix_hyphenation("house-\nholds and water-\n borne") == "households and waterborne"
    assert slugify("Asia/syria/large/Copy of Annex_14 (final).xlsx") == "asia-syria-large-copy-of-annex-14-final-xlsx"
