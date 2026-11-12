# Phase 2 of 3 — Report preprocessing: parsers, knowledge bases, retrieval

You are working in the `proposal_investigated` repository. This is the second of three phases that transform the project to the method of *ESGReveal* (Zou et al., 2023, arXiv:2312.17264). Read `docs/plan.md`, `docs/phase_1.md` and `docs/phase_1_report.md` first; this file is the binding scope for Phase 2.

| Phase | Scope | Status |
|---|---|---|
| 1 — Foundation | Repo clean-up, `pipeline/` skeleton, metadata module, evaluation scaffold | Done (verify, Section 2) |
| **2 — Preprocessing** | Layout-aware parsers (PDF, docx, xlsx), outline, reference detection, three knowledge bases (text / outline / tables), vector index, retrieval + re-ranking, plain-RAG baseline index | **This task** |
| 3 — Agent & evaluation | Five-part prompt answering, novel-indicator pass, benchmark + ablations, full run, Streamlit integration | Later |

**Do not make any LLM call in this phase.** Everything here is parsing, indexing and retrieval. Phase 3 consumes what you build through the interfaces in Section 5.

---

## 1. What this phase implements

Paper Section 3.3 (*Report Preprocessing Module*) and Section 3.4.1 (*Retrieval of Knowledge*):

1. **Preprocess** each document: extract structural components (headings, paragraphs, tables), build an outline, reconstruct tables.
2. **Build a multi-type knowledge base** per document: text contents, document outline, table contents — each with vectors.
3. **Retrieve** per (document, indicator): build a query from the indicator metadata, vector-search the three knowledge bases, re-rank the top results, return the best evidence.

End state: for any document in `proposals/` and any indicator in `indikit_metadata.jsonl`, `retrieve(doc_id, indicator)` returns ranked evidence passages with page/section labels and scores — without calling an LLM.

---

## 2. Preconditions (check before writing code)

Run and confirm; if any fails, fix it only if trivial, otherwise stop and report.

- `pytest -q` passes offline.
- `python -m pipeline.run build-metadata` regenerates `pipeline/metadata/indikit_metadata.jsonl`; note the record count from `docs/phase_1_report.md`.
- `pipeline/preprocess/` and `pipeline/agent/retrieve.py` exist as stubs raising `NotImplementedError("Phase 2")`.
- Record models `DocumentChunk`, `Indicator`, `IndicatorMetadata`, `DisclosureAnswer` exist; note their module paths.
- Check whether the seven PIP indicators (5889–5895) have wording or are seed placeholders. Placeholders are still indexed (by name and search terms).

Where Phase 1 named things differently from this brief, **follow the repository** and list the differences in the final report.

---

## 3. Rules

1. **No LLM calls, no API keys.** Never read, print or copy `.env`.
2. **Tests run offline and fast.** No model downloads in tests: all embedding and re-ranking goes through interfaces with a deterministic fake implementation for tests (Section 4, Task 6). Real models are loaded only by explicit CLI commands.
3. **Never modify** files under `proposals/` or the raw IndiKit data. All derived data goes to `outputs/kb/` (git-ignored).
4. **A parse failure on one document must never stop a batch.** Catch, record in the manifest, continue.
5. Deterministic outputs: same input file + same config → identical chunk IDs and JSON.
6. Python ≥ 3.10, type hints, Pydantic v2 records, `pathlib`. Config values live in `pipeline/config.py`, not in function bodies.
7. Keep commits logical (parsers → outline/references → knowledge base → retrieval → baseline → CLI/docs).

---

## 4. Tasks

### Task 1 — Document registry and common models

`pipeline/preprocess/registry.py`, `pipeline/preprocess/models.py`

- Discover documents under `proposals/` recursively: `.pdf`, `.docx`, `.xlsx` (115 files: 75 docx, 33 pdf, 7 xlsx; folders `africa/{burundi,sudan,Ethiopia,uganda,DRC,south_sudan,Burkina}`, `Asia/{yemen,syria}`). Skip temp files (`~$…`).
- `doc_id`: stable and unique — slug of the relative path plus the first 8 hex characters of the file's SHA-256. File names alone are not unique across country folders.
- Manifest `outputs/kb/manifest.csv`: `doc_id, relative_path, region, country (from folder), file_type, sha256, n_pages_or_sheets, n_chars, n_tables, language, has_text_layer, parse_status (ok | partial | failed), error, parsed_at, pipeline_version`.
- `language`: `en`, `fr` or `mixed`, by a small stop-word heuristic (no new dependency). Unit-test it.
- Common models:
  - `Block`: `block_id, doc_id, kind (heading | paragraph | table | list_item), text, page (PDF) or None, order (reading order index), heading_path (list of ancestor heading titles), style_info (font size / bold / heading level)`.
  - `Table`: `table_id, doc_id, page_start, page_end, heading_path, caption, header (list[str]), rows (list[list[str]]), n_rows, n_cols, source (pdf | docx | xlsx), sheet_name`.
  - `TableRow`: `row_id, table_id, cells (dict header → value), text (serialised "header: value | header: value")`.
- Incremental processing: skip a document whose SHA-256 and `pipeline_version` are unchanged unless `--force`.

### Task 2 — Parsers (paper 3.3.1)

Replace the Phase-1 stubs. Each parser returns `(blocks, tables, warnings)` in **reading order**.

**`parse_docx.py`**
- Walk the document body element by element so paragraphs and tables stay in their original order (the old `simple_parse_docx` appended all tables after all paragraphs, which lost the heading context — do not repeat this).
- Headings from paragraph styles (`Heading 1…`, `Title`, localised variants such as `Titre 1`) with a fallback for bold, short, numbered lines ("3.2 Logical framework").
- Tables: handle merged cells (python-docx repeats the text of a merged cell — de-duplicate horizontally and vertically), nested tables (flatten into the parent cell text), empty rows/columns (drop).
- Header row detection: first non-empty row, or the first row matching logframe header patterns (Task 3).

**`parse_pdf.py`**
- Use PyMuPDF with block/span information (text, font size, bold flag, bounding box). Upgrade the pinned `pymupdf` to a version that provides `page.find_tables()`; fall back to `pdfplumber` when it finds none on a page that looks tabular.
- Remove repeating headers/footers and page numbers (text repeated at the same position on most pages).
- Headings: font size / weight relative to the document's body-text size, plus numbering patterns.
- Tables: exclude table regions from paragraph text so content is not duplicated. **Merge continuation tables** across consecutive pages (same column count and compatible/repeated header).
- Pages without a text layer: set `has_text_layer=False` in the manifest and add a warning. OCR is **optional**: implement behind a flag `--ocr` using `pytesseract` only if it is installed; never required by tests. Without OCR, mark the document `partial`.

**`parse_xlsx.py`** (new — 7 logframes are currently skipped)
- `openpyxl`, `data_only=True`. One `Table` per sheet region: detect the header row (first row with ≥ 3 non-empty cells matching header patterns, else first dense row); unmerge merged cells by forward-filling their value; drop empty rows/columns; keep the sheet name; treat title rows above the header as the table caption.
- Very wide or sparse sheets: split into regions separated by fully empty rows.

**Shared**
- Normalise whitespace, fix hyphenation at line breaks, keep original characters (accents) intact.
- Hard limit on cell and block length with a warning rather than silent truncation.

**Accept when:** unit tests pass on synthetic fixtures generated inside the tests (a docx with heading → paragraph → table → paragraph order and a merged cell; an xlsx with merged cells and a title row; a small PDF with a two-page table), and a batch run over all 115 documents completes with every document in the manifest.

### Task 3 — Logframe detection, outline, references

**`outline.py`** (paper: "report outline that connects to the text")
- Build a heading tree per document from the blocks; every block and table gets its `heading_path`.
- Output `outline.json`: nodes with `node_id, title, level, page, block_ids, table_ids`.

**`logframe.py`**
- Header patterns in English and French, mapped to roles:
  - `indicator`: indicator, indicateur, OVI, "objectively verifiable"
  - `baseline`: baseline, "valeur de référence", "ligne de base", "situation de référence"
  - `target`: target, cible, "valeur cible", milestone
  - `mov`: "means of verification", "source of verification", "sources et moyens de vérification", MoV
  - `result`: outcome, output, impact, result, résultat, objectif, "intervention logic", "logique d'intervention"
  - `assumption`: assumptions, hypothèses, risks
- A table is a **logframe table** if its header maps to `indicator` plus at least one other role, or its heading path / caption matches logframe titles ("logical framework", "logframe", "results framework", "cadre logique", "cadre de résultats", "indicator table", "performance indicators"). Store `is_logframe`, `column_roles` on the table.
- Also flag **logframe sections** in text (headings as above) for documents that list indicators in prose or bullet form (e.g. BHA proposals list indicators per sector as numbered lines).
- For each logframe row, resolve the `result` context (carry the last non-empty outcome/output cell downwards) and store it on the row.
- Reuse and extend the keyword lists from the old `INDICATOR_KEYWORDS` / `TABLE_HEADER_PATTERNS`.

**`references.py`**
- Detect statements that indicators are held in another document. Patterns (EN/FR), e.g.: "see Annex …", "refer to Annex …", "Annex C – Logical Framework", "fill in Annex C", "Indicator tracker", "Indicator Tracking Table", "IPTT", "indicator database", "logframe is attached", "voir annexe", "cadre logique en annexe".
- Output per document: `references.json` — list of `{referenced_document, quote, location, kind (logframe | indicator_table | tracker | other), confidence}`.
- Derive `indicators_elsewhere` = true when references exist **and** no logframe table/section was found in the document; `indicators_also_elsewhere` when both exist. Add both to the manifest.

**Accept when:** unit tests cover EN and FR header mapping, the carried-down result context, and reference detection positives and negatives ("see Annex 3 for the budget" must not be flagged as an indicator reference).

### Task 4 — Multi-type knowledge base (paper 3.3.2)

`pipeline/preprocess/knowledge_base.py`. One knowledge base per document, stored under `outputs/kb/<doc_id>/`:

```
blocks.jsonl        tables.jsonl       outline.json      references.json
kb_text.jsonl       kb_outline.jsonl   kb_table.jsonl
index_text.*        index_outline.*    index_table.*     (vector index + id map)
kb_meta.json        (embedding model, dims, pipeline_version, counts)
```

**Common entry model** `KBEntry`: `entry_id, doc_id, kb_type (text | outline | table), text_for_embedding, payload_ref (block/table/row/node ids), display_text, location (page / sheet / heading path), is_logframe`.

- **Text KB:** section-aware chunks — consecutive paragraph blocks under the same heading, target ~200–300 words, never crossing a top-level heading, with the heading path prefixed to `text_for_embedding`. (Not one chunk per paragraph, as the old code did, and not one per page.)
- **Outline KB:** one entry per outline node (title + parent titles). `payload_ref` points to the section's chunks and tables, so an outline hit can be expanded to its content.
- **Table KB (one-to-many, as in the paper):** for each table, several entries that all point to the same table:
  1. one entry per **row** (serialised `header: value` text, plus the carried-down result context);
  2. **keyword** entries: header cells, the `indicator` column cells, first-column cells, caption — "to exclude non-essential details and highlight key information such as … indicator names";
  3. one **table summary** entry (caption + header + heading path).
- Summaries: the paper embeds mT5 summaries of text. Do **not** generate summaries in this phase; leave an optional `summary` field and a hook so Phase 3 can test it as an ablation.

**Embeddings and index**
- `Embedder` interface (`embed(list[str]) -> np.ndarray`, L2-normalised). Implementations:
  - `SentenceTransformerEmbedder` — default model multilingual (documents are English and French): `paraphrase-multilingual-MiniLM-L12-v2`; model name in config. The old `all-MiniLM-L6-v2` is English-only and is kept only for the baseline (Task 7).
  - `HashingEmbedder` — deterministic, dependency-free (hashed token n-grams), used in all tests.
- Index: `faiss-cpu` inner-product index if installed, otherwise a NumPy brute-force index behind the same interface (documents are small; brute force is acceptable). Persist the index and the id map.
- Embed in batches; cache embeddings by text hash + model name so re-runs are cheap.

**Accept when:** `python -m pipeline.run build-kb --doc <path>` builds a complete folder for one document; `--all` processes every document incrementally; loading a KB from disk and searching it returns the same results as before saving.

### Task 5 — Indicator-side query vectors

`pipeline/agent/query.py`

- Build the query text per indicator from metadata: `name` + `kpi_wording_en` (placeholders such as `[specify …]` removed) + `search_terms`; append `kpi_wording_fr` when present. Two variants selectable by flag, because Phase 3 ablates them: `use_search_terms=True|False`.
- Pre-compute and cache query embeddings for all indicators in `outputs/kb/_queries/<model>/`.

### Task 6 — Retrieval and re-ranking (paper 3.4.1)

Replace the stub in `pipeline/agent/retrieve.py`.

```python
def retrieve(
    doc_id: str,
    indicator: IndicatorMetadata,
    *,
    top_n: int = 5,              # per knowledge base
    top_k: int = 4,              # after re-ranking
    kb_types: tuple[str, ...] = ("text", "outline", "table"),
    use_search_terms: bool = True,
    use_rerank: bool = True,
    config: RetrievalConfig | None = None,
) -> RetrievalResult: ...
```

- Steps: query vector → cosine top-N from each selected KB → **expand** (outline hit → its section chunks and tables; table keyword/summary hit → the matching rows, or the whole table if small) → de-duplicate by payload → **re-rank** → top-k.
- `Reranker` interface (`score(query: str, passages: list[str]) -> list[float]`). Implementations:
  - `CrossEncoderReranker` — multilingual cross-encoder (default `BAAI/bge-reranker-v2-m3`; if too heavy for the environment, use `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` and say so in the report); model name in config;
  - `LexicalReranker` — deterministic token-overlap scorer, used in tests and as the `use_rerank=False` fallback ordering.
- `RetrievalResult`: `doc_id, indikit_id, evidence (list[Evidence]), best_vector_score, best_rerank_score, n_candidates, config_fingerprint`.
- `Evidence`: `entry_id, kb_type, display_text, location, is_logframe, vector_score, rerank_score, table_id/row_id/block_ids`. `display_text` for a table row must include the table header and the row's result context, so the LLM in Phase 3 can read it standalone.
- Small boost (configurable, default on) for evidence with `is_logframe=True`; record whether the boost was applied.
- **Retrieval gate data:** `python -m pipeline.run score-pairs --all` computes, for every document × indicator pair, the best vector score and best re-rank score and writes `outputs/kb/pair_scores.parquet` (or CSV if `pyarrow` is unavailable): `doc_id, indikit_id, best_vector_score, best_rerank_score, top_entry_id, top_is_logframe`. Phase 3 tunes the gate threshold on this file. Make re-ranking optional here (`--no-rerank`) since it dominates runtime; report timings.

**Accept when:** unit tests with the fake embedder/reranker verify top-N per KB, outline expansion, de-duplication, the flag variants, and determinism.

### Task 7 — Plain-RAG baseline index (for the paper's "Benchmark")

`pipeline/preprocess/baseline_kb.py`

The paper compares its method to a basic RAG implementation. Build that baseline now so Phase 3 can evaluate both on equal terms:

- Flat text only (the old `simple_parse_pdf` / `simple_parse_docx` behaviour: page text and paragraphs + serialised tables, no outline, no table KB, no xlsx-specific handling beyond dumping cell text).
- Fixed-size chunks (~1,000 characters, 200 overlap), `all-MiniLM-L6-v2`, no search terms, no re-ranking.
- Stored under `outputs/kb_baseline/<doc_id>/`, searched by `retrieve_baseline(doc_id, indicator, top_k=4)` returning the same `RetrievalResult` type.

### Task 8 — CLI, reports, sanity checks

**CLI (`pipeline/run.py`)**
- `build-kb [--doc PATH | --all] [--force] [--ocr] [--embedder NAME]`
- `build-baseline-kb [--all]`
- `score-pairs [--all] [--no-rerank]`
- `retrieve --doc DOC_ID --indicator ID [--top-k N] [--no-rerank] [--no-search-terms]` → prints the ranked evidence (for manual inspection)
- `kb-report` → writes the report below

**`outputs/kb/preprocess_report.md`**
- Documents by type, country, language, parse status; failures with error messages.
- Per document: pages/sheets, blocks, tables, logframe tables found, logframe sections found, references found, `indicators_elsewhere`.
- Lists: documents with **no** logframe table or section and no reference (need manual inspection); documents without a text layer; the 20 largest tables; continuation-table merges performed.
- KB sizes (entries per type), embedding model, build time per document, total disk size.

**Sanity checks on real documents** (tests marked `integration`, skipped when files or models are missing; they use the real embedder). These documents are in `MEAL_Proposal-main/MEAL_Proposal-main/data/proposals/template/`; add that folder as an extra, clearly separate source for these checks only:

| Document | Indicator ID | Expected in top-k evidence |
|---|---|---|
| `ZOA Revised Proposal Tigray Emergency Response 2021-07-27, BHA.pdf` | 5138 (Handwashing Facilities With Soap and Water) | Passage with "soap and water at a handwashing station" |
| same | 5330 (Drinking Water Quality – residual chlorine) | Passage with "free residual chlorine" / "FRC" |
| `ZOA-we-are-able-full-proposal-v6.pdf` | 5620 (Food Consumption Score) | Passage/row with "Food Consumption Score" (indicator WaA!5) |
| same | 5643 (Land Tenure Security) | Passage/row on secure tenure rights to land (indicator WaA!6) |

Reference detection on the same files: the BHA proposal should yield an "Indicator Tracking Table" (Annex O) reference; We are able! should yield an "indicator database" reference. Verify the indicator names for these IDs in the metadata first; if an ID's name differs, report it instead of forcing the test.

If any check fails, investigate and fix the parser or KB — do not loosen the check. If it still fails, report it with the evidence that was retrieved.

**Retrieval recall hook:** `pipeline/eval/retrieval_eval.py` — given gold files (Phase 1 format), compute recall@k: the share of gold (document, IndiKit indicator) pairs whose gold `location` or `indicator_text` appears in the retrieved evidence, for the full method and the baseline. It must run cleanly with zero gold documents (prints "no verified gold documents") and is unit-tested on the fictional example gold file with a synthetic document.

### Task 9 — Requirements, docs, tests

- `requirements.txt`: add pinned `pymupdf` (upgraded), `pdfplumber`, `openpyxl`, `python-docx`, `sentence-transformers`, `numpy`; optional extras documented in comments: `faiss-cpu`, `pyarrow`, `pytesseract`. The code must work without the optional ones.
- README: add "Phase 2 — preprocessing" with the commands, the KB folder layout, how to inspect retrieval for one pair, and known limitations (scanned PDFs, very complex merged tables).
- Tests (all offline, fake embedder/reranker, fixtures created in `tmp_path`): `test_registry.py`, `test_parse_docx.py`, `test_parse_xlsx.py`, `test_parse_pdf.py`, `test_logframe.py`, `test_references.py`, `test_outline.py`, `test_knowledge_base.py`, `test_query.py`, `test_retrieve.py`, `test_baseline_kb.py`, `test_retrieval_eval.py`. Phase 1 tests must still pass.

---

## 5. Interfaces Phase 3 will rely on (keep stable)

```python
from pipeline.preprocess.registry import list_documents, get_document      # manifest rows
from pipeline.preprocess.knowledge_base import load_kb                     # KnowledgeBase
from pipeline.agent.retrieve import retrieve, retrieve_baseline            # RetrievalResult
from pipeline.preprocess.logframe import logframe_content                  # blocks/rows flagged as logframe (for the novel-indicator pass)
from pipeline.preprocess.references import load_references                 # per-document references
```

- `logframe_content(doc_id)` returns the logframe tables (with rows and column roles) and logframe text sections of a document, in reading order. Phase 3 feeds these — and only these — to the novel-indicator extraction.
- `pair_scores` file as specified in Task 6.

Document these in `docs/interfaces.md` with one short example each.

---

## 6. Out of scope for Phase 2

- Any LLM call: answering, extraction, summaries, sector classification, metadata enrichment.
- Tuning the retrieval-gate threshold or deciding sector routing (Phase 3; you only produce the scores).
- Accuracy evaluation (Acc_DC / Acc_DE), experiments, the full disclosure run.
- Streamlit changes; labelling gold documents.
- Heavy layout models (LayoutLMv3, Table-Transformer). If the report shows many documents failing with the lighter tools, recommend them in the report instead of adding them.

---

## 7. Definition of done

- [ ] `pytest -q` passes offline with no API key and no model download.
- [ ] `python -m pipeline.run build-kb --all` completes; every one of the 115 documents is in `outputs/kb/manifest.csv` with a status; no unhandled exception.
- [ ] All 7 xlsx files are parsed into tables.
- [ ] docx tables keep their position and heading context; PDF continuation tables are merged.
- [ ] `build-baseline-kb --all` and `score-pairs --all` complete; timings recorded.
- [ ] `retrieve` CLI prints evidence for a chosen pair; integration sanity checks pass or are reported with evidence.
- [ ] `outputs/kb/preprocess_report.md` and `docs/interfaces.md` written; README updated.
- [ ] No file under `proposals/` changed (`git status` shows none); nothing under `outputs/kb*` is tracked.

## 8. Final report (write to `docs/phase_2_report.md`)

1. What was built per task, with file paths; deviations from this brief and why.
2. Parse results: counts by status, type, language; list of failed/partial documents with causes.
3. Logframe coverage: how many documents have a logframe table, a logframe section only, references only, or nothing — with the "nothing" list.
4. References found: table of documents and the documents they point to.
5. KB statistics and runtimes (build, score-pairs with and without re-ranking); models actually used.
6. Sanity-check results with the retrieved evidence for each pair.
7. Known weaknesses seen in the data (scanned files, broken tables, mixed-language documents) and recommendations for Phase 3, including a first look at the score distributions in `pair_scores` (e.g. histogram summary) to help choose the gate threshold.
8. Open questions for the repository owner.