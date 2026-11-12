# plan.md — Transforming `proposal_investigated` to the ESGReveal method

**Source method:** Zou et al. (2023), *ESGReveal: An LLM-based approach for extracting structured data from ESG reports*, arXiv:2312.17264.
**Project:** `proposal_investigated` (extraction notebook + `extraction_utils.py`) and `MEAL_Proposal-main` (IndiKit scraper, loader, Streamlit profiler).
**Goal:** for every ZOA project document, determine which ZOA IndiKit indicators it contains, how closely they match, and their baseline/target values — using the paper's three-module architecture and its evaluation method.

---

## 1. What "the paper's method" means here

ESGReveal is **standard-driven**: it starts from a fixed list of indicators (the metadata module) and, for each indicator, queries a pre-processed knowledge base of the report and asks an LLM whether the indicator is disclosed and what its value is. The answer format is `<Disclosure, KPI, Topic, Value, Unit, Target, Action>`.

The current project is **document-driven**: it extracts whatever indicators it finds and then tries to match them to a library.

The transformation therefore changes the core flow:

| | Current project | After transformation (paper method) |
|---|---|---|
| Starting point | Document chunks | IndiKit indicator list (metadata module) |
| Question asked | "Which indicators are in this chunk?" | "Is IndiKit indicator X in this document, and with what wording, baseline, target?" |
| Retrieval | Keyword score per chunk | Vector search over three knowledge bases (text, outline, tables) + re-ranking |
| LLM input | One chunk | Retrieved evidence + indicator metadata + expert knowledge + question + answer format |
| Output | List of extracted indicators | Document × indicator **disclosure matrix** with values and evidence |
| Evaluation | Unit tests on helpers | Disclosure accuracy (Acc_DC) and extraction accuracy (Acc_DE), baseline RAG, ablations |

**One addition beyond the paper:** a standard-driven method cannot see indicators that are not in IndiKit. The existing extraction step is kept as a secondary **novel-indicator pass**, so the output still reports document indicators with no IndiKit match.

---

## 2. Mapping: paper module → current code → target

| Paper module / component | Current code | Target |
|---|---|---|
| **Metadata – Entities** `<Aspect, KPI, Topic, Quantity>` | `zoa_indikit_app/src/loader.py` (CSV, latin-1), `notebooks/data/zoa_indikit_net/indicators.csv` (303 scraped) | `pipeline/metadata/build_metadata.py` → `indikit_metadata.jsonl` |
| **Metadata – Extensions** `<Knowledge>`, `<SearchTerm>` | none | Generated + curated fields per indicator |
| **Metadata – Expressions** (prompt templates) | `_build_extraction_prompt`, `link_indicator_to_zoa` (notebook) | `pipeline/agent/prompts.py` |
| **Preprocessing – layout analysis** | `simple_parse_pdf` (plain page text), `simple_parse_docx` (paragraphs, tables appended at end) | `pipeline/preprocess/` layout-aware parsers for PDF, docx, xlsx |
| **Preprocessing – table reconstruction** | `_serialize_docx_table` (docx only) | Table extraction for PDF, docx, xlsx with row/column structure |
| **Multi-type knowledge base** (text, outline, tables) | none (chunks only, keyword-scored) | `pipeline/preprocess/knowledge_base.py`, FAISS index per document |
| **LLM agent – retrieval + re-rank** | `filter_indicator_chunks` (keyword score), `_best_embedding_matches` | `pipeline/agent/retrieve.py` (vector top-N + cross-encoder re-rank) |
| **LLM agent – answering** | `LLMClient.extract_indicators_from_chunk`, cache helpers | `pipeline/agent/answer.py` (structured output, cached per doc × indicator) |
| **Evaluation** (Acc_DC, Acc_DE, ablation) | `tests/test_extraction_utils.py` (6 helper tests) | `pipeline/eval/` + gold set + experiment runner |
| Output / analysis | `outputs/indicator_matches.csv`, figures, Streamlit app (dummy proposal data) | Disclosure matrix + summaries feeding the Streamlit app |

---

## 3. Target structure

```
proposal_investigated/
├── pipeline/
│   ├── config.py                  # models, paths, thresholds, prompt versions
│   ├── metadata/
│   │   ├── build_metadata.py      # merge CSV export + site scrape → entities
│   │   ├── extensions.py          # Knowledge + SearchTerm generation
│   │   └── indikit_metadata.jsonl # output: one record per indicator
│   ├── preprocess/
│   │   ├── parse_pdf.py           # blocks, fonts, tables, OCR fallback
│   │   ├── parse_docx.py          # paragraphs + tables in document order
│   │   ├── parse_xlsx.py          # sheets → table rows
│   │   ├── outline.py             # heading tree from font/style info
│   │   ├── references.py          # "see Annex C", "Indicator tracker", IPTT
│   │   └── knowledge_base.py      # text / outline / table KBs + FAISS
│   ├── agent/
│   │   ├── retrieve.py            # query construction, top-N, re-rank
│   │   ├── prompts.py             # the five-part prompt template
│   │   ├── answer.py              # LLM call, schema validation, cache
│   │   └── novel_pass.py          # reuses existing extraction for non-IndiKit indicators
│   ├── eval/
│   │   ├── gold/                  # labelled documents
│   │   ├── metrics.py             # Acc_DC, Acc_DE, match-type accuracy
│   │   └── run_experiments.py     # baseline vs enhanced vs ablations
│   └── run.py                     # CLI: build KB → query → write outputs
├── notebooks/
│   └── proposal_indicator_analysis.ipynb   # thin: calls pipeline, shows results
├── outputs/
│   ├── kb/<doc_id>/               # per-document knowledge base + index
│   ├── cache/                     # LLM responses keyed by (doc, indicator, prompt version)
│   ├── disclosure_matrix.csv
│   ├── novel_indicators.csv
│   └── eval/
└── tests/
```

`extraction_utils.py` is split into these modules; functions worth keeping move rather than being rewritten (see Section 5).

---

## 4. Module specifications

### 4.1 Metadata module

**Entities.** Map IndiKit fields onto the paper's four-part entity:

| Paper field | Meaning in paper | IndiKit source |
|---|---|---|
| Aspect | ESG issue category | Sector + Sub-Sector |
| KPI | Indicator definition | Indicator name + English wording (French wording kept for search) |
| Topic | Sub-items to report | "Disaggregate by" section |
| Quantity | Absolute value, intensity, key action | Measurement type derived from wording: number, percentage, score, yes/no |

Plus: ID, level (Output / Outcome / Impact), URL.

**Build steps**
1. Load the export with `loader.load_indikit()` (latin-1) and the site scrape (`scraper.py` cache `scraped.json`).
2. Merge on indicator ID; deduplicate (the same ID appears under several sub-sectors in the export). Keep all sector assignments as a list.
3. Add site-only indicators (at least PIP 5889–5895, absent from both current sources); re-run the scraper's "only missing" mode.
4. Record which source each field came from, and which source wins on conflicts.

**Extensions** (the paper's `<Knowledge>` and `<SearchTerm>`)
- **Knowledge:** a short, user-friendly explanation of what the indicator measures and what does *not* count, generated by an LLM from Purpose + "How to collect and analyse" + "Important comments", then reviewed by a MEAL colleague. Store reviewer-confirmed matching decisions here over time.
- **SearchTerm:** retrieval keywords: synonyms, typical donor phrasings and codes (e.g. BHA "W8", EU "FCS > 35", "PIP plan"), French terms. Seed by LLM, then harvest from real logframes in the gold set (the paper gathered them "from an extensive array of" reports).

**Expressions:** prompt templates live in `prompts.py`, parameterised by entity + extension fields (Section 4.3).

**Output:** `indikit_metadata.jsonl`, one record per indicator.

### 4.2 Report preprocessing module

**Layout analysis and parsing**
- **PDF:** PyMuPDF text blocks with font size/weight (upgrade from the pinned 1.22.3 to a version with `find_tables`), or pdfplumber for tables. The paper used LayoutLMv3 / GeoLayoutLM and Table-Transformer; use these only if the simpler tools fail on the gold set. OCR fallback (Tesseract) for pages without a text layer.
- **docx:** walk the document body in order so tables stay next to their headings (currently all tables are appended after all paragraphs, losing context).
- **xlsx:** new; read every sheet as table rows (7 logframes in `proposals/` are currently skipped).

**Outline:** build a heading tree from font sizes (PDF) or heading styles (docx), linking each section to its text and tables. This is what lets retrieval find "Logical Framework" or "Annex 1" sections quickly.

**Reference detection:** scan for phrases pointing to another document ("see Annex C", "Indicator tracker", "IPTT", "indicator database", "cadre logique en annexe") and record the referenced document name per document.

**Multi-type knowledge base** (one per document, as in the paper)

| KB | Content | Vectors |
|---|---|---|
| Text | Section-sized chunks (not one per paragraph, as now) | One vector per chunk; optional LLM summary vector (paper used mT5 summaries) |
| Outline | Heading tree entries | One vector per heading, linked to its section |
| Tables | Reconstructed tables | One-to-many: per table, extracted keywords (indicator names, column headers) each get a vector pointing to the table; also one vector per logframe row |

**Embedding model:** the current `all-MiniLM-L6-v2` is English-only; ZOA documents are English and French. Use a multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2` or `bge-m3`). The paper used m3e for the same reason (Chinese/English).

**Storage:** FAISS index + metadata JSON under `outputs/kb/<doc_id>/`, rebuilt only when the file hash changes.

### 4.3 LLM agent module

**Retrieval of knowledge** (per document × indicator)
1. Build the query from metadata: KPI + SearchTerms.
2. Cosine search over the three KBs; take top-N from each (start N = 5).
3. Re-rank the combined candidates with a cross-encoder (paper used coROM; e.g. `bge-reranker-v2-m3`), keep top-k (start k = 4).

**Cost control.** 390 indicators × 115 documents ≈ 45,000 LLM calls if every pair is asked. Reduce before answering:
- **Sector routing:** only query sectors the document covers (detected from its outline or a single LLM classification call).
- **Retrieval gate:** skip pairs whose best re-rank score is below a threshold and record them as absent. Tune the threshold on the gold set so it does not lower recall.

**LLM answering** — prompt in the paper's five parts:
1. **Preset information:** role and rules ("answer only from the reference content", "an activity is not an indicator").
2. **Reference content:** the top-k retrieved passages/table rows with page or section labels.
3. **Expert knowledge:** the indicator's Knowledge field + the match rubric.
4. **Question:** "Does this document contain an indicator that measures [KPI] for [Aspect], disaggregated by [Topic]? If so, give its exact wording and values."
5. **Answer format** (JSON, schema-validated with Pydantic):

```json
{
  "indikit_id": "5179",
  "disclosure": "present | absent",
  "match_type": "verbatim | close | null",
  "document_indicator_text": "string or null",
  "level": "impact | outcome | output | unclear | null",
  "baseline": "string or null",
  "target": "string or null",
  "unit": "string or null",
  "location": "page/section label",
  "evidence": "short quote from reference content",
  "reason": "one sentence"
}
```

This mirrors the paper's `<Disclosure, KPI, Topic, Value, Unit, Target, Action>`, with `match_type` added for IndiKit.

Settings: temperature 0; cache key = (document hash, indicator ID, prompt version, model); retries with backoff (reuse `_call_openai_with_retries`).

**Novel-indicator pass** (beyond the paper)
1. Run the existing extraction (`LLMClient.extract_indicators_from_chunk`) only on chunks that the outline/table KB marks as logframe content.
2. Remove extracted indicators already linked to an IndiKit ID by the disclosure step (compare by evidence location and text similarity).
3. What remains goes to `novel_indicators.csv` (match type "none").

**Outputs**
- `disclosure_matrix.csv`: one row per document × present IndiKit indicator.
- `novel_indicators.csv`: document indicators with no IndiKit match.
- `document_summary.csv`: per document — indicators present (verbatim / close), novel count, referenced documents.

---

## 5. Changes to existing code

| Item | Action |
|---|---|
| `LLMClient` (defined twice, lines ~233 and ~515) | Keep the second version; move to `agent/answer.py` and `agent/novel_pass.py` |
| Cache helpers (`_chunk_hash`, `cache_response_for_chunk`, `load_cached_response`) | Move to `agent/answer.py`; extend key with indicator ID and model |
| `_call_openai_with_retries`, `_extract_json_from_response` | Keep; switch to schema-enforced structured output where the API supports it |
| `simple_parse_pdf`, `simple_parse_docx` | Replace with layout-aware parsers |
| `score_chunk_for_indicators`, `filter_indicator_chunks` | Keep as a cheap pre-filter for the novel pass only |
| `deduplicate_indicators`, `normalize_text_for_dedupe` | Keep (novel pass) |
| `match_indicators_to_library`, `_best_lexical_match`, `classify_match_status`, `THRESHOLDS` | Retire (replaced by the agent); keep tests until removed |
| `load_reference_library` dummy fallback | Remove; fail loudly if metadata is missing |
| Notebook cells 17–34 (duplicated Phase 4/5; `head(30)`; `client.chat` bug) | Delete; replace with calls to `pipeline.run` |
| `outputs/api_cache/` (3,409 responses) | Keep as cache for the novel pass; not reusable for the disclosure step (different prompt) |
| `zoa_indikit_app/src/embeddings.py` | Point to the shared multilingual model and metadata file |
| `zoa_indikit_app/src/proposals.py` (dummy JSON) | Read from `document_summary.csv` |
| Committed virtualenv folders (`zoa_indikit_app/&&/`, `zoa_indikit_app/source/`) | Delete; add to `.gitignore` |
| `.env` with live-looking OpenAI key | Revoke the key now; keep `.env` git-ignored and out of shared zips |
| `requirements.txt` | Upgrade `pymupdf`; add `faiss-cpu`, a reranker, `pdfplumber`/`pytesseract` as needed; pin versions |
| `plan/` folder (7 earlier plan files) | Archive; this file supersedes them |

---

## 6. Evaluation (following the paper)

### 6.1 Gold set
- Start from the 11 documents analysed by hand on 10 Nov 2026 (Kinanira, Ruyigi, Addendum Bururi, SNV contract, Annex A2, TEAR, We are able!, EU logframe xlsx, EUTF DoA, BHA Tigray, CBPF final proposal).
- Those labels were produced by an LLM-assisted review and must be **verified by a MEAL colleague** before use.
- Label format: per document, every IndiKit indicator present (ID, match type, document wording, baseline, target, location). All other IndiKit indicators count as absent.
- Add 5–10 documents from `proposals/` covering other countries, French-language documents and xlsx logframes.
- Hold out ~30% of documents; never tune prompts or thresholds on them.

### 6.2 Metrics

| Metric | Definition | Paper equivalent |
|---|---|---|
| **Acc_DC** — disclosure accuracy | Over all (document, indicator) pairs, share where predicted present/absent equals the label | Disclosure coverage accuracy |
| **Acc_DE** — extraction accuracy | Over truly present pairs, share where baseline and target values are extracted correctly | Data extraction accuracy |
| Precision / recall of "present" | Because most pairs are absent, Acc_DC alone is inflated; report these too | (addition) |
| Match-type accuracy | Over correctly found pairs, share with correct verbatim/close label | (addition) |
| Novel-pass precision | Share of "novel" indicators that are real indicators with no IndiKit match | (addition) |
| Reference detection | Share of reference-only documents flagged with the correct document name | (addition) |
| Cost and time | LLM calls, tokens, minutes per document | (addition) |

### 6.3 Experiments (mirror the paper's Section 5.1)

| Run | Description |
|---|---|
| **Benchmark** | Plain RAG: fixed-size chunks, MiniLM, no outline/table KBs, no re-rank, no Knowledge/SearchTerm |
| **Enhanced-RAG** | + layout-aware preprocessing, three KBs, re-ranking |
| **+ Knowledge** | + expert knowledge in the prompt |
| **+ SearchTerm** | + search terms in the retrieval query (= full method) |
| **Model comparison** | Full method with 2–3 models (e.g. a small and a large OpenAI/Azure model, one open model) |
| **Gate sensitivity** | Recall and cost at different retrieval-gate thresholds |

Report results as a table like the paper's Table 4 and an ablation figure like its Figure 3.

---

## 7. Roadmap

| Step | Work | Deliverable | Done when | Estimate |
|---|---|---|---|---|
| 0 | Revoke API key; delete venv folders; archive old plans; create `pipeline/` skeleton | Clean repo | Repo runs `pytest` from clean install | 0.5 day |
| 1 | Metadata module (entities + extensions) | `indikit_metadata.jsonl` | All site indicators present incl. PIP; Knowledge/SearchTerm reviewed for at least the sectors in the gold set | 2–3 days |
| 2 | Gold set + metrics code | `eval/gold/`, `metrics.py` | Colleague-verified labels for ≥ 11 documents | 2 days |
| 3 | Benchmark run (plain RAG) | First Acc_DC / Acc_DE | Baseline numbers recorded | 1 day |
| 4 | Preprocessing module (parsers, outline, tables, references, KBs) | `outputs/kb/` per document | Logframe tables reconstructed for all gold documents incl. xlsx and French | 3–5 days |
| 5 | LLM agent (retrieval, re-rank, five-part prompt, gate) | `disclosure_matrix.csv` | Enhanced-RAG beats Benchmark on held-out documents | 3–4 days |
| 6 | Novel-indicator pass | `novel_indicators.csv` | Novel-pass precision measured | 1–2 days |
| 7 | Ablations + model comparison | `outputs/eval/` report | Table and figure as in the paper | 1–2 days |
| 8 | Full run on 115 proposals; Streamlit integration; review workflow | Live app pages, summaries | MEAL colleague can review "close" matches and see coverage per sector/country | 2–3 days |

Estimates are for one person in Python and are not from the paper. The Benchmark run (step 3) comes early on purpose: it gives the number every later step must beat.

---

## 8. Risks and open decisions

- **Data policy:** documents contain partner names and budgets. Decide whether OpenAI, Azure OpenAI (fits ZOA's Microsoft stack) or a local model may process them.
- **Cost:** the standard-driven method is call-heavy; sector routing and the retrieval gate are required, not optional.
- **Accuracy ceiling:** in the paper, the full method with GPT-4 still had 76.9% extraction accuracy; plan for human review of "close" matches and low-confidence answers.
- **Source of truth:** site vs CSV export when they differ (seven PIP indicators exist only on the site).
- **Scanned and messy PDFs:** may need the heavier layout models (LayoutLMv3, Table-Transformer) the paper used.
- **Scope:** proposals only, or also reports, contracts and indicator trackers/IPTTs.

---

## References

- Zou, Y., Shi, M., Chen, Z., et al. (2023). *ESGReveal: An LLM-based approach for extracting structured data from ESG reports.* arXiv:2312.17264.
- ZOA IndiKit: https://zoa.indikit.net/