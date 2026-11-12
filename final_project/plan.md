# Indicator Matching Pipeline – Implementation Plan

**Goal:** automatically read ZOA project documents (proposals, logframes, contracts), extract their M&E indicators, and match each one to the ZOA IndiKit indicator library (CSV export + zoa.indikit.net) as **verbatim**, **close** or **none**. Flag cases where the indicators sit in another document (e.g. "see Annex C – Logical Framework").

**Basis:** the architecture of *ESGReveal* (Zou et al., 2023, arXiv:2312.17264), adapted to this task, and built on top of the existing `proposal_investigated` project.

---

## 1. Design

### 1.1 Adapting ESGReveal

ESGReveal works from the standard outwards: for each of its 70 indicators it searches a report and asks "is this disclosed?". IndiKit has ~390 indicators, and our documents keep their indicators in one place (the logframe), so we reverse the direction: **extract the document's indicators first, then match each one to IndiKit.** This also captures indicators that have no IndiKit match.

| ESGReveal module | In the paper | In this pipeline |
|---|---|---|
| Metadata module | Indicator stored as Aspect, KPI, Topic, Quantity + search terms, expert knowledge, prompt template | IndiKit library: ID, sector, sub-sector, level, name, wording (EN/FR), purpose + search terms and matching notes |
| Report preprocessing | Layout analysis, table reconstruction, separate knowledge bases for text, tables, outline | Parse docx / PDF / xlsx with tables intact; locate the logframe; detect references to other documents |
| LLM agent | Retrieve top-k, re-rank, prompt with reference content + expert knowledge + question + fixed answer format | Retrieve top-k IndiKit candidates per indicator, re-rank, LLM judges verbatim / close / none with ID and reason |
| Evaluation | Labelled reports; disclosure and extraction accuracy; ablation per module | Gold set; extraction and matching accuracy; same ablations |

The paper's ablation is the argument for building the modules properly: with GPT-4, basic retrieval reached 57.8% disclosure accuracy / 52.2% extraction accuracy; improved preprocessing and retrieval raised this to 81.2% / 74.0%.

### 1.2 Pipeline overview

```
documents ──► parse (text + tables) ──► locate logframe / detect references
                                              │
                                              ▼
                                  LLM extraction (fixed schema)
                                              │
                                              ▼
 IndiKit library ──► embeddings ──► top-10 candidates ──► re-rank to top-5
                                              │
                                              ▼
                              LLM judge: verbatim / close / none
                                              │
                                              ▼
                     results table ──► human review of "close" ──► dashboard
```

### 1.3 Match definitions (rubric, to be refined with examples)

- **Verbatim:** same measurement, (nearly) identical wording. Example: "Number of people having access to basic, safe and dignified shelter solutions" ↔ *Access to Adequate Shelter* (5179).
- **Close:** same measurement, different wording or scope. Example: "75% of trained PIP farmers report increased yields" ↔ *Agricultural Productivity* (5260).
- **None:** no IndiKit indicator measures the same thing (often activity counts: trainings held, VSLAs supported).
- **Not an indicator:** activities, assumptions, or targets without a measure are excluded at extraction.

---

## 2. Current state of the existing project

Reviewed: `proposal_investigated.zip` (code and saved outputs read; pipeline not re-run).

| Stage | Status | Findings |
|---|---|---|
| Indicator library | Mostly done | CSV export (386 rows), site scraper (303 indicators scraped), Streamlit profiler app. The seven PIP indicators (IDs 5889–5895) are missing from both; the two sources are not merged |
| Document parsing | Partly done | PDF and docx parsed; docx tables kept. The 7 xlsx files in `proposals/` are skipped; PDFs read as plain page text (table structure lost); no OCR |
| Logframe location | Done, simple | Keyword scoring per chunk, neighbours kept. No detection of references to other documents |
| Indicator extraction | Works | 3,409 cached responses, 1,081 raw indicators over 115 proposals. Docx paragraphs are sent one per chunk, so 3,146 responses are empty. Field validation is loose (`level` contains values such as "refugees") |
| Matching v1 (fuzzy + embeddings) | Not working | `reference_library.csv` is missing, so a 3-row dummy library is used; embedding score is 0 everywhere. Saved output: 11 indicators, all "novel" — including one that is verbatim IndiKit 5179 |
| Matching v2 (LLM linking) | Not working | Only the first 30 IndiKit indicators are shown (`head(30)`), names only. Calls `client.chat` on `LLMClient`, which has no such attribute → every row returns "Error". Cells are duplicated in the notebook |
| Evaluation | Missing | Six unit tests on helpers; no gold set, no accuracy numbers |
| Output / review | Missing | Notebook has no saved outputs; summary tables depend on broken linking; app's Proposal Overview uses dummy data |

> **Security:** the `.env` in the zip contains what appears to be a live OpenAI API key. Revoke it, issue a new one, and keep `.env` out of shared archives and version control.

---

## 3. Implementation phases

### Phase 1 – Indicator library (metadata module)

1. Merge the CSV export with the site scrape, deduplicating by indicator ID (the same ID appears under several sub-sectors in the export).
2. Add indicators that exist only on the site (at minimum PIP 5889–5895) and decide which source is authoritative.
3. Fix encoding (export is Windows-1252 / latin-1) and normalise fields: ID, sector, sub-sector, level, name, wording EN/FR, purpose, URL.
4. Add the two ESGReveal extension fields per indicator:
   - **Search terms:** synonyms and donor phrasings (e.g. BHA codes W8, S5; EU "FCS > 35").
   - **Knowledge:** 1–2 lines on what counts as a match and what does not.
5. Embed name + wording + purpose; store the vector index.
6. Save as `reference_library.csv` (the path the notebook already expects).

**Done when:** one deduplicated library file with all site indicators, and an embedding index built from it.

### Phase 2 – Gold set and rubric (before further pipeline work)

1. Use the eleven documents already analysed by hand (Kinanira, Ruyigi, Addendum Bururi, SNV contract, Annex A2, TEAR, We are able!, EU logframe xlsx, EUTF DoA, BHA Tigray, CBPF final proposal): ~240 labelled indicators.
2. Correct the manual matches where MEAL colleagues disagree.
3. Write the rubric with 3–4 worked examples per class (verbatim / close / none / not an indicator).
4. Hold back 2–3 documents never used for prompt tuning.

**Done when:** a CSV with one row per document indicator → expected IndiKit ID and match class, plus the rubric document.

### Phase 3 – Document preprocessing

1. Parse per format, outputting text plus tables as rows and columns:
   - docx: keep paragraphs and tables in document order.
   - PDF: table-aware extraction (e.g. pdfplumber / PyMuPDF table finder); OCR fallback for scanned pages.
   - xlsx: read every sheet row by row (logframes are often delivered this way).
2. Chunk by **section or whole table**, not per paragraph.
3. Logframe locator: headings ("Logical Framework", "Indicator", "Cadre logique", "Results framework") and table-header detection (indicator / baseline / target / means of verification).
4. Reference detector: flag phrases pointing elsewhere ("see Annex C", "Indicator tracker", "IPTT", "indicator database") and return the referenced document name.

**Done when:** every gold-set document yields its logframe chunk(s), and reference-only documents (e.g. Annex A2) are flagged with the other document's name.

### Phase 4 – Indicator extraction

1. One LLM call per logframe chunk, structured output enforced (JSON schema / Pydantic):
   `indicator_text, level (impact|outcome|output|unclear), baseline, target, unit, means_of_verification, donor_code, page_or_section`.
2. Prompt rules from observed problems:
   - activities are not indicators;
   - one row may contain several indicators → split them;
   - keep donor codes (S4, W8, SCS2);
   - `level` only from the allowed values.
3. Deduplicate within each document.
4. Reuse the existing cache where the prompt is unchanged; bump the prompt version when it changes.

**Done when:** extraction recall and precision are measured on the gold set (see Phase 6).

### Phase 5 – Matching (LLM agent)

1. Retrieval: embed each extracted indicator; take the top 10 IndiKit candidates from the **full** library.
2. Re-rank to top 5 (cross-encoder or LLM-based).
3. Judge prompt, structured as in ESGReveal:
   1. instructions + rubric;
   2. the document indicator (with target and unit);
   3. the 5 candidates with full wording, purpose and knowledge notes;
   4. question;
   5. answer format: `match_type, indikit_id, reason` (allow `none`).
4. Optional cross-check with a second model; disagreements go to human review.
5. Replace notebook cells 17–34 (Phase 4/5 duplicates) with a single call to the new module; remove `head(30)` and the `client.chat` bug.

**Done when:** every extracted indicator has a match type, an ID (or none) and a reason.

### Phase 6 – Evaluation

| Metric | Definition |
|---|---|
| Extraction recall | Share of gold indicators found |
| Extraction precision | Share of extracted items that are real indicators |
| Matching accuracy | Share with correct class **and** correct ID, reported per class (verbatim / close / none) |
| Reference detection | Share of reference-only documents correctly flagged with the right document name |

Ablations (as in the paper):
- with vs. without search terms and knowledge notes;
- with vs. without re-ranking;
- top-k = 5 / 10 / 20;
- two or three models (cost vs. accuracy).

**Done when:** a results table per configuration on the held-out documents.

### Phase 7 – Output, review and dashboard

1. Results table: `document, indicator, level, target, match_type, indikit_id, indikit_name, reason, source_page, reviewer_decision`.
2. Review step: MEAL colleague confirms or rejects **close** matches only; confirmed decisions are added to the knowledge notes.
3. Summary counts computed in code (per document, per sector, overall), never by the LLM.
4. Feed results into the Streamlit app's Proposal Overview page (replacing dummy data) and/or the ZOA indicator dashboard.
5. Report for the portfolio: IndiKit coverage per sector and the most frequent unmatched indicators (candidates to add to IndiKit).

**Done when:** colleagues can open the results, review close matches, and see aggregated coverage.

---

## 4. Order and effort

| Step | Phases | Estimated effort | Result |
|---|---|---|---|
| 0 | — | 0.5 day | API key rotated; repo cleaned (venv folders, duplicate cells, `.env` ignored) |
| 1 | 1 + 2 | 2–3 days | Clean library and gold set |
| 2 | 5 (on existing cached extractions) | 2–3 days | Working matcher; first accuracy numbers |
| 3 | 3 + 4 | 3–5 days | Better parsing and extraction, measured |
| 4 | 6 | 1–2 days | Ablations and model choice |
| 5 | 7 | 2–3 days | Reviewable output and dashboard |

Effort figures are estimates for one person working in Python. Step 2 comes before parsing improvements because the cached extractions (1,081 indicators) can be matched immediately, which gives value early.

---

## 5. Open decisions

- **Runtime:** Python prototype first (recommended, for evaluation and ablations), then possibly move to the Microsoft stack (Azure OpenAI, Power Automate, SharePoint, Dataverse).
- **Model and data policy:** which LLM provider may process ZOA documents (partner names, budgets).
- **Scope:** proposals only, or also reports, contracts and tracking tables (IPTT / indicator trackers).
- **Source of truth:** IndiKit site vs. CSV export when they differ.

---

## 6. References

- Zou, Y. et al. (2023). *ESGReveal: An LLM-based approach for extracting structured data from ESG reports.* arXiv:2312.17264.
- ZOA IndiKit: https://zoa.indikit.net/
- Related work: UNDP Seoul Policy Centre residency, *From gender commitments to practice: Using AI to look inside aid projects* (2026); FCDO / Frontier Technologies Hub International Development Knowledge Assistant; DEEP / HumSet; HumQuant (Liberatore et al., 2024).