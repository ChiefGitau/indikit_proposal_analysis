# Task: Build a Jupyter notebook that extracts indicators and metadata from NGO proposals

## Goal
Create a well-structured Jupyter notebook (`proposal_indicator_analysis.ipynb`) plus a small supporting module (`extraction_utils.py`) that ingests a folder of humanitarian/NGO project proposals (10–50 documents, mixed PDF and Word), extracts all monitoring & evaluation indicators together with their metadata, and produces (a) a clean structured dataset, (b) a similarity match against an existing reference indicator library, and (c) descriptive statistics and visualizations. The notebook must be readable and re-runnable end-to-end by a non-author.

## Inputs
- `proposals/` — folder containing `.pdf` and `.docx` files. Assume heterogeneous layouts: some proposals contain indicators in logframe / results-framework tables, others describe them in narrative text, many do both.
- `reference_library.csv` — an existing indicator library (~500 indicators) with at minimum columns: `indicator_id`, `indicator_text`, `sector`, `level`. (If the file is absent, generate a small dummy version so the notebook still runs, and mark the matching section clearly as using dummy data.)
- OpenAI API key from environment variable `OPENAI_API_KEY` (never hardcode it). Load via `python-dotenv` from a `.env` file that is listed in `.gitignore`.

## Architecture — four layers, each its own notebook section

### 1. Document parsing (deterministic)
- Use `pymupdf` (fitz) for PDFs and `python-docx` for Word files.
- Extract BOTH plain text (page by page) and tables. For PDFs, use pymupdf's `find_tables()`; for docx, iterate `document.tables` preserving merged-cell structure as best as possible.
- Normalize into a common intermediate format: a list of `DocumentChunk` records with fields `doc_id`, `source_file`, `chunk_type` ("table" | "text"), `page_or_section`, `content` (tables serialized as markdown), `char_count`.
- Handle failures gracefully: log per-file parse errors to a summary table, never crash the whole run.

### 2. Section targeting (deterministic)
- Score chunks for indicator-likelihood using keyword heuristics: "logframe", "logical framework", "results framework", "indicator", "means of verification", "MoV", "baseline", "target", "output", "outcome", "impact", "M&E", "monitoring".
- Tables matching logframe column patterns (e.g., headers containing indicator/baseline/target/MoV) get maximum priority.
- Keep all high-scoring chunks plus their immediate neighbors (indicators often span a chunk boundary). Discard clearly irrelevant sections (budgets, annex CVs, procurement) but record what was discarded so recall can be audited.
- Output: a filtered chunk list per document, with a coverage summary (share of document text retained).

### 3. LLM extraction (Claude API)
- For each candidate chunk, call the OpenAI API using the `openai` Python package's `OpenAI` client. Default model: `gpt-4o-mini` (cheap, strong at structured extraction — a good fit for 10–50 proposals); expose the model name as a config constant so it can be bumped to `gpt-4o` if extraction quality on messy logframes proves insufficient.
- Use **structured outputs**: define the extraction schema once as a Pydantic model and pass it via `client.beta.chat.completions.parse(response_format=IndicatorList)` so the API guarantees schema-valid JSON. Reuse the same Pydantic model for downstream validation.
- Extraction schema per indicator:
  ```json
  {
    "indicator_text": str,          // verbatim wording as found
    "level": "impact" | "outcome" | "output" | "activity" | "unclear",
    "baseline": str | null,
    "target": str | null,
    "means_of_verification": str | null,
    "sector": str | null,           // e.g. WASH, FSL, protection, health, education
    "disaggregation": str | null,   // e.g. by sex/age/disability
    "unit_or_type": str | null,     // %, number, ratio, qualitative
    "source_confidence": "high" | "medium" | "low"
  }
  ```
- Return an empty list for chunks with no indicators. Include the source `doc_id` and chunk reference alongside every extracted indicator so each row is traceable to its origin.
- Robustness requirements:
  - Retry with exponential backoff on API errors (max 3 attempts).
  - Structured outputs make malformed JSON rare; still validate every response against the Pydantic model. On validation failure or refusal, retry once, then log and skip.
  - Handle API errors distinctly: 429 rate limits (exponential backoff, respect `Retry-After` when present), model refusals (the `refusal` field in parsed responses — log the chunk and continue), and auth errors (fail fast with a clear message).
  - Cache raw API responses to disk keyed by a hash of the chunk content + prompt version, so re-running the notebook does not re-call the API for unchanged chunks. Include a `FORCE_REFRESH` flag.
- Deduplicate indicators that appear in both a logframe table and narrative text of the same proposal: normalize whitespace/case and use fuzzy matching (`rapidfuzz`, token_sort_ratio ≥ 90) within each document, keeping the table version as canonical and recording the merge.

### 4. Validation & assembly (pandas)
- Assemble everything into one DataFrame `indicators_df` (one row per unique indicator per proposal).
- Validation checks, each reported in a small QA summary table: missing indicator_text, missing level, duplicate rows, targets that are non-numeric where unit_or_type suggests a number, proposals that yielded zero indicators (flag for manual review).
- Save to `outputs/indicators_extracted.csv` and `outputs/indicators_extracted.xlsx` (xlsx with one sheet per proposal plus a combined sheet).

## Matching against the reference library
- Compute similarity between each extracted indicator and the reference library using two methods and report both:
  1. `rapidfuzz` token_sort_ratio (fast lexical baseline).
  2. Sentence embeddings (`sentence-transformers`, `all-MiniLM-L6-v2`) with cosine similarity.
- For each extracted indicator, record best match id/text/score for both methods and a combined status: `exact` (≥95 embedding & ≥90 lexical), `close` (embedding ≥ 0.8), `related` (embedding ≥ 0.65), `novel` (below). Thresholds must be defined as named constants at the top so they are easy to tune.
- Output `outputs/indicator_matches.csv` and a summary: how many proposal indicators are already covered by the library vs. novel candidates for library expansion.

## Statistics & visualizations (matplotlib, one chart per cell)
- Indicators per proposal (bar).
- Distribution by level (impact/outcome/output) and by sector (stacked or grouped bar).
- Share with baseline defined, target defined, MoV defined — the "logframe completeness" view (horizontal bar).
- Match status distribution vs. the reference library (bar or pie).
- A completeness heatmap: proposals × metadata fields, showing % filled.
- Save all figures to `outputs/figures/` as PNG in addition to inline display.

## Notebook quality requirements
- Markdown cell before every section explaining what it does and why, written for a mixed technical/M&E audience.
- All configuration (paths, model name, thresholds, flags) in a single config cell at the top.
- Deterministic where possible: set seeds, sort outputs, so diffs between runs are meaningful.
- A final "Findings" markdown cell template with placeholders the analyst fills in.
- Total runtime target for 50 proposals with warm cache: under 2 minutes (excluding first-run API calls).

## Deliverables
1. `proposal_indicator_analysis.ipynb`
2. `extraction_utils.py` (parsing, chunking, API client with caching, matching functions — unit-testable)
3. `requirements.txt` (pin: pymupdf, python-docx, openai, pydantic, python-dotenv, pandas, rapidfuzz, sentence-transformers, matplotlib, openpyxl)
4. `tests/test_extraction_utils.py` — pytest tests for the deterministic parts: chunk scoring, JSON repair/parsing, dedup logic, threshold classification (use tiny fixture documents; do not call the live API in tests — mock it)
5. `README.md` — setup, how to run, how to interpret outputs

## Out of scope
- OCR of scanned PDFs (detect and flag image-only PDFs in the parse-error table instead).
- Non-English proposals (flag documents whose detected language is not English rather than attempting extraction).
- Building a UI — this is notebook-first; a Streamlit wrapper may come later, so keep logic in `extraction_utils.py` rather than inline in cells.
