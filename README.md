# Proposal Indicator Extraction (ZOA IndiKit)

For each ZOA project document (proposal, logframe, contract) determine which **ZOA IndiKit** indicators it
contains (verbatim / close), with baseline and target, plus the document indicators that have no IndiKit
match. The method follows *ESGReveal* (Zou et al., 2023, arXiv:2312.17264), adapted in `docs/plan.md`.

> **Security note.** The `.env` file that shipped with the original archive contained a live-looking OpenAI
> key. **Revoke that key in the OpenAI dashboard and issue a new one.** `.env` is git-ignored; copy
> `.env.example` to `.env` and fill in the new key locally. Never put keys in code, notebooks or docs.

## Architecture (three modules, as in the paper)

| Module | Package | What it holds | Status |
|---|---|---|---|
| **Metadata** (paper 3.2) | `pipeline/metadata/` | every IndiKit indicator as `<Aspect, KPI, Topic, Quantity>` plus `<Knowledge>` and `<SearchTerm>`; `indikit_metadata.jsonl` | **Phase 1, done** |
| **Preprocessing** (paper 3.3) | `pipeline/preprocess/` | layout-aware parsers (PDF, docx, xlsx), outline, logframe detection, reference detection, text / outline / table knowledge bases + vector index, plain-RAG baseline | **Phase 2, done** |
| **LLM agent** (paper 3.4) | `pipeline/agent/` | retrieval + re-ranking, provider layer with budget enforcement, five-part prompt, multi-match `DisclosureAnswer`, pair selection (gate + routing), novel-indicator pass, consolidation + reverse check, run outputs | **Phase 3, done** |
| **Evaluation** (paper 5.1) | `pipeline/eval/` | gold-set format + validator, metrics (Acc_DC, Acc_DE, precision/recall, match-type, reference detection, novel P/R, extraction recall), experiments, error analysis | **Phase 3, done** (live experiments wait for verified gold) |
| **Review** | `pipeline/review/` | export rows that need review to Excel, import decisions, propose overrides, export gold files | **Phase 3, done** |

| Phase | Scope | Status |
|---|---|---|
| 1 — Foundation | repo clean-up, `pipeline/` skeleton, metadata module, evaluation scaffold | done (`docs/phase_1_report.md`) |
| 2 — Preprocessing | parsers, outline, references, knowledge bases, retrieval, baseline index | done (`docs/phase_2_report.md`) |
| 3 — Agent & evaluation | provider layer, answering, pair selection, novel pass, consolidation, evaluation, runs, review, Streamlit | done (`docs/phase_3_report.md`) |

`extraction_utils.py` is a thin compatibility shim over `pipeline/` (the old fuzzy matcher was removed in Phase 3);
`proposal_indicator_analysis.ipynb` is a thin walkthrough of the pipeline outputs (no LLM calls).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # optional: lets `python -m pipeline.run` work from any directory
cp .env.example .env        # then add your own key (only needed for LLM calls; see "Configuring the LLM")
```

On this machine the project virtualenv is `.venv/` (fastembed / ONNX; PyTorch is not usable on the Intel-Mac build);
the Streamlit app runs from the system Python which has `streamlit` installed.

## Commands

```bash
# Build the IndiKit metadata library (deterministic, no network, no API key):
python -m pipeline.run build-metadata
#   -> pipeline/metadata/indikit_metadata.jsonl   one record per unique indicator ID
#   -> reference_library.csv                      legacy format used by the notebook
#   -> outputs/metadata/build_report.md, outputs/metadata/merge_conflicts.csv

# Optional LLM enrichment of Knowledge / SearchTerm (makes API calls, cached under outputs/cache/metadata/):
python -m pipeline.run build-metadata --enrich --model gpt-4o-mini

# Fetch indicator pages that are missing from the local site data (network; writes pipeline/metadata/site_supplement.json):
python -m pipeline.metadata.fetch_missing

# Validate the gold set (pipeline/eval/gold/):
python -m pipeline.run validate-gold
python -m pipeline.run validate-gold --file pipeline/eval/gold/_example.csv

# Tests (no network, no API key needed; a socket guard in tests/conftest.py enforces this):
pytest -q
```

## Phase 2 — preprocessing, knowledge bases, retrieval

Everything here is offline (no LLM). Real models are loaded only by the CLI; tests use a deterministic hashing
embedder and a lexical re-ranker.

```bash
python -m pipeline.run build-kb --all                 # parse + index every document (incremental; --force to redo)
python -m pipeline.run build-kb --doc "proposals/africa/uganda/large/Hope/Copy of Logframe HOPE project - final.xlsx"
python -m pipeline.run build-baseline-kb --all        # plain-RAG baseline (fixed chunks, all-MiniLM-L6-v2)
python -m pipeline.run score-pairs --all --no-rerank  # best vector score per document x indicator -> outputs/kb/pair_scores.*
python -m pipeline.run score-pairs --limit 5          # with the cross-encoder (slow: ~13 pairs/s on CPU)
python -m pipeline.run retrieve --doc Tigray --indicator 5138          # inspect the evidence for one pair
python -m pipeline.run retrieve --doc Tigray --indicator 5138 --baseline --top-k 4
python -m pipeline.run kb-report                      # outputs/kb/preprocess_report.md
pytest -m integration                                 # sanity checks on the template documents with the real models
```

Knowledge-base folder (`outputs/kb/<doc_id>/`, git-ignored): `blocks.jsonl`, `tables.jsonl`, `rows.jsonl`,
`outline.json`, `references.json`, `kb_text.jsonl`, `kb_outline.jsonl`, `kb_table.jsonl`, `index_<type>.npy` +
`.ids.json`, `kb_meta.json`. `outputs/kb/manifest.csv` lists every document with its parse status, language,
logframe / reference counts and build time; `outputs/kb/_cache/embeddings.sqlite` memoises embeddings by text hash.

Models: embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and the re-ranker
`jinaai/jina-reranker-v2-base-multilingual`, both through fastembed (ONNX; the Intel-Mac PyTorch build is too old
for `BAAI/bge-reranker-v2-m3`, which is selectable as `--reranker st:BAAI/bge-reranker-v2-m3` where PyTorch works).
The interfaces Phase 3 uses are documented in `docs/interfaces.md`.

Known limitations: scanned PDF pages are only read with `--ocr` (needs `pytesseract` and the `tesseract` binary;
otherwise the document is marked `partial`); heavily merged or nested tables may lose cell alignment; PDF heading
levels are inferred from font sizes and can nest wrongly; the language heuristic is stop-word based.

## Phase 3 — LLM agent, runs, evaluation, review (end-to-end usage)

```bash
python -m pipeline.run build-metadata                       # 1. IndiKit metadata (351 records)
python -m pipeline.run build-kb --all                        # 2. parse + index every document (incremental)
python -m pipeline.run build-baseline-kb --all               #    plain-RAG index for the benchmark experiment
python -m pipeline.run score-pairs --all --no-rerank         # 3. best score per document x indicator (gate input)
python -m pipeline.run estimate --all                        # 4. pairs, calls, cost - no LLM call
python -m pipeline.run select-pairs --pilot                  #    the gate for the 5 pilot documents (selected_pairs.csv + estimate)
python -m pipeline.run run --pilot --copy-latest             # 5. pilot: answer -> novel pass -> consolidate -> reverse check -> outputs
python -m pipeline.run run --all --confirm-full-run          # 6. full run (resumable with --run-id; stops cleanly at the budget)
python -m pipeline.run report --doc "Annex C-Logical Framework"   # 7. one document's three-column table
python -m pipeline.run export-review                         # 8. outputs/latest/review.xlsx for MEAL staff
python -m pipeline.run import-review --file outputs/latest/review.xlsx --export-gold   # 9. apply decisions, propose overrides, grow the gold set
python -m pipeline.run experiments --split dev               # 10. benchmark / ablations / models / gate / reverse check (needs verified gold)
cd MEAL_Proposal-main/MEAL_Proposal-main/zoa_indikit_app && streamlit run app.py      # 11. pages 8 and 10 show outputs/latest
```

Step-only commands: `answer` (disclosure step of a run), `novel-pass [--no-llm]`, `consolidate`. Every `run`
prints the estimate first and refuses to start when it exceeds `MAX_CALLS_PER_RUN` / `MAX_COST_USD_PER_RUN`.

### Configuring the LLM (owner decision)

Nothing calls a model until `LLM_PROVIDER` is set, in `pipeline/config.py` or the environment / `.env`:

| Setting | Where | Meaning |
|---|---|---|
| `LLM_PROVIDER` | config / env | `openai`, `azure` (Azure OpenAI) or `ollama` (local, OpenAI-compatible). Empty = no live calls. |
| `OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` | `.env` only | read at call time, never logged or cached |
| `ANSWER_MODEL`, `STRONG_MODEL`, `EXTRACTION_MODEL` | config / env | model roles (Azure: deployment names); `STRONG_MODEL` only for the model comparison |
| `MAX_CALLS_PER_RUN`, `MAX_COST_USD_PER_RUN`, `MODEL_PRICES_USD_PER_M` | config | enforced in code (`UsageTracker`); unknown models are priced at the most expensive entry |
| `LLM_MAX_WORKERS`, `LLM_RATE_LIMIT_PER_MIN` | config | bounded concurrency and rate limiting |
| `GATE_TAU` / `GATE_TAU_PERCENTILE`, `GATE_TOP_M`, `ROUTE_SECTORS`, `INCLUDE_EXPORT_ONLY` | config / flags | pair selection |

Answers are cached under `outputs/cache/agent/` keyed by (document SHA-256, IndiKit ID, evidence fingerprint,
prompt version, rubric version, model); `--force` ignores the cache. Runs live under `outputs/runs/<run_id>/`
(`selected_pairs.csv`, `answers.jsonl`, `usage.json`, `docs/*.json`, the output CSVs, `run_report.md`); a confirmed
full run is copied to `outputs/latest/`.

### Adding gold documents

Label a document in the Phase 1 format (`pipeline/eval/gold/README.md`): one `<doc_id>.csv` with every document
indicator (IndiKit ID + match type, or `none`) and a line in `index.csv` with `verified=true` once a second MEAL
colleague has checked it, `split=dev` (tuning allowed) or `test` (held out). `python -m pipeline.run validate-gold`
checks the files. `import-review --export-gold` writes gold files from reviewed pipeline output.

### Reading the outputs

| File | One row per | Use |
|---|---|---|
| `document_indicators.csv` | document indicator | the main table: wording, level, baseline, target, `match_type` (verbatim / close / none), `indikit_id`, alternates, source (disclosure / reverse_check / extraction), evidence quote, `needs_review` |
| `disclosure_matrix.csv` | document x present IndiKit indicator | the paper's disclosure matrix |
| `novel_indicators.csv` | document indicator without IndiKit match | nearest IndiKit indicators and cluster id |
| `document_summary.csv` | document | counts, % matched, `note` = "indicators are in another document: ..." |
| `portfolio_summary.md` + `portfolio_*.csv` | - | totals, by country, by sector, most / never used, novel clusters |
| `run_report.md` | - | config fingerprint, versions, counts, tokens, cost, runtime, extrapolation |

### Limitations

- No verified gold labels exist yet, so no accuracy figure has been measured; the experiments run as soon as
  `index.csv` has `verified=true` rows (unverified labels only with `--allow-unverified`, watermarked).
- The retrieval gate uses vector scores (the re-ranked pair scores take ~1 h on CPU); its defaults (p90, M = 60)
  were not tuned.
- The reverse check and the LLM extraction of text-only logframes cost extra calls (~1 per novel candidate / section).
- Scanned PDFs need `--ocr`; nine documents are `partial`.

## Data sources and their quirks

All raw IndiKit files live under `MEAL_Proposal-main/MEAL_Proposal-main/` and are read-only; derived files go
to `pipeline/metadata/` and `outputs/`.

| Source | Path | Quirks |
|---|---|---|
| IndiKit export | `data/indicators/Export indikit.csv` | **latin-1 / cp1252** (mojibake such as `\x92` for `’` is repaired on read); 386 rows but only **344 unique IDs**: an indicator repeats once per sub-sector; trailing empty columns; `Guidance` is the site URL; 48 rows without level; 11 rows without wording |
| Site scrape | `data/indicators/data_analysis/site_scrape/scraped.json` | 303 live indicator pages (July 2026) with section / sector / sub-sector, EN/FR wording, purpose, how-to-collect, disaggregation, comments; levels for 263 of them read from the sector listings |
| Site CSV | `notebooks/data/zoa_indikit_net/indicators.csv` | same 303 pages, UTF-8; its `sector` column is the IndiKit **sub-sector**; used only to fill gaps |
| Supplement | `pipeline/metadata/site_supplement.json` | the seven **PIP indicators (5889–5895)** that exist only on the live site; fetched with `fetch_missing.py`; their pages carry a name and purpose but no wording block |

- **41 export-only indicators** are no longer on the site; they are kept with `on_site=false`.
- Export and site agree on every name, wording and purpose (0 merge conflicts); when they differ the site wins
  and the conflict is written to `outputs/metadata/merge_conflicts.csv`.
- Manual overrides for Knowledge / SearchTerm live in `pipeline/metadata/overrides.yaml` and always win.

## Repository layout

```
pipeline/            config.py, models.py, run.py (CLI), metadata/, preprocess/, agent/, eval/, review/
docs/                plan.md (ESGReveal transformation plan), method.md (how the code maps to the paper), rubric.md,
                     phase_{1,2,3}.md (briefs), phase_{1,3}_report.md, plan_v1_roadmap.md, archive/plan_2026-08/
tests/               pytest suites (conftest blocks network access)
proposals/           115 project documents (75 docx, 33 pdf, 7 xlsx)
outputs/             api_cache/ (legacy cache), kb/, kb_baseline/, cache/agent/, runs/, latest/, eval/ (all git-ignored), metadata/, figures/
extraction_utils.py  compatibility shim re-exporting the moved functions
```

## LLM providers (legacy extraction and optional enrichment)

Configured through `pipeline.agent.answer.configure_llm` (also reachable as `extraction_utils.configure_llm`):

- `openai`: `OPENAI_API_KEY` in `.env`. The cached extraction responses in `outputs/api_cache` were produced with `gpt-4o-mini`.
- `ollama`: a local model through Ollama's OpenAI-compatible endpoint (`ollama pull llama3.2`, server at `http://localhost:11434`).
- `azure`: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, optional `AZURE_OPENAI_API_VERSION`; the model name is the *deployment name*.

Responses are cached per chunk and per model, so switching models never reuses another model's answers.
Embeddings for the legacy matcher use `all-MiniLM-L6-v2` through fastembed (ONNX, no PyTorch); Phase 2 moves to a
multilingual model.
