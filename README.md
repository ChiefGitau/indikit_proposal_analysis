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
| **Preprocessing** (paper 3.3) | `pipeline/preprocess/` | layout-aware parsers (PDF, docx, xlsx), outline, reference detection, text / outline / table knowledge bases + vector index | Phase 2 (stubs) |
| **LLM agent** (paper 3.4) | `pipeline/agent/` | retrieval + re-ranking, five-part prompt, `DisclosureAnswer` schema, novel-indicator pass | Phase 3 (prompt template and LLM plumbing exist) |
| **Evaluation** (paper 5.1) | `pipeline/eval/` | gold-set format + validator, metrics (Acc_DC, Acc_DE, precision/recall, match-type, reference detection), experiment runner | Phase 1 scaffold; experiments in Phase 3 |

| Phase | Scope | Status |
|---|---|---|
| 1 — Foundation | repo clean-up, `pipeline/` skeleton, metadata module, evaluation scaffold | done (`docs/phase_1_report.md`) |
| 2 — Preprocessing | parsers, outline, references, knowledge bases, retrieval | next |
| 3 — Agent & evaluation | prompt + answering, novel pass, benchmark + ablations, full run, Streamlit | later |

`extraction_utils.py` is now a thin compatibility shim over `pipeline/`; the notebook
`proposal_indicator_analysis.ipynb` still runs the legacy extraction and (deprecated) matching v1.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # optional: lets `python -m pipeline.run` work from any directory
cp .env.example .env        # then add your own OPENAI_API_KEY (only needed for LLM calls)
```

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
pipeline/            config.py, models.py, run.py (CLI), metadata/, preprocess/, agent/, eval/
docs/                plan.md (ESGReveal transformation plan), phase_1.md (brief), phase_1_report.md,
                     plan_v1_roadmap.md, archive/plan_2026-08/ (earlier plans)
tests/               pytest suites (conftest blocks network access)
proposals/           115 project documents (75 docx, 33 pdf, 7 xlsx)
outputs/             api_cache/ (legacy extraction cache, git-ignored), metadata/, figures/, result CSVs
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
