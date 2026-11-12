# Phase 1 report — Foundation (clean repo, pipeline skeleton, metadata module, evaluation scaffold)

Brief: `docs/phase_1.md`. Plan: `docs/plan.md`. Date: 2026-11-12.

## 1. What was done per task

### Task 1 — Repository clean-up
- `git init` in `proposal_investigated/` (branch `main`); first commit is the untouched state, excluding the items below.
- `.gitignore`: `.env`, `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `.DS_Store`, `.idea/`, `*.zip`, `.venv/`, `venv/`,
  the two committed virtualenvs under `zoa_indikit_app/`, `outputs/api_cache/`, `outputs/kb/`, `outputs/cache/`, `*.egg-info/`.
- Deleted `zoa_indikit_app/&&/`, `zoa_indikit_app/source/` (6.4 MB each) and the empty `code/`. `MEAL_Proposal-main.zip` stays on disk, ignored.
- `plan/` (7 files) → `docs/archive/plan_2026-08/`; `final_project/plan_expand.md` → `docs/plan.md`; `final_project/plan.md` → `docs/plan_v1_roadmap.md`; `phase_1.md` → `docs/phase_1.md`.
- `.env.example` with `OPENAI_API_KEY=` only. README carries the key-revocation reminder.
- `git ls-files | grep -c site-packages` → 0.

### Task 2 — `pipeline/` package skeleton
- `pipeline/__init__.py`, `pipeline/config.py` (paths resolved from the file, model names, prompt versions, raw-data locations, PIP seed URLs), `pipeline/models.py` (`DocumentChunk`, `Indicator`).
- `pipeline/agent/answer.py`: provider config, cache helpers (`_chunk_hash`, `cache_response_for_chunk`, `load_cached_response`), `_call_openai_with_retries`, `_extract_json_from_response`, `CacheMiss`; Phase 3 stub `answer_indicator`.
- `pipeline/agent/novel_pass.py`: `normalize_text_for_dedupe`, `deduplicate_indicators`, `score_chunk_for_indicators`, `filter_indicator_chunks`, `is_table_like_text`, the surviving `LLMClient`, the extraction prompt; Phase 3 stub `run_novel_pass`.
- Stubs with docstrings naming the paper section and phase: `pipeline/preprocess/{parse_pdf,parse_docx,parse_xlsx,outline,references,knowledge_base}.py`, `pipeline/agent/retrieve.py`.
- `pipeline/run.py`: subcommands `build-metadata`, `validate-gold`, `fetch-missing` (implemented) and `build-kb`, `query`, `novel-pass`, `run-experiments` (raise `NotImplementedError("Phase 2/3")`).
- `extraction_utils.py` is a shim: re-exports the moved names; runtime settings (`LLM_PROVIDER`, `MODEL_NAME`, `CACHE_DIR`, `FORCE_REFRESH`, `_call_openai_with_retries`, …) are forwarded to `pipeline.agent.answer` on read and write so the notebook and the existing `monkeypatch`-based tests keep working. The orphaned first `LLMClient` block is gone. `match_indicators_to_library`, `classify_match_status`, `_best_lexical_match`, `THRESHOLDS` warn `DeprecationWarning` but work. `load_reference_library` raises `FileNotFoundError` pointing to `python -m pipeline.run build-metadata` (no dummy fallback).
- `pyproject.toml` (pytest config + minimal setuptools metadata so `pip install -e .` makes `python -m pipeline.run` work from any directory); `tests/conftest.py` blocks every socket connection during tests.

### Task 3 — Metadata module: Entities
- `pipeline/metadata/models.py`: `IndicatorMetadata` (all fields of the brief, `field_sources` values `export | site | seed | derived | llm | override`), `Aspect`, `Quantity` enum.
- `pipeline/metadata/build_metadata.py`: latin-1 read + mojibake repair, collapse of repeated export rows into `aspect`, consistency check of repeated rows, site data combined from `scraped.json` (primary), the site CSV (gap filler) and `site_supplement.json`, merge with "site wins" conflict rule, level normalisation, rule-based `derive_quantity`, deterministic JSONL writer, `reference_library.csv` writer, conflicts CSV, build report.
- `pipeline/metadata/fetch_missing.py`: fetches pages by URL reusing `zoa_indikit_app/src/scraper.py` (its User-Agent and 1 s crawl delay), writes `pipeline/metadata/site_supplement.json`; seeds are used when no supplement exists.
- Outputs: `pipeline/metadata/indikit_metadata.jsonl` (351 records, byte-identical on rebuild), `reference_library.csv` (351 rows), `outputs/metadata/build_report.md`, `outputs/metadata/merge_conflicts.csv`.

### Task 4 — Metadata module: Extensions
- `pipeline/metadata/extensions.py`: deterministic baseline (`search_terms` from name, EN/FR key phrases without `[specify …]` placeholders, 50 known abbreviations, sector and sub-sector; lower-cased, de-duplicated, max 15; `knowledge` from purpose + first two sentences of how-to-collect + first sentence of important comments, ≤ 120 words), optional LLM enrichment (`--enrich`; cached under `outputs/cache/metadata/<id>__<prompt version>__<model>.json`; `EnrichmentResponse` validated with Pydantic), manual overrides (`overrides.yaml`, applied last, name verified with exact or token-set match).
- `pipeline/metadata/overrides.yaml` seeded with the five donor-code examples from the brief.
- After the baseline every record has ≥ 3 search terms and non-empty knowledge (0 exceptions).

### Task 5 — Prompt templates
- `pipeline/agent/prompts.py`: `build_disclosure_prompt(indicator, reference_content, rubric)` assembling the five parts (preset information, reference content, expert knowledge, question, answer format), `MATCH_RUBRIC` from `docs/plan.md` 1.3, `PROMPT_VERSION` from `config`.
- `pipeline/agent/models.py`: `DisclosureAnswer` (schema of `docs/plan.md` 4.3; `match_type` required iff `present`).

### Task 6 — Evaluation scaffold
- `pipeline/eval/gold/README.md` (format), `index.csv` (headers only), `_example.csv` (4 fictional rows).
- `pipeline/eval/gold.py`: Pydantic models, loader, validator (`validate_gold`, `validate_gold_file`, `load_gold`), exposed as `python -m pipeline.run validate-gold`.
- `pipeline/eval/metrics.py`: `acc_dc`, `precision_recall_present`, `acc_de` with `normalise_value` (handles `75 %`, `0,75`, `24.397,00`, `24,397`, `1,234.5`, units, N/A), `match_type_accuracy`, `reference_detection_accuracy`; all on lists of Pydantic records.
- `pipeline/eval/run_experiments.py`: experiment names from `docs/plan.md` 6.3 and a CSV + Markdown results-table writer; bodies raise `NotImplementedError("Phase 3")`.

### Task 7 — Notebook, requirements, docs, tests
- Notebook: cells 17–28 (the duplicated Phase 4/5 block; see discrepancies) replaced by one markdown cell and one code cell that loads `indikit_metadata.jsonl` and shows counts per sector and level; the "build reference library" cell now only loads the generated `reference_library.csv`; all outputs cleared (19 cells, 0 outputs).
- `requirements.txt`: added `pyyaml`, `requests`, `beautifulsoup4`; Phase 2/3 additions noted as comments.
- `README.md` rewritten (architecture, phase table, commands, data quirks, security note).
- Tests: `test_metadata_build.py` (31), `test_metadata_real.py` (4, skipped without the raw files), `test_extensions.py` (8), `test_prompts.py` (3), `test_gold.py` (5), `test_metrics.py` (37), `test_shim.py` (5); the 18 pre-existing tests in `test_extraction_utils.py` are untouched and pass. Total: 123 passed, no network, no API key.

## 2. Metadata numbers

| Item | Value |
|---|---|
| Export rows read / unique IDs | 386 / 344 |
| Site records (`scraped.json`) | 303 (all `ok`; levels for 263) |
| Supplement records (fetched PIP pages) | 7 |
| **Total records** | **351** (= 344 + 7, as expected) |
| In export and on site | 303 |
| Export-only (`on_site=false`) | 41 |
| Site-only (`in_export=false`) | 7 (PIP 5889–5895) |
| Seed-only (needs fetch) | 0 |
| Merge conflicts | **0** — export and site agree on every name, wording and purpose after normalisation (one French wording differs in whitespace only); sector/sub-sector triples agree for all 303 shared IDs |
| Repeated export rows that disagree | 0 |
| Records without English wording | 18: 5806–5814 and 5886 (empty in both export and site), 5885 (export-only, empty), 5889–5895 (PIP pages have no wording block) |
| Records without purpose | 0 |
| Records without level | 49 |
| Records without topic (disaggregation) | 154 |
| Quantity distribution | percentage 185, number 126, other 23, score 15, ratio 1, yes_no 1 |
| Extension exceptions (< 3 search terms or empty knowledge) | 0 |

Records per sector (primary aspect): Food Security and Livelihoods 172, WASH 56, CVA 32, Gender Diversity & Inclusion 24, Peacebuilding 23, Shelter 21, Climate Resilience 17, Localisation 3, Peace Nexus 3.

## 3. PIP indicators (5889–5895)

**Fetched**, not seeded: `python -m pipeline.metadata.fetch_missing` retrieved all seven pages on 2026-11-12 (7/7) into
`pipeline/metadata/site_supplement.json`. The pages carry a name (e.g. "PIP: # of households who are implementing their PIP
plans"), sector "Food Security and Livelihoods" and a purpose text, but **no wording block, no sub-sector, no disaggregation**
and no level. `kpi_wording_en` is therefore empty for the seven records (they are listed in the build report); the reference
library and the prompt fall back to the name. Levels for these IDs were not fetched because the site publishes levels only on
the sector listing pages (a full crawl of all sector × level filters; not run).

## 4. Overrides

All five override entries were applied (5138, 5330, 5620, 5639, 5889); none were skipped. Note: the brief's name for 5889
("PIP: % of households …") differs from the live page's name ("PIP: **#** of households …"); the entry matched through the
token-set similarity check (≥ 85), and `overrides.yaml` keeps the brief's spelling.

## 5. Discrepancies between the brief and the repository

| Brief says | Found |
|---|---|
| The root is not a git repository | True for `proposal_investigated/`, but the grand-parent folder `MEAL_Proposal/` is a git repository that does **not** track `version_2.1/`. A nested repository was initialised in `proposal_investigated/` as instructed; the parent still shows `version_2.1/` as untracked. |
| `extraction_utils.py` has 679 lines | 1,137 lines |
| 6 existing tests | 18 tests in `tests/test_extraction_utils.py` (all kept) |
| Notebook has 35 cells; cells 17–34 are the duplicated block | 29 cells; the duplicated Phase 4/5 block is cells 17–28 (an empty markdown cell, Phase 4 linking, manual check, Phase 5 summaries). Deleted 17–28. |
| `LLMClient` defined twice (lines ~233 and ~515) | The first "definition" (line 388) had lost its `class` line: an indented docstring and two methods nested as dead code inside `load_cached_response`. Removed; the second (line 689) survives in `pipeline/agent/novel_pass.py`. |
| `plan.md` at the root | No `plan.md` at the root. The plan the brief refers to (Sections 3, 4.3, 6.3) is `final_project/plan_expand.md`, now `docs/plan.md`; the older `final_project/plan.md` is `docs/plan_v1_roadmap.md`. |
| `keep a copy of plan.md at the root if it is already there` | Not applicable (none at the root). |
| Site CSV columns `sector`, `section:…` | Its `sector` column holds the IndiKit **sub-sector**; `scraped.json` already has how-to-collect / disaggregate / comments, so the CSV only fills gaps (it filled none). |
| `requirements.txt` with pinned versions | The existing file deliberately uses minimum versions with compatibility comments (Intel-Mac PyTorch issue). Kept that convention and added the three new packages the same way; exact pins would need a per-machine lock file. |
| `field_sources` values `export | site | seed` | Extended with `derived` (quantity, baseline extensions), `llm` (enrichment) and `override`. |
| `reference_library.csv` "missing" (plan Section 2) | A 303-row file generated by the notebook from the site CSV existed at the root; it is now regenerated from the metadata (351 rows, IDs instead of slugs). |
| `load_reference_library` returns `(df, dummy_flag)` | Signature kept; the flag is always `False` now, missing file raises. |

## 6. Deferred work and open questions for the repository owner

Deferred (by design or scope):
- LLM enrichment of Knowledge / SearchTerm was **not run** (needs a key and a data-policy decision); the mechanism is tested with a fake client.
- Levels for the seven PIP indicators (requires the sector-listing crawl in `scraper.fetch_site_levels`).
- `proposals/` (255 MB, partner names and budgets) is tracked in the local repository as the brief implies; decide before pushing to any remote whether it should be git-ignored or moved to a private store.
- The Streamlit app under `MEAL_Proposal-main/` was not touched; its `data/Export indikit.csv` pointer file is still not a CSV.
- Existing `tests/test_extraction_utils.py` now emits `DeprecationWarning`s for matching v1 (expected).

Open questions:
1. **Source of truth on conflicts.** The rule implemented is "site wins" (export is a dated extract). No conflict exists today, but the next export may differ: confirm the rule, or switch to "export wins" in `build_metadata._pick`.
2. **Export-only indicators (41).** Should the agent search documents for indicators that are no longer published on the site? They are kept (`on_site=false`) so Phase 3 can filter either way.
3. **PIP wording.** Accept the page name ("PIP: # of households …") as the KPI wording for 5889–5895, or ask the IndiKit team to publish wording blocks?
4. **Enrichment model and data policy.** Which provider may be used for `--enrich` (indicator texts only, no project data) and later for the documents?
5. **Repository nesting.** Keep `proposal_investigated/` as its own repository (current state), or move it to the top of `MEAL_Proposal/` and drop the nesting?
6. **Dependency pinning.** Keep minimum versions (portable) or add an exact lock file for reproducible installs?
