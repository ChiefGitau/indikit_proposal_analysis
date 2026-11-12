# Phase 1 of 3 — Foundation: clean repo, pipeline skeleton, metadata module, evaluation scaffold

You are working in the `proposal_investigated` repository. This is the first of three phases that transform the project to the method of *ESGReveal* (Zou et al., 2023, arXiv:2312.17264), as described in `plan.md`. Read `plan.md` first; this file is the binding scope for Phase 1.

| Phase | Scope | Status |
|---|---|---|
| **1 — Foundation** | Repo clean-up, `pipeline/` package skeleton, **metadata module** (IndiKit library with Entities + Extensions), evaluation scaffold (gold-set format + metrics) | **This task** |
| 2 — Preprocessing | Layout-aware parsers (PDF, docx, xlsx), outline, reference detection, three knowledge bases (text / outline / tables), vector index, retrieval + re-ranking | Later |
| 3 — Agent & evaluation | Five-part prompt, LLM answering, novel-indicator pass, benchmark + ablations, full run, Streamlit integration | Later |

**Do not start Phase 2 or 3 work.** Leave clearly named stubs where later phases plug in.

---

## 1. Context you need

**Goal of the whole project:** for each ZOA project document, determine which ZOA IndiKit indicators it contains (verbatim / close), with baseline and target, plus document indicators that have no IndiKit match.

**Method (paper):** standard-driven. A *metadata module* describes every indicator; a *preprocessing module* turns each document into searchable knowledge bases; an *LLM agent* queries the document once per indicator. Phase 1 builds the metadata module and everything the later phases stand on.

**Repository facts (verified):**

- Root: `extraction_utils.py` (679 lines), `proposal_indicator_analysis.ipynb` (35 cells), `requirements.txt`, `tests/test_extraction_utils.py` (6 tests), `outputs/` (`api_cache/` with 3,409 cached LLM responses, `figures/`, `indicator_matches.csv`), `plan/` (7 older plan files), `proposals/` (115 documents: 75 docx, 33 pdf, 7 xlsx), `.env`.
- The root is **not** a git repository. `MEAL_Proposal-main.zip` at the root duplicates the `MEAL_Proposal-main/` folder.
- IndiKit data lives under `MEAL_Proposal-main/MEAL_Proposal-main/`:
  - `data/indicators/Export indikit.csv` — encoding **latin-1**; header: `Section, Sector, Sub-Sector, Indicator Level, Indicator Name, Indicator Wording - ENG, Indicator Wording - FRA, Guidance, Unique ID, What is its Purpose?` followed by empty columns. **386 rows, 344 unique IDs** (an indicator repeats once per sub-sector it belongs to).
  - `data/indicators/data_analysis/site_scrape/scraped.json` — keys `scraped_at`, `records` (dict by ID, 303 entries with `url, unique_id, name, section, sector, sub_sector, wording_en, …`), `live_ids`, `levels` (ID → list of levels), `levels_at`. Scraped 2026-07-06.
  - `notebooks/data/zoa_indikit_net/indicators.csv` — 303 rows, UTF-8, columns `url, domain, slug, title, sector, section:Indicator’s Wording, section:Indicator’s Purpose, section:How to Collect and Analyse the Required Data, section:Disaggregate by, section:Important Comments, …`. The ID is the number at the start of `slug`.
  - `zoa_indikit_app/src/loader.py` (`load_indikit`), `scraper.py` (sitemap + per-indicator scraper, 1 s crawl delay, has an "only missing" mode), `embeddings.py`.
  - `zoa_indikit_app/data/Export indikit.csv` is **not** a CSV: it is a one-line pointer to `../../data/indicators/Export indikit.csv`.
  - `zoa_indikit_app/&&/` and `zoa_indikit_app/source/` are committed Python virtual environments.
- ID overlap: all 303 scraped IDs are in the export; **41 IDs are export-only** (no longer on the site in July 2026). Seven newer **PIP indicators exist only on the live site** (IDs 5889–5895) and are in neither file.
- Known defects you will touch: `LLMClient` is defined twice in `extraction_utils.py` (around lines 233 and 515); `load_reference_library` silently falls back to a 3-row dummy library; notebook cells 17–34 are a duplicated, broken "Phase 4/5" block (uses `head(30)` and calls `client.chat` on a class without that attribute).

---

## 2. Rules

1. **Secrets:** never print, log, copy or commit the contents of `.env`. Do not put any key in code, tests, notebooks or docs. (The repository owner will revoke the existing key manually; add a line to the README reminding them.)
2. **No live LLM calls and no network access in tests.** Anything that needs the network or an API key must be an explicit script with a deterministic offline fallback.
3. **Do not modify or delete:** `proposals/`, `outputs/api_cache/`, the raw IndiKit files listed above. Derived files go to new locations.
4. **Do not break existing tests.** `pytest -q` must pass at the end, including the 6 existing tests.
5. Python ≥ 3.10, type hints, `pathlib`, Pydantic v2 models for data records, no new heavy dependencies in this phase (no FAISS, no torch-based models beyond what is already required).
6. Keep changes reviewable: small modules, docstrings stating which part of the paper each implements.
7. If something in this brief contradicts what you find in the repo, stop guessing: follow the repo, and record the discrepancy in the final report.

---

## 3. Tasks

### Task 1 — Repository clean-up

1. Initialise git at the root if absent (`git init`), and make an initial commit of the current state **before** any change, excluding items in step 2.
2. Create/extend the root `.gitignore`: `.env`, `__pycache__/`, `.pytest_cache/`, `.DS_Store`, `*.zip`, virtualenv folders (`.venv/`, `venv/`, `zoa_indikit_app/&&/`, `zoa_indikit_app/source/`), `outputs/api_cache/`, `outputs/kb/`, `outputs/cache/`.
3. Delete the two committed virtualenv folders (`zoa_indikit_app/&&/`, `zoa_indikit_app/source/`) and the empty `code/` folder. Leave `MEAL_Proposal-main.zip` on disk but git-ignored.
4. Move the 7 files in `plan/` to `docs/archive/plan_2026-08/`. Put `plan.md` and the phase files in `docs/` (keep a copy of `plan.md` at the root if it is already there).
5. Create `.env.example` containing only `OPENAI_API_KEY=` and a comment.

**Accept when:** `git status` is clean after commit; no virtualenv or secret is tracked; `git ls-files | grep -c site-packages` returns 0.

### Task 2 — `pipeline/` package skeleton

Create the structure from `plan.md` Section 3:

```
pipeline/
├── __init__.py
├── config.py
├── metadata/        (Task 3–5)
├── preprocess/      __init__.py + stubs: parse_pdf.py, parse_docx.py, parse_xlsx.py,
│                    outline.py, references.py, knowledge_base.py
├── agent/           __init__.py + stubs: retrieve.py, prompts.py, answer.py, novel_pass.py
├── eval/            (Task 6)
└── run.py           CLI entry point with subcommands; only `build-metadata` and
                     `validate-gold` implemented in this phase
```

- `config.py`: one place for paths (repo root resolved from the file, not the working directory), model names, prompt versions, and the raw-data locations listed in Section 1. No hard-coded absolute paths.
- Stubs: module docstring naming the paper section they implement and the phase that fills them, function signatures with `raise NotImplementedError("Phase 2")` / `("Phase 3")`.
- Move, do not rewrite, the reusable parts of `extraction_utils.py`:
  - cache helpers (`_chunk_hash`, `cache_response_for_chunk`, `load_cached_response`), `_call_openai_with_retries`, `_extract_json_from_response`, and the **second** `LLMClient` definition → `pipeline/agent/answer.py` / `novel_pass.py`;
  - `normalize_text_for_dedupe`, `deduplicate_indicators`, `score_chunk_for_indicators`, `filter_indicator_chunks`, `is_table_like_text`, `DocumentChunk`, `Indicator` → `pipeline/agent/novel_pass.py` (or a small `pipeline/models.py`).
  - Remove the first, duplicate `LLMClient` definition.
- Keep `extraction_utils.py` as a thin compatibility shim that re-exports the moved names, so the notebook and existing tests keep working. Mark `match_indicators_to_library`, `classify_match_status`, `_best_lexical_match`, `THRESHOLDS` as deprecated (docstring + `warnings.warn`), but leave them functional.
- `load_reference_library`: remove the silent dummy fallback. Raise `FileNotFoundError` with a message pointing to `python -m pipeline.run build-metadata`. Update the existing test/notebook usage accordingly.

**Accept when:** `python -c "import pipeline, pipeline.config"` works from any directory; `python -m pipeline.run --help` lists the subcommands; existing tests pass through the shim.

### Task 3 — Metadata module: Entities

Implement `pipeline/metadata/build_metadata.py` (paper Section 3.2.1).

**Record model** (`pipeline/metadata/models.py`, Pydantic):

| Field | Type | Source / rule |
|---|---|---|
| `indikit_id` | str | `Unique ID` (export) / `unique_id` or slug prefix (site) |
| `name` | str | `Indicator Name` / `name` / `title` |
| `aspect` | list of `{section, sector, sub_sector}` | **All** sector/sub-sector assignments of the ID (collapse the export's repeated rows into this list) |
| `kpi_wording_en` | str | `Indicator Wording - ENG` / `wording_en` / `section:Indicator’s Wording` (English part) |
| `kpi_wording_fr` | str or None | `Indicator Wording - FRA` |
| `levels` | list[str] | `Indicator Level` (may hold several values) and `levels` in `scraped.json`; normalise to `Output`, `Outcome`, `Impact`, `Organisational level` |
| `topic` | str or None | `section:Disaggregate by` |
| `quantity` | enum: `number`, `percentage`, `score`, `ratio`, `yes_no`, `other` | Derived by rule from the wording (`%`/"percentage"/"proportion" → percentage; `#`/"number of" → number; "score"/"index" → score; etc.). Unit-tested. |
| `purpose` | str or None | `What is its Purpose?` / `section:Indicator’s Purpose` |
| `how_to_collect` | str or None | `section:How to Collect and Analyse the Required Data` |
| `important_comments` | str or None | `section:Important Comments` |
| `guidance` | str or None | `Guidance` (export) |
| `url` | str or None | site |
| `on_site` | bool | ID present in the site data |
| `in_export` | bool | ID present in the export |
| `field_sources` | dict[str, "export" \| "site" \| "seed"] | Which source supplied each field |
| `knowledge` | str or None | Task 4 |
| `search_terms` | list[str] | Task 4 |

**Merge rules**
1. Read the export as latin-1; drop the empty trailing columns; strip whitespace; repair obvious mojibake.
2. Group export rows by `Unique ID`; build the `aspect` list; verify that name and wording are identical across the repeated rows and log any ID where they differ.
3. Join with the site data (`scraped.json` records, enriched with the section columns from `notebooks/data/zoa_indikit_net/indicators.csv`).
4. Conflict rule: when both sources have a non-empty value that differs, **the site wins** for wording, purpose and sector fields; record the conflict in `outputs/metadata/merge_conflicts.csv` (ID, field, export value, site value).
5. Keep the 41 export-only indicators with `on_site=False` (do not drop them).
6. **PIP indicators (IDs 5889–5895):** add a script `pipeline/metadata/fetch_missing.py` that fetches indicator pages by URL reusing the parsing functions in `zoa_indikit_app/src/scraper.py` (respect the 1-second crawl delay and existing User-Agent). It must be run explicitly, not from tests. If the network is unavailable, create the seven records from the seed list below with `field_sources="seed"`, wording left empty, and flag them in the final report as needing a fetch.

   ```
   https://zoa.indikit.net/indicator/5889-pip-of-households-who-are-implementing-their-pip-plans
   https://zoa.indikit.net/indicator/5890-pip-of-people-undertaking-actions-to-manage-and-protect-natural-resources
   https://zoa.indikit.net/indicator/5891-pip-of-farmer-households-who-are-trained-through-peer-to-peer-learning
   https://zoa.indikit.net/indicator/5892-pip-of-households-with-increased-income
   https://zoa.indikit.net/indicator/5893-pip-of-farm-plans-pip-developed-jointly-by-the-household
   https://zoa.indikit.net/indicator/5894-pip-of-community-members-participating-in-collective-activities-in-the-community-as-defined-in-the-village-plan
   https://zoa.indikit.net/indicator/5895-pip-farm-productivity
   ```
   All seven belong to sector "Food Security and Livelihoods".

**Outputs** (under `pipeline/metadata/` for the library, `outputs/metadata/` for reports):
- `pipeline/metadata/indikit_metadata.jsonl` — one record per unique ID (expected: 344 + 7 = **351**; report the actual number and explain any difference).
- `reference_library.csv` at the repo root with columns `indicator_id, indicator_text, sector, level` (the format the notebook and `load_reference_library` expect), generated from the JSONL.
- `outputs/metadata/build_report.md` — counts: rows read, unique IDs, on-site / export-only / site-only, records per sector, missing-field counts, number of conflicts, quantity-type distribution.

**Accept when:** `python -m pipeline.run build-metadata` regenerates all three outputs deterministically (same input → byte-identical JSONL); every record validates against the model; no duplicate IDs; every record has a non-empty `name`, and a non-empty `kpi_wording_en` unless listed in the build report.

### Task 4 — Metadata module: Extensions (`<Knowledge>`, `<SearchTerm>`)

Implement `pipeline/metadata/extensions.py` (paper Section 3.2.2).

1. **Deterministic baseline (always runs, no API):**
   - `search_terms`: indicator name; key noun phrases from the English and French wording with template placeholders removed (`[specify …]`); known abbreviations found in the text (FCS, rCSI, HHS, FIES, MDD-W, VSLA, PIP, NFI, CVA, WASH, GBV …); sector and sub-sector names. Lower-cased, deduplicated, max 15 per indicator.
   - `knowledge`: a compact text assembled from `purpose` + first sentences of `how_to_collect` + `important_comments`, truncated to ~120 words.
2. **Optional LLM enrichment (explicit flag `--enrich`, off by default):** one call per indicator that rewrites `knowledge` as (a) what the indicator measures, (b) what does *not* count as the same indicator, and proposes additional `search_terms` (synonyms, donor phrasings). Use the existing retry/caching helpers; cache under `outputs/cache/metadata/` keyed by (ID, prompt version, model); validate the JSON response with Pydantic; never overwrite a manual override.
3. **Manual overrides:** `pipeline/metadata/overrides.yaml` (ID → `knowledge`, `search_terms_add`, `search_terms_remove`). Overrides always win and are applied last. Seed it with these donor-code examples so the mechanism is exercised:
   - `5138` (Handwashing Facilities With Soap and Water): add `W8`, "handwashing station", "soap and water"
   - `5330` (Drinking Water Quality – residual chlorine): add `W35`, "free residual chlorine", "FRC"
   - `5639` (Total CVA Value Distributed): add `K1`, "total USD value of cash transferred"
   - `5620` (Food Consumption Score): add "FCS", "FCS > 35", "acceptable food consumption"
   - `5889` (PIP households implementing plans): add "PIP plan", "plan intégré du paysan", "integrated farm plan"
   (Verify each ID's name against the built metadata before applying; if a name does not match, skip it and report.)

**Accept when:** every record has ≥ 3 `search_terms` and a non-empty `knowledge` after the deterministic baseline (list exceptions in the build report); running without `--enrich` makes zero network calls; overrides are reflected in the JSONL.

### Task 5 — Prompt templates placeholder (Expressions)

Create `pipeline/agent/prompts.py` with the **template only** (paper Section 3.2 "Expressions" and 3.4.2), no LLM call:

- A function `build_disclosure_prompt(indicator: IndicatorMetadata, reference_content: list[str], rubric: str) -> str` that assembles the five parts: preset information, reference content, expert knowledge, question, answer format.
- The answer format is the JSON schema from `plan.md` Section 4.3; define it as a Pydantic model `DisclosureAnswer` in `pipeline/agent/models.py`.
- `PROMPT_VERSION` constant in `config.py`.
- A unit test that renders the prompt for one metadata record and asserts all five parts and the indicator's wording, knowledge and ID are present.

### Task 6 — Evaluation scaffold

Implement under `pipeline/eval/` (paper Section 5.1.1). No model is evaluated in this phase; build the format, loader, metrics and tests.

1. **Gold-set format** — `pipeline/eval/gold/README.md` documents it; one CSV per labelled document plus an index:
   - `gold/index.csv`: `doc_id, source_file, language, has_logframe, indicators_in_other_document (name or empty), labelled_by, verified (true/false), split (dev/test)`
   - `gold/<doc_id>.csv`, one row per **document indicator**: `doc_indicator_id, indicator_text, level, baseline, target, unit, location, indikit_id (empty = no match), match_type (verbatim | close | none)`
   - IndiKit indicators not listed for a document count as **absent** for that document.
2. `gold.py`: loader + validator (IDs exist in the metadata, enums valid, no duplicate `doc_indicator_id`, `match_type=none` ⇔ empty `indikit_id`). Exposed as `python -m pipeline.run validate-gold`.
3. Add a **template** gold file with 3–4 clearly fictional example rows (`gold/_example.csv`) and an empty `index.csv` with headers. **Do not invent labels for real documents**; a MEAL colleague supplies and verifies those.
4. `metrics.py`:
   - `acc_dc` — disclosure accuracy over all (document, IndiKit indicator) pairs (paper Eq. 3);
   - `precision_recall_present` — precision, recall, F1 of the "present" class (needed because most pairs are absent);
   - `acc_de` — over truly present pairs, share with correct baseline **and** target after normalisation (paper Eq. 4); implement and test the value normaliser (`"75 %"`, `"75%"`, `"0,75"`, `"24.397,00"`, `"24,397"` etc.);
   - `match_type_accuracy` — over correctly found pairs;
   - `reference_detection_accuracy`.
   Each function takes plain data structures (lists of Pydantic records), not files.
5. `run_experiments.py`: stub with the experiment names from `plan.md` Section 6.3 (`benchmark`, `enhanced_rag`, `plus_knowledge`, `plus_search_term`, `model_comparison`, `gate_sensitivity`) and a results-table writer; bodies raise `NotImplementedError("Phase 3")`.

**Accept when:** metrics have unit tests with hand-computed expected values, including edge cases (no present pairs, all absent, empty prediction); `validate-gold` passes on the example file and fails with a clear message on a deliberately broken fixture.

### Task 7 — Notebook, requirements, docs, tests

1. **Notebook:** delete cells 17–34 (the duplicated Phase 4/5 block). Replace with one markdown cell pointing to `docs/plan.md` and the pipeline CLI, and one code cell that loads `indikit_metadata.jsonl` into a DataFrame and shows counts per sector and level. Make the reference-library path point to the generated `reference_library.csv`. Do not execute cells that call the OpenAI API; clear all outputs.
2. **Requirements:** keep one root `requirements.txt` with pinned versions; add only what this phase needs (e.g. `pyyaml`, `requests`, `beautifulsoup4` if the scraper is reused). Note Phase 2/3 additions as comments, not installs.
3. **README.md (root):** rewrite to describe the three-module architecture, the phase status table, how to run `build-metadata`, `validate-gold` and the tests, the data sources and their quirks (latin-1, repeated rows, export-only and site-only indicators), and a security note (revoke the old OpenAI key; `.env` is git-ignored).
4. **Tests** (`tests/`): add at least
   - `test_metadata_build.py` — merge logic on small synthetic fixtures (repeated-row collapse, conflict rule, export-only flag, quantity derivation, determinism);
   - `test_metadata_real.py` — runs against the real files if present (skip otherwise): unique IDs, expected count, required fields;
   - `test_extensions.py` — baseline search terms / knowledge, overrides precedence, no network (monkeypatch the HTTP/LLM client to raise if called);
   - `test_prompts.py`, `test_gold.py`, `test_metrics.py`.
   Existing `test_extraction_utils.py` must still pass.

---

## 4. Out of scope for Phase 1

- Any document parsing, chunking, OCR, table reconstruction, embeddings, vector index or retrieval (Phase 2).
- Any LLM call on proposal documents, the novel-indicator pass, experiments, the full run, or Streamlit changes (Phase 3).
- Labelling real documents for the gold set.
- Changing the Streamlit app, other than not breaking its imports.

---

## 5. Definition of done

- [ ] Git initialised; clean history with logical commits (clean-up → skeleton → metadata → extensions → eval scaffold → docs); no secrets or virtualenvs tracked.
- [ ] `pip install -r requirements.txt` in a fresh environment, then `pytest -q` passes with no network access and no API key set.
- [ ] `python -m pipeline.run build-metadata` produces `indikit_metadata.jsonl`, `reference_library.csv`, `outputs/metadata/build_report.md`, `outputs/metadata/merge_conflicts.csv`.
- [ ] `python -m pipeline.run validate-gold` works on the example gold file.
- [ ] `extraction_utils.py` is a shim; no duplicate `LLMClient`; no dummy-library fallback.
- [ ] Notebook has no Phase 4/5 duplicate cells and no stored outputs.
- [ ] README updated.

## 6. Final report (write to `docs/phase_1_report.md`)

1. What was done per task, with file paths.
2. Metadata numbers: total records, on-site / export-only / site-only, conflicts found, records missing wording or purpose, quantity-type distribution.
3. Whether the seven PIP indicators were fetched or seeded.
4. Override IDs that were skipped because names did not match.
5. Discrepancies between this brief and the repository.
6. Anything deferred, and open questions for the repository owner (at minimum: which source should win on conflicts; whether export-only indicators should be searched for in documents).