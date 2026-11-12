# Phase 3 report — LLM agent, novel-indicator pass, evaluation, runs, review, app integration

Brief: `docs/phase_3.md`. Method mapping: `docs/method.md`. Date: 2026-11-12.

**Headline.** Everything in the brief is built and tested offline (`pytest -q`: 199 passed in the project venv, plus the
12 Streamlit smoke tests in the system Python). **No live LLM call was made**: the brief's Section 2 says to stop before
any live call when the provider, models and budget have not been set by the repository owner, and they have not
(`config.LLM_PROVIDER` is empty; an `OPENAI_API_KEY` exists in `.env` but the data-policy decision - which service may
process ZOA documents - is still open, see Section 10). The gold set has **0 verified documents**, so the experiments
and the gate tuning are skipped as instructed; every accuracy figure in this report is therefore absent, not estimated.

## 1. What was built per task

| Task | Files | Notes / deviations |
|---|---|---|
| 1 Provider layer | `pipeline/agent/llm.py` | `LLMProvider.complete_json`, `OpenAIProvider` (also serves a local Ollama server through `base_url`), `AzureOpenAIProvider`, `MockLLMProvider`; `UsageTracker` (limits, per-model tokens / cost, `usage.json`, resume), `RateLimiter`, `run_parallel` (bounded concurrency), `ResponseCache`, `strict_json_schema`. Model roles `ANSWER_MODEL` / `STRONG_MODEL` / `EXTRACTION_MODEL` and the budget live in `pipeline/config.py` and can be overridden by environment variables. Keys are read from `.env` at call time only. |
| 2 Rubric | `docs/rubric.md`, `pipeline/agent/rubric.py` | definitions, nine worked examples (all IDs verified against the metadata: 5179, 5181, 5138, 5260, 5129, 5239, none dropped; 5239 is export-only), one-to-many rule; loaded between markers into the prompt; `RUBRIC_VERSION` in every cache key |
| 3 Answering | `pipeline/agent/prompts.py`, `answer.py`, `models.py` | five parts (part 1 = system message), `[E1] (page, section)` labels, question from the entity, multi-match `DisclosureAnswer` (`PROMPT_VERSION` bumped to `disclosure-v2`), post-validation (label exists, quote and indicator text found in that evidence with fuzzy tolerance, values normalised, label mapped back to entry / table / row / blocks), `answer_pair` with cache key (doc SHA-256, ID, evidence fingerprint, prompt + rubric version, model, knowledge flag), `answer_pairs` with a resumable progress file |
| 4 Pair selection | `pipeline/agent/select.py` | hard skips, gate (tau or top-M), optional sector routing (`--route-sectors`, off), export-only switch, `selected_pairs.csv` with reasons + per-document counts, `gate_recall` / `tune_gate` (dev split) |
| 5 Novel pass + consolidation | `pipeline/agent/novel_pass.py`, `consolidate.py` | code-first rows from logframe tables with an indicator column (cells split on bullets / numbering / line breaks; activities flagged by a verb-first heuristic), LLM only for text sections and unmapped tables (batched by section / table, `is_indicator` flag in the same call), dedupe with `deduplicate_indicators`, stable `D001..` ids; linking by row / block / text similarity, primary + alternates, reverse check with top-10 nearest IndiKit indicators |
| 6 Outputs | `pipeline/agent/outputs.py`, `orchestrate.py` | all six files of the brief, `needs_review` rule, "indicators are in another document: <name>" note, portfolio tables (by country / sector, most / never used, novel clusters by greedy cosine clustering in code), `run_report.md` with config fingerprint, versions, counts, tokens, cost, runtime, extrapolation; `report --doc` prints the three-column table |
| 7 Evaluation | `pipeline/eval/run_experiments.py`, `metrics.py` | scoring (per document and overall, with n), seven experiments, `results.md` (Table-4-like, Figure-3-like, gate, reverse check), `figures/ablation.png`, `errors.csv` with the seven categories and a top fix per category; `--allow-unverified` watermarks every output "UNVERIFIED GOLD" |
| 8 Runs | `pipeline/run.py` (`select-pairs`, `estimate`, `answer`, `novel-pass`, `consolidate`, `run --docs/--pilot/--all --confirm-full-run`, `report`, `experiments`, `export-review`, `import-review`) | the estimate is printed before every run and refuses to start above the limits; runs are resumable (`answers.jsonl`, `docs/<doc_id>.json`, `usage.json`); a failed document is recorded and the run continues; the pilot selection and the code-first novel pass were executed on real documents (no LLM) |
| 9 Review | `pipeline/review/export.py`, `import_.py` | `review.xlsx` one sheet per country, decision validation, `review_decisions.csv`, `document_indicators_reviewed.csv` (raw output unchanged), `overrides_proposed.yaml` + `.diff` (never applied), gold export with `verified=true` |
| 10 Streamlit | `zoa_indikit_app/src/{config,results,proposals,embeddings,loader}.py`, `pages/8_Proposal_Overview.py`, `pages/10_Indicator_Matching.py` | real per-document data with a demo banner fallback, map by country (world scope when Syria / Yemen are present), per-document three-column table with evidence on expand and IndiKit links, coverage views, novel clusters, documents with indicators elsewhere, downloads; `embeddings.py` uses the shared multilingual model through fastembed; `loader.py` follows the one-line pointer file so the existing pages load; no LLM calls (asserted by a test) |
| 11 Notebook, docs, tests | `proposal_indicator_analysis.ipynb`, `README.md`, `docs/method.md`, tests | notebook is a thin reader of `outputs/latest/`; matching v1 (`match_indicators_to_library`, `classify_match_status`, `_best_lexical_match`, `THRESHOLDS`) removed from `extraction_utils.py` and its tests; new tests: `test_llm`, `test_answer`, `test_select`, `test_novel_pass`, `test_consolidate`, `test_outputs` (incl. the end-to-end synthetic run), `test_experiments`, `test_review`, `test_app_smoke`; shared fixture `tests/fixtures_agent.py` |

Deviations from the brief and why:

- **Ollama as a third provider.** The brief names OpenAI and Azure OpenAI. The repository's existing default
  (`DEFAULT_LLM_PROVIDER = "ollama"`, `llama3.2`) speaks the OpenAI protocol, so `OpenAIProvider(base_url=...)` serves it;
  it is the only option that keeps the documents on the machine. It is not enabled by default either.
- **`disclosure = present <=> matches non-empty` is coerced, not rejected**, so that a model answer listing matches
  under "absent" does not burn a retry; the evidence check decides what counts.
- **Gate score.** The pair scores were computed without the re-ranker (`--no-rerank`, 37 s instead of ~1 h on CPU and
  no model download); the gate falls back to the vector score automatically and says so (`score_kind`).
- **Sector routing threshold** is the 75th percentile of all scores (`SECTOR_ROUTE_PERCENTILE`), independent of
  tau, because with tau at p90 no sector would pass.
- **Legacy `outputs/api_cache/`** is only consulted when the prompt version is the legacy `v1`; the Phase 3
  extraction prompt is `extract-v2` (batched by section, with the rubric), so the old cache is effectively unused.
- **Reference-name comparison** in `reference_detection_accuracy` was made lenient (containment or the same annex
  letter) because Phase 2 names references like "Annex C (indicator" while a label says "Annex C - Logical Framework".
- Phase 2 left no `docs/phase_2_report.md` (the README refers to one); the counts in Section 2 of the brief were
  taken from the manifest instead.

Bugs found and fixed on the way: `TableRow.cells` is stored with sorted keys, so reading cells by dict position
(as the first version of the code-first extraction did) returned the wrong column - cells are now looked up by
header name; concurrent workers shared one temp file when writing `usage.json` (now per-thread temp names);
repeated header rows inside a table body were extracted as indicators.

## 2. Configuration used

| Item | Value |
|---|---|
| Provider | none configured (`LLM_PROVIDER = ""`); `openai`, `azure`, `ollama` implemented |
| Models | `ANSWER_MODEL = llama3.2` (repository default), `STRONG_MODEL` empty, `EXTRACTION_MODEL = llama3.2` - owner fills in |
| Budget | `MAX_CALLS_PER_RUN = 1500`, `MAX_COST_USD_PER_RUN = 5.0`; prices for gpt-4o(-mini), gpt-4.1(-mini/-nano), o4-mini, llama3.2 (0) |
| Prompt / rubric versions | `disclosure-v2`, `rubric-v1`, `extract-v2`, `reverse-v1` |
| Retrieval | top-N 5, top-k 4, re-rank on (default `jinaai/jina-reranker-v2-base-multilingual`), search terms on, logframe boost 0.05 |
| Gate | tau = 90th percentile of all pair scores = **0.620** (vector scores), M = 60, routing off, export-only indicators included (719 of the 7 016 selected pairs) |
| Concurrency | 4 workers, 120 calls / min |

## 3. Gold set

`pipeline/eval/gold/index.csv` has no entries; only the fictional `_example.csv` exists. Verified documents: **0**
(dev 0, test 0). The 11 hand-analysed documents of 10 Nov 2026 (docs/plan.md Section 6.1) still have to be labelled
in the gold format and verified by a MEAL colleague. `import-review --export-gold` can produce gold files from
reviewed pipeline output once a pilot has run.

## 4. Results

Not available: no verified gold, no live run. The evaluation code was exercised with the mock provider on two
synthetic documents and fictional labels (`tests/test_experiments.py`): all seven experiments write their
`metrics.json` / `predictions.csv`, `results.md` carries the Table-4-like model table and the Figure-3-like ablation
table, `figures/ablation.png` is produced, and the error analysis assigns every wrong pair to one of the seven
categories. The paper's figures (GPT-4: Acc_DC 76.9 % ...) remain context, not a target.

## 5. Error analysis

Not run (no gold). The categories, the counting and the per-category top fixes are implemented; the changelog of
targeted improvements is empty.

## 6. Cost and runtime

Measured without an LLM:

| Step | Measured |
|---|---|
| `build-kb --all` (91 remaining documents, fastembed on CPU) | 1 167 s; 82 ok, 9 partial, 0 failed (120 documents incl. 5 templates) |
| `score-pairs --all --no-rerank` | 42 120 pairs in 37 s |
| `estimate --all` | 115 documents, 7 skipped (indicators elsewhere), **7 016 pairs**, ~170 novel-pass calls, ~920 reverse-check calls = **~8 100 calls**; **$3.14 with gpt-4o-mini** (prompt ~1 800 / completion ~220 tokens per pair), $0 local |
| `select-pairs --pilot` | 5 documents (Burkina docx, DRC logframe docx, Ethiopia xlsx, Syria pdf, Yemen docx), 333 pairs, ~387 calls |
| Cheaper settings | p95 / M = 40: 4 519 pairs (~5 600 calls); routing at the defaults saves almost nothing (7 012 pairs) because top-M dominates |

Per pair and per document: not measured (no live call). With gpt-4o-mini the estimate is ~$0.00045 per pair and
~$0.03 per document; a local llama3.2 on this Intel Mac would be slow (tens of seconds per call).

The full-run estimate stays under `MAX_COST_USD_PER_RUN` for gpt-4o-mini but **exceeds `MAX_CALLS_PER_RUN = 1500`**
by a factor of five. As instructed the limit was not raised. Options for the owner: raise `MAX_CALLS_PER_RUN` to
~9 000 (cost ~$3.1 with gpt-4o-mini), or run country by country with `--docs`, or tighten the gate (p95 / M = 40
-> ~5 600 calls) after the gate curve has been measured on gold.

## 7. Portfolio findings

Only structural findings are available (no run): 115 documents (south_sudan 20, uganda 19, sudan 17, burundi 14,
syria 13, Ethiopia 11, yemen 10, DRC 9, Burkina 2); 67 with logframe content, 48 without; **7 documents point to
another document for their indicators** and are reported, not queried (Syria joint-response narrative, two DRC
FS-IWRM / TRIDE files, two South Sudan proposals, SIPRA Sudan, ARFSA Uganda); 33 have both. The code-first novel pass
on the pilot found 83 document indicators in the DRC RAISE logframe and 58 in the Ethiopia EU logframe (99 and 73
rows before de-duplication) without any LLM call.

## 8. Reverse-check yield

Not measured. It is implemented as a separate experiment (`reverse_check_value`) and reported per run
(`reverse_check_candidates` / `reverse_check_recovered` in `run_report.md`).

## 9. Known limitations and next steps

Limitations: no measured accuracy; vector-only gate scores; the code-first activity filter is a heuristic (verb-first
lines) - the LLM flag applies only to text sections; the novel-cluster threshold (0.80 cosine) is untuned; text-only
logframes (e.g. the Syria narrative with six logframe sections) need the LLM extraction and are empty in a no-LLM run;
`partial` PDFs (nine, mostly copies of the same "We are able" proposal) need OCR; the Streamlit pages require
Streamlit >= 1.49 (the existing pages use `width="stretch"`; the smoke test shims older versions).

Three most valuable next steps:

1. **Verify gold for 8-11 documents** (dev) and run `experiments --split dev`: this unlocks the gate tuning, the
   ablations and the error analysis, which decide every other parameter.
2. **Run the pilot with the chosen provider** (`run --pilot --copy-latest`, ~390 calls), read the five output
   files, and let MEAL staff review `review.xlsx`; import the decisions to grow the gold set and the overrides.
3. **Compute re-ranked pair scores** (`score-pairs --all`, ~1 h) and re-tune the gate; then decide between a
   stricter gate and sector routing for the full run.

## 10. Decisions needed from the repository owner

1. **Provider and data policy**: may ZOA documents (partner names, budgets) be sent to OpenAI, to Azure OpenAI, or
   only to a local model? Set `LLM_PROVIDER` and the keys accordingly; nothing runs until then.
2. **Models**: `ANSWER_MODEL` / `EXTRACTION_MODEL` (gpt-4o-mini is the cheapest priced option) and `STRONG_MODEL`
   for the comparison.
3. **Budget for the full run**: raise `MAX_CALLS_PER_RUN` (~9 000 needed at the default gate) or approve a stricter
   gate / batched runs; `MAX_COST_USD_PER_RUN = 5` covers gpt-4o-mini.
4. **Export-only indicators**: keep asking the 41 indicators that are no longer on the site (default) or set
   `INCLUDE_EXPORT_ONLY = False`.
5. **Overrides**: review `outputs/latest/overrides_proposed.yaml` after each `import-review` before merging.
6. **Gold labels**: who verifies the 11 hand-analysed documents, and the dev / test split.
