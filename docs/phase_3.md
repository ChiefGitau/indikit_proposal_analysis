# Phase 3 of 3 — LLM agent, novel-indicator pass, evaluation, full run, app integration

You are working in the `proposal_investigated` repository. This is the last of three phases that transform the project to the method of *ESGReveal* (Zou et al., 2023, arXiv:2312.17264). Read `docs/plan.md`, `docs/phase_1_report.md`, `docs/phase_2_report.md` and `docs/interfaces.md` first; this file is the binding scope for Phase 3.

| Phase | Scope | Status |
|---|---|---|
| 1 — Foundation | Repo clean-up, `pipeline/` skeleton, metadata module, evaluation scaffold | Done |
| 2 — Preprocessing | Parsers, outline, references, three knowledge bases, retrieval + re-ranking, baseline index, pair scores | Done (verify, Section 2) |
| **3 — Agent & evaluation** | LLM answering with the five-part prompt, pair selection (routing + gate), novel-indicator pass, experiments (benchmark, ablations, models), full run, review workflow, Streamlit integration | **This task** |

This is the first phase that calls an LLM on ZOA documents. Section 3 sets hard limits on that.

---

## 1. What this phase implements

Paper Section 3.4.2 (*LLM Answering*), Section 5.1 (accuracy indicators, LLM benchmarking, ablation study), and the application of the method to the whole document set (paper Section 5.2), plus one addition the paper does not have: a pass that finds document indicators with **no** IndiKit match.

End state: one command processes a document and produces (a) which IndiKit indicators it contains, as verbatim or close, with wording, baseline, target and evidence; (b) its indicators that are not in IndiKit; (c) where its indicators are held if in another document. Accuracy is measured on the gold set, against a plain-RAG baseline, with ablations. Results are reviewable by MEAL staff and visible in the Streamlit app.

---

## 2. Preconditions (check before writing code)

Confirm each; stop and report if one fails and the fix is not trivial.

- `pytest -q` passes offline.
- `outputs/kb/manifest.csv` covers all documents; note counts of `ok / partial / failed` and the list of documents with no logframe found (from `docs/phase_2_report.md`).
- `retrieve`, `retrieve_baseline`, `logframe_content`, `load_references`, `load_kb` import and work as described in `docs/interfaces.md`.
- `outputs/kb/pair_scores.*` exists.
- Gold set: run `python -m pipeline.run validate-gold` and count documents with `verified=true`, by split (`dev` / `test`).
  - **≥ 8 verified** documents: run everything in this brief.
  - **1–7 verified:** run everything, but mark every accuracy figure "preliminary (n = …)" in outputs and reports.
  - **0 verified:** build and unit-test everything, run the pilot (Task 8 step 1) only, **skip** Task 7's experiments and the threshold tuning (use the defaults given), and say so prominently in the final report. Never invent or silently use unverified labels; an `--allow-unverified` flag may exist for debugging and must watermark its outputs "UNVERIFIED GOLD".
- LLM access: the provider, model names and budget are set by the repository owner in `pipeline/config.py` / environment (Section 3). If they are not set, implement and test with the mock client and stop before any live call.

Where earlier phases named things differently from this brief, follow the repository and list the differences in the final report.

---

## 3. Rules for LLM use

1. **Secrets:** the key is read from the environment / `.env` at runtime only. Never print, log, cache or commit it. Never write it into notebooks, test fixtures or reports.
2. **Provider is configurable, not assumed.** Implement `LLMProvider` with two back ends selected by config: OpenAI and Azure OpenAI (endpoint, deployment and API version from environment). ZOA documents contain partner names and budgets; which provider may process them is the owner's decision. Do not send documents to any service not named in the config.
3. **Budget is enforced in code.** Config values `MAX_CALLS_PER_RUN`, `MAX_COST_USD_PER_RUN` (owner-set; ship conservative defaults) and per-model price entries. Every run prints an estimate (pairs × average tokens) **before** starting, tracks actual tokens and cost, and stops cleanly when a limit is reached. Runs are **resumable** (cache + progress file).
4. **A full run over all documents requires the explicit flag `--confirm-full-run`.** Without it, runs are limited to the pilot size.
5. **No live LLM calls in tests.** All tests use `MockLLMProvider` with scripted responses. Live calls happen only through CLI commands.
6. **Determinism and caching:** temperature 0; cache every response under `outputs/cache/agent/` keyed by (document SHA-256, IndiKit ID, evidence fingerprint, prompt version, model). A cached answer is never re-requested unless `--force`.
7. **Structured output only:** request JSON conforming to the schema (use the provider's schema-enforced mode where available), validate with Pydantic, retry once on invalid output with the validation error appended, then record a failure. Never parse free text.
8. **Counts and summaries are computed in code**, never by the LLM.
9. Do not modify `proposals/`, the raw IndiKit files, or gold labels. Commit logically (provider → answering → selection → novel pass → consolidation → eval → run → review → app → docs).

---

## 4. Tasks

### Task 1 — LLM provider layer

`pipeline/agent/llm.py`

- `LLMProvider` interface: `complete_json(system: str, user: str, schema: type[BaseModel], *, model: str) -> LLMResult` where `LLMResult` holds the parsed object, raw text, prompt/completion tokens, latency, cached flag.
- `OpenAIProvider`, `AzureOpenAIProvider`, `MockLLMProvider`.
- Reuse the retry/backoff logic moved in Phase 1; add rate limiting and bounded concurrency (config, default 4 workers).
- `UsageTracker`: accumulates calls, tokens, cost per model; enforces the limits in Section 3; writes `outputs/runs/<run_id>/usage.json`.
- Model roles in config, not literals in code: `ANSWER_MODEL` (default: the repository's existing default model), `STRONG_MODEL` (for the model comparison), `EXTRACTION_MODEL` (novel pass). The owner fills in the names.

### Task 2 — Match rubric

`docs/rubric.md` + `pipeline/agent/rubric.py` (loads the text for prompts; versioned with `RUBRIC_VERSION`).

Definitions:

- **verbatim** — same measurement and (nearly) identical wording; differences limited to targets, placeholders filled in, or word order.
- **close** — same thing measured, with different wording, unit (number vs percentage), population or scope.
- **absent / none** — nothing in the evidence measures the same thing. Same *topic* is not enough.
- **not an indicator** — activities, deliverables, assumptions, or targets with no measure.

Seed the rubric with these worked examples (verify each ID's name in the metadata before using it; drop any that does not match and report it):

| Document text | IndiKit | Label | Why |
|---|---|---|---|
| "Number of people having access to basic, safe and dignified shelter solutions" | 5179 Access to Adequate Shelter | verbatim | Same measure, same wording |
| "Number of local people involved in voluntary or paid work on shelter (re)construction" | 5181 Local Participation in Shelter Construction | verbatim | Same measure, same wording |
| "% of households with soap and water at a handwashing station" | 5138 Handwashing Facilities With Soap and Water | verbatim | Same measure |
| "75% of trained PIP farmers report increased yields" | 5260 Agricultural Productivity | close | Same outcome, narrower population, self-reported |
| "Number of functional water points constructed or rehabilitated" | 5129 Water Source Functionality | close | Counts water points; IndiKit measures share functional |
| "Number of people receiving multi-purpose cash" | 5239 Number of Cash Transfer Recipients | close | Same count, restricted to one modality |
| "5 VSLAs supported" | — | none | Activity count; self-help-group indicators measure functionality/coverage, not groups supported |
| "70% reduction in seasonal migration" | — | none | No IndiKit indicator measures migration |
| "Train 30 farmers on climate-smart agriculture" | — | not an indicator | An activity |

Add the rule for one-to-many cases: a document indicator may match more than one IndiKit indicator, and one IndiKit indicator may match several document indicators; record all, mark one as primary (Task 5).

### Task 3 — LLM answering (paper 3.4.2)

`pipeline/agent/answer.py`, `pipeline/agent/prompts.py`, `pipeline/agent/models.py`

**Prompt** — exactly the paper's five parts, built by `build_disclosure_prompt` from Phase 1 (extend as needed, bump `PROMPT_VERSION`):

1. **Preset information** — role; answer only from the reference content; do not use outside knowledge; an activity is not an indicator; if unsure, answer absent.
2. **Reference content** — the top-k `Evidence.display_text` items from `retrieve`, each prefixed with a label `[E1] (page 12, Logical Framework > Outcome 2)`.
3. **Expert knowledge** — the indicator's `knowledge` field + the rubric text. Controlled by flag `use_knowledge` (needed for the ablation).
4. **Question** — built from the metadata entity: sector/sub-sector (Aspect), name and wording (KPI), disaggregation (Topic), measurement type (Quantity): "Does the reference content contain one or more indicators that measure this? For each, give the exact wording and its values."
5. **Answer format** — the schema below.

**Answer schema** (extend the Phase 1 `DisclosureAnswer`; one IndiKit indicator can match several document indicators):

```json
{
  "indikit_id": "5239",
  "disclosure": "present | absent",
  "matches": [
    {
      "document_indicator_text": "exact wording copied from the reference content",
      "match_type": "verbatim | close",
      "level": "impact | outcome | output | unclear",
      "baseline": "string or null",
      "target": "string or null",
      "unit": "string or null",
      "evidence_label": "E2",
      "evidence_quote": "short exact quote from that evidence",
      "reason": "one sentence"
    }
  ]
}
```

`disclosure = present` ⇔ `matches` non-empty.

**Post-validation (in code, after every answer)**
- `evidence_label` must exist; `evidence_quote` and `document_indicator_text` must be found in that evidence's text (whitespace/case-insensitive, small fuzzy tolerance). If not → mark the match `unsupported`, exclude it from results, count it in the run report. This is the guard against invented indicators.
- Normalise baseline/target with the Phase 1 value normaliser, keeping the raw string too.
- Map `evidence_label` back to `entry_id`, page/section, table/row IDs.

**Function:** `answer_pair(doc_id, indicator, *, retrieval_cfg, use_knowledge=True, model=ANSWER_MODEL) -> PairAnswer` (answer + evidence + usage + flags).

Tests (mock provider): prompt contains the five parts; flags change the prompt as intended; invalid JSON triggers one retry; unsupported quotes are dropped; caching prevents a second call.

### Task 4 — Pair selection: sector routing and retrieval gate

`pipeline/agent/select.py`

Asking every indicator of every document is too expensive (≈ documents × ~350 indicators). Select pairs before answering:

1. **Hard skips:** documents with `parse_status = failed`; documents with `indicators_elsewhere = true` and no logframe content (they are reported via Task 6, not queried).
2. **Retrieval gate:** from `pair_scores`, keep a pair if `best_rerank_score ≥ τ` **or** the indicator is among the document's top-`M` by score. Defaults when no verified gold exists: `M = 60`, `τ` = the 90th percentile of all pair scores (state the value used). With verified gold, tune `τ` and `M` on the **dev** split so that ≥ 95% of gold-present pairs pass the gate, then pick the cheapest setting meeting that; record the trade-off curve (Task 7, `gate_sensitivity`).
3. **Sector routing (optional, flag `--route-sectors`, default off until evaluated):** derive each document's sectors without an LLM — aggregate `pair_scores` by IndiKit sector and keep sectors whose top scores exceed a threshold — and restrict the gate's top-`M` rule to those sectors. Evaluate its effect on recall before enabling.
4. Output `outputs/runs/<run_id>/selected_pairs.csv` with the reason each pair was kept, and a per-document count. Print the cost estimate from this file.

Indicators flagged `on_site = false` (export-only) are included by default with a column marking them; make this a config switch and report how many matches come from them.

### Task 5 — Novel-indicator pass and consolidation

`pipeline/agent/novel_pass.py`, `pipeline/agent/consolidate.py`

**A. Extract the document's own indicators** (document-driven; not in the paper)
- Input: `logframe_content(doc_id)` only — logframe tables (using `column_roles`: indicator, baseline, target, MoV, result context) and logframe text sections.
- For logframe tables with an `indicator` column, extract rows **in code first** (no LLM): each non-empty indicator cell becomes a candidate, with baseline/target from the mapped columns. Split cells containing several indicators (bullets, numbering, line breaks).
- Use the LLM (`EXTRACTION_MODEL`, existing extraction prompt as a starting point, schema-enforced, level restricted to the enum) only for logframe **text** sections and for tables whose columns could not be mapped. Batch by section/table, not by paragraph.
- Filter out activities using the rubric's "not an indicator" rule (LLM flag in the same call; no extra call).
- De-duplicate within the document (reuse `deduplicate_indicators`). Assign stable `doc_indicator_id`s.
- The old `outputs/api_cache/` may be used as a cache only where chunk content and prompt version are identical; otherwise ignore it.

**B. Consolidate** disclosure results with extracted indicators into one list per document
- Link each disclosure match to an extracted document indicator by location (same table row / block) and text similarity. A match with no extracted counterpart creates a new document indicator (source `disclosure_only`).
- **Primary match:** when a document indicator has several IndiKit matches, primary = verbatim over close, then higher re-rank score; the others are kept as `alternate_ids`.
- Document indicators with no IndiKit match → candidates for **novel**.

**C. Reverse check** (safety net for the gate; counts toward the budget)
- For each novel candidate: embed its text, take the top 10 IndiKit indicators from the metadata by similarity, and ask the LLM once, with the rubric and the candidates' full wording and knowledge, whether any is a verbatim or close match (allow none).
- A hit becomes a match with `source = reverse_check` and is added to the disclosure results. What remains is **novel** (`match_type = none`).
- Report how many matches were recovered this way: it is a direct measure of what the gate/retrieval missed.

Tests (mock provider, synthetic documents): code-first row extraction incl. multi-indicator cells; linking; primary/alternate selection; reverse check adds and labels a recovered match.

### Task 6 — Outputs

Written per run to `outputs/runs/<run_id>/` and, for the latest full run, copied to `outputs/latest/`:

| File | One row per | Key columns |
|---|---|---|
| `document_indicators.csv` | document indicator | `doc_id, doc_indicator_id, indicator_text, level, baseline, target, unit, location, match_type (verbatim / close / none), indikit_id, indikit_name, alternate_ids, source (disclosure / reverse_check / extraction), reason, evidence_quote, needs_review` |
| `disclosure_matrix.csv` | document × present IndiKit indicator | `doc_id, indikit_id, indikit_name, sector, match_type, n_document_indicators, doc_indicator_ids` |
| `novel_indicators.csv` | novel document indicator | text, level, document, country, nearest IndiKit indicators (IDs + similarity) |
| `document_summary.csv` | document | country, region, language, parse status, total indicators, verbatim, close, none, % matched, `indicators_elsewhere`, referenced documents, pairs asked, LLM calls, cost |
| `portfolio_summary.md` + CSVs | — | totals; by country; by IndiKit sector; most-used IndiKit indicators; IndiKit indicators never used; clusters of similar novel indicators (embedding clustering in code) as candidates to add to IndiKit |
| `run_report.md` | — | config fingerprint, models, prompt/rubric versions, pairs selected/asked/cached, unsupported answers dropped, failures, tokens, cost, runtime |

`needs_review = true` for: every close match; reverse-check matches; matches with alternates; verbatim matches whose text similarity to the IndiKit wording is low.

For documents with `indicators_elsewhere`, `document_summary.csv` states "indicators are in another document" with the document name(s) from `references.json` — this is a required output of the project.

Also provide `python -m pipeline.run report --doc DOC_ID` that prints, for one document, a Markdown table in this shape (the format MEAL staff already use):

```
| Indicator in document | IndiKit indicator (ID) | Match |
```

followed by the summary line (indicators in document, verbatim, close, total matched, referenced document if any).

### Task 7 — Evaluation and experiments (paper 5.1)

Fill `pipeline/eval/run_experiments.py`. Skip this task's live runs if there is no verified gold (Section 2); still implement and unit-test the code with the mock provider and the fictional example gold.

**Scoring:** predictions vs gold with the Phase 1 metrics — `acc_dc`, precision / recall / F1 of "present", `acc_de`, `match_type_accuracy`, `reference_detection_accuracy` — plus novel-pass precision/recall (document indicators labelled `none` in gold) and extraction recall of document indicators. Report per document and overall, with n.

**Experiments** (each writes `outputs/eval/<experiment>/metrics.json` + predictions; all on the same documents and the same selected pairs unless the experiment is about selection):

| Experiment | Configuration | Paper equivalent |
|---|---|---|
| `benchmark` | `retrieve_baseline` (flat chunks, MiniLM, no re-rank), no knowledge, no search terms | "Benchmark" basic RAG |
| `enhanced_rag` | Phase 2 retrieval (three KBs + re-rank), no knowledge, no search terms | "Enhanced-RAG" |
| `plus_knowledge` | enhanced + `use_knowledge=True` | + ESG knowledge |
| `plus_search_term` | enhanced + knowledge + `use_search_terms=True` (**full method**) | Full ESGReveal |
| `model_comparison` | full method with `ANSWER_MODEL` vs `STRONG_MODEL` (and any further models the owner configured) | Table 4 |
| `gate_sensitivity` | recall of gold-present pairs and number of pairs asked across `τ` / `M` values; with and without sector routing | (addition) |
| `reverse_check_value` | matches recovered and added cost | (addition) |

Protocol:
- Tune prompts, `τ`, `M`, top-k on **dev** only. Run the **test** split once, at the end, with the frozen configuration; report dev and test separately.
- Model comparison and ablations run on dev (to limit cost) unless the budget allows both.
- Produce `outputs/eval/results.md` with a table like the paper's Table 4 (models × Acc_DE, Acc_DC) and an ablation table/figure like its Figure 3 (matplotlib, saved to `outputs/eval/figures/`), each showing n and cost.
- **Error analysis:** `outputs/eval/errors.csv` listing every wrong pair with category — `retrieval_miss` (gold evidence not retrieved), `gate_miss` (pair not selected), `llm_false_negative`, `llm_false_positive`, `wrong_match_type`, `wrong_value`, `parse_problem` — and a short section in `results.md` with the counts per category and the top fixes.

Use the error analysis to make targeted improvements (search terms/overrides for specific indicators, prompt wording, parser fixes), then re-run dev. Keep a changelog of what was changed and its effect. Do not touch the test split while iterating.

### Task 8 — Runs

1. **Pilot:** 5 documents — prefer verified dev gold documents; otherwise the first 5 with logframes spread over countries and file types. Run end to end; check outputs by reading them; record cost per document and per pair.
2. **Extrapolate:** estimate calls, cost and time for all documents from the pilot; write it to the run report. If the estimate exceeds `MAX_COST_USD_PER_RUN`, do **not** raise the limit: propose a cheaper configuration (stricter gate, sector routing, smaller model) with its expected recall from `gate_sensitivity`, and stop for the owner's decision.
3. **Full run:** only with `--confirm-full-run` and within budget. Resumable; a failed document does not stop the run. Finish with `outputs/latest/` and `portfolio_summary.md`.

CLI (`pipeline/run.py`): `select-pairs`, `answer` (disclosure step), `novel-pass`, `consolidate`, `run --docs … | --pilot | --all --confirm-full-run`, `report --doc`, `experiments --name … --split dev|test`, `estimate`.

### Task 9 — Review workflow

`pipeline/review/`

- `export-review`: writes `outputs/latest/review.xlsx`, one sheet per country, rows with `needs_review = true`: document, document indicator, IndiKit name + wording + link, match type, reason, evidence quote, and empty columns `reviewer_decision (accept | change_to_verbatim | change_to_close | reject | other_id)`, `correct_indikit_id`, `comment`, `reviewer`.
- `import-review`: reads the filled file, validates decisions, writes `outputs/latest/review_decisions.csv`, applies them to produce `document_indicators_reviewed.csv`, and:
  - proposes additions to `pipeline/metadata/overrides.yaml` (knowledge notes and search terms) as a **separate diff file** for the owner to approve — do not edit overrides automatically;
  - can export reviewed documents as gold files in the Phase 1 format with `verified=true` and the reviewer's name, so the gold set grows from real use.
- Reviewer decisions always override pipeline output in the reviewed file; the raw output stays unchanged.

Unit-test the round trip with a small synthetic file.

### Task 10 — Streamlit integration

App: `MEAL_Proposal-main/MEAL_Proposal-main/zoa_indikit_app/`.

- `src/proposals.py`: read `outputs/latest/document_summary.csv` (path from config) instead of `data/proposals_dummy.json`; keep the dummy file as a fallback with a visible "demo data" banner when no run exists. Keep the existing columns the page expects; add the new ones.
- `pages/8_Proposal_Overview.py`: real per-proposal data; map by country; share matched.
- New page `pages/10_Indicator_Matching.py`:
  - filters: country, sector, match type, document;
  - per-document table in the three-column format (document indicator | IndiKit indicator (ID) | match), with evidence quote on expand and a link to the IndiKit page;
  - coverage views: IndiKit indicators by number of documents using them; indicators never used; sector × country heat map;
  - novel indicators: clusters and their documents;
  - documents whose indicators are in another document, with the document names;
  - download buttons for the CSVs and the review file.
- Point `src/embeddings.py` at the shared multilingual embedder and metadata file if not already done; do not break existing pages (run each page's import in a smoke test).
- The app must make **no** LLM calls and must load without the pipeline's optional dependencies.

### Task 11 — Notebook, docs, tests

- **Notebook:** `proposal_indicator_analysis.ipynb` becomes a thin walkthrough: load latest outputs, show one document's table, portfolio charts, and the evaluation table. No LLM calls in the notebook; outputs cleared.
- **README:** end-to-end usage (build-metadata → build-kb → score-pairs → run → export-review → app), configuration of provider/models/budget, how to add gold documents, how to read the outputs, limitations.
- **`docs/method.md`:** how this implementation maps to the paper (modules, prompt parts, metrics), what was adapted (IndiKit entities, match types, gate) and what was added (novel pass, reverse check, review loop), with the final accuracy and cost numbers.
- **Retire** deprecated matching code in `extraction_utils.py` (`match_indicators_to_library`, `classify_match_status`, `_best_lexical_match`, `THRESHOLDS`) and its tests, once nothing imports them; keep the shim only for names still used.
- **Tests** (offline, mock provider): `test_llm.py` (limits, retries, caching, usage tracking, resumability), `test_answer.py`, `test_select.py`, `test_novel_pass.py`, `test_consolidate.py`, `test_outputs.py` (schemas, counts computed correctly), `test_experiments.py`, `test_review.py`, `test_app_smoke.py`. One end-to-end test on a synthetic docx + tiny metadata + mock LLM producing all output files.

---

## 5. Out of scope

- Changing the gold labels, or labelling documents yourself.
- Fine-tuning models; heavy layout models (recommend them only if the error analysis shows parsing is the main error source).
- Deployment to SharePoint / Power Automate / Azure hosting; authentication for the app.
- Editing `overrides.yaml` from review results without owner approval.
- Sending documents to any provider not configured by the owner.

---

## 6. Definition of done

- [ ] `pytest -q` passes offline, no API key, no model download; end-to-end synthetic test included.
- [ ] `python -m pipeline.run estimate --all` prints pairs, calls and cost without calling the LLM.
- [ ] Pilot run completed (if LLM access is configured) with all output files; every reported match has a supporting quote found in the document.
- [ ] Budget limits demonstrably stop a run (test + one manual check); runs resume from cache.
- [ ] Experiments run on verified gold per Section 2, or explicitly skipped with the reason; `outputs/eval/results.md` and `errors.csv` written when run.
- [ ] Full run completed **only if** confirmed and within budget; otherwise the estimate and recommended configuration are in the report.
- [ ] `report --doc` prints the three-column table; documents with indicators elsewhere name the other document.
- [ ] Review export/import round trip works.
- [ ] Streamlit app shows real results (or the demo banner when none exist); all pages import.
- [ ] README, `docs/method.md` updated; deprecated matching code removed; no secrets, caches or outputs tracked by git.

## 7. Final report (write to `docs/phase_3_report.md`)

1. What was built per task, with file paths; deviations from this brief and why.
2. Configuration used: provider, models, prompt and rubric versions, gate values (`τ`, `M`), top-k, routing on/off.
3. Gold set used: documents, verified count, dev/test split.
4. Results: the model table and ablation table (dev and test), with n; comparison with the paper's figures only as context, not as a target.
5. Error analysis: counts by category, the changes made in response, their measured effect.
6. Cost and runtime: per pair, per document, pilot total, full-run estimate or actual.
7. Portfolio findings (if the full run was done): share of indicators matched overall, by country and sector; most-used and never-used IndiKit indicators; main novel-indicator clusters; documents pointing to other documents.
8. Reverse-check yield and what it says about the gate.
9. Known limitations and the three most valuable next steps.
10. Decisions needed from the repository owner (budget for the full run, provider, export-only indicators, overrides to approve).