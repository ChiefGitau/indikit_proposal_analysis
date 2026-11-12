# How this implementation maps to ESGReveal (Zou et al., 2023, arXiv:2312.17264)

Status 2026-11-12: all three modules are implemented and tested offline. **No accuracy or cost has been measured
on real documents yet** because the gold set has no verified document and no LLM provider is configured
(`docs/phase_3_report.md`, Sections 3 and 10). The numbers below are structural (documents, pairs, estimates).

## 1. Paper modules -> code

| Paper | Section | Code | Notes |
|---|---|---|---|
| Metadata module - Entities `<Aspect, KPI, Topic, Quantity>` | 3.2.1 | `pipeline/metadata/build_metadata.py`, `models.py` (`IndicatorMetadata`) | Aspect = sector / sub-sector, KPI = name + wording (EN/FR), Topic = disaggregation, Quantity derived from the wording |
| Metadata module - Extensions `<Knowledge>`, `<SearchTerm>` | 3.2.2 | `pipeline/metadata/extensions.py`, `overrides.yaml` | deterministic baseline, optional LLM enrichment, manual overrides; the review loop proposes additions |
| Metadata module - Expressions (prompt templates) | 3.2.3 | `pipeline/agent/prompts.py` | five-part prompt; `PROMPT_VERSION` in every cache key |
| Preprocessing - layout analysis, tables, outline | 3.3 | `pipeline/preprocess/parse_{pdf,docx,xlsx}.py`, `outline.py`, `tables.py`, `logframe.py`, `references.py` | PyMuPDF / python-docx / openpyxl instead of LayoutLMv3 + Table-Transformer |
| Multi-type knowledge base (text, outline, tables) | 3.3 | `pipeline/preprocess/knowledge_base.py` | three vector indexes per document (`outputs/kb/<doc_id>/`), multilingual MiniLM through fastembed |
| LLM agent - retrieval of knowledge (top-N per KB, re-rank, top-k) | 3.4.1 | `pipeline/agent/retrieve.py`, `query.py`, `rerank.py` | query = KPI + search terms; cross-encoder re-ranking; logframe boost |
| LLM agent - answering (preset, reference content, expert knowledge, question, answer format) | 3.4.2 | `pipeline/agent/prompts.py`, `answer.py`, `models.py` (`DisclosureAnswer`) | part 1 = system message, parts 2-5 = user message; evidence items labelled `[E1] (page, section)` |
| Evaluation - Acc_DC, Acc_DE, benchmark vs enhanced RAG, ablations, model comparison | 5.1 | `pipeline/eval/metrics.py`, `run_experiments.py` | + precision / recall / F1 of "present", match-type accuracy, reference detection, novel P/R, extraction recall |
| Application to the full report set | 5.2 | `pipeline/agent/orchestrate.py`, `outputs.py`, `pipeline/run.py run --all` | per-document and portfolio outputs; Streamlit pages 8 and 10 |

### Prompt parts (paper 3.4.2) as implemented

1. **Preset information** - role (MEAL specialist at ZOA), rules: answer only from the reference content, activities are
   not indicators, copy wording / quotes exactly, answer absent when unsure. Sent as the system message together
   with the indicator block (ID, name, wording EN/FR, aspect, levels, quantity, topic).
2. **Reference content** - the top-k `Evidence.display_text` from `retrieve`, each prefixed with `[E1] (page 12,
   Logical Framework > Outcome 2)`. Table rows stand alone (title, columns, carried result context, row).
3. **Expert knowledge** - the indicator's `knowledge` field + the rubric (`docs/rubric.md`, `RUBRIC_VERSION`).
   `use_knowledge=False` drops the knowledge note (ablation); the rubric always stays.
4. **Question** - built from the entity: "Does the reference content contain one or more indicators that measure
   *KPI* (*name*; *quantity*) for *sector / sub-sector*, disaggregated by *topic*? For each, give the exact wording and
   its values ...".
5. **Answer format** - JSON (`DisclosureAnswer`: `indikit_id`, `disclosure`, `matches[]` with text, match type, level,
   baseline, target, unit, evidence label, quote, reason), requested in the provider's JSON-schema mode and validated
   with Pydantic; one retry with the validation error; never parsed from free text.

### Metrics (paper 5.1.1)

| Metric | Definition here | Paper |
|---|---|---|
| Acc_DC | over all (document, IndiKit indicator) pairs of the gold documents, share with the right present / absent | disclosure accuracy |
| Acc_DE | over gold-present pairs, share predicted present with baseline **and** target equal after normalisation | extraction accuracy |
| precision / recall / F1 (present) | because most pairs are absent | addition |
| match-type accuracy | over correctly found pairs, verbatim vs close | addition |
| reference detection | documents whose indicators are elsewhere: predicted document name matches (lenient: containment or same annex letter) | addition |
| novel precision / recall | gold rows with `match_type = none` vs predicted novel rows (fuzzy text match) | addition |
| extraction recall | share of gold document indicators found by the novel pass | addition |

## 2. What was adapted

- **IndiKit entities instead of ESG standards.** 351 indicators (344 export + 7 site-only PIP), each with EN and FR
  wording, levels and disaggregation; 41 export-only indicators are kept (`on_site = false`) and asked by default
  (config `INCLUDE_EXPORT_ONLY`), marked in the outputs.
- **Match types.** The paper's answer is disclosed / not disclosed. IndiKit matching needs *verbatim* / *close* /
  *none* (and *not an indicator*), defined in `docs/rubric.md` with nine worked examples, plus a one-to-many rule
  (several matches recorded, one primary).
- **Multi-match answers.** One IndiKit indicator can match several document indicators (one table row often holds
  several), so `matches` is a list; `disclosure = present` iff the list is non-empty after validation.
- **Evidence post-validation.** Every match must quote text that is found (fuzzy, >= 88 partial ratio) in the labelled
  evidence item; otherwise it is `unsupported`, excluded and counted. This replaces the paper's implicit trust in the
  model and is the guard against invented indicators.
- **Pair selection (gate + routing).** The paper queries every standard indicator. Here documents x ~350 indicators
  would be ~40 000 calls, so a retrieval gate keeps a pair when its best score >= tau (default: 90th percentile of all
  pair scores) or it is in the document's top-M (default 60); optional sector routing restricts the top-M rule to
  sectors the document covers. Hard skips: parse failures and documents whose indicators are in another document.
- **Deterministic, cached, budgeted calls.** Temperature 0, schema-enforced output, cache by (document SHA-256,
  indicator, evidence fingerprint, prompt + rubric version, model), usage tracker with hard call / cost limits, a
  pre-run estimate, resumable runs (progress file + per-document result files).
- **Values.** Baseline / target are copied as written and normalised in code (`normalise_value`) for scoring.

## 3. What was added beyond the paper

- **Novel-indicator pass** (`pipeline/agent/novel_pass.py`): the standard-driven method cannot see indicators that
  are not in IndiKit. The document's own indicators are extracted from its logframe content only - tables with an
  indicator column in code (cells split on bullets / numbering; baseline, target, MoV from the mapped columns; the
  header-name lookup avoids the alphabetical key order of the stored rows), text sections and unmapped tables with one
  schema-enforced LLM call per section (`EXTRACTION_MODEL`, activity flag in the same call), then de-duplicated.
- **Consolidation** (`consolidate.py`): disclosure matches are linked to extracted indicators by table row / block,
  else text similarity; unmatched ones become `disclosure_only`; primary = verbatim > close > higher re-rank score,
  others are alternates.
- **Reverse check**: each novel candidate is embedded, its 10 nearest IndiKit indicators are shown to the model once
  with the rubric and knowledge; a hit becomes a `reverse_check` match. The number recovered measures what the gate
  and retrieval missed.
- **Review loop** (`pipeline/review/`): rows that need review (close matches, reverse-check matches, alternates,
  verbatim with low wording similarity) go to Excel per country; decisions are applied to a reviewed copy, metadata
  overrides are proposed as a separate diff (never applied automatically), and reviewed documents become gold files.
- **Portfolio outputs**: disclosure matrix, document summary with "indicators are in another document: <name>",
  most-used / never-used IndiKit indicators, clusters of similar novel indicators (greedy cosine clustering in code).
- **Error analysis** categories (retrieval_miss, gate_miss, llm_false_negative, llm_false_positive,
  wrong_match_type, wrong_value, parse_problem) with a top-fix per category.

## 4. Numbers so far (structural; no LLM run yet)

| Quantity | Value |
|---|---|
| Documents in `proposals/` | 115 (75 docx, 33 pdf, 7 xlsx; 112 en, 3 fr); parse status 107 ok, 8 partial, 0 failed |
| Documents with logframe content | 67; 48 without; 7 with indicators only in another document; 33 with both |
| IndiKit indicators | 351 (41 export-only) |
| Pair scores | 42 120 (vector scores; `score-pairs --all --no-rerank`, 37 s) |
| Gate defaults | tau = 0.620 (p90 of vector scores), M = 60 -> 7 016 pairs over 108 documents (mean 65 per document); 536 pairs kept by tau only, 3 281 by top-M only |
| Pilot (5 documents) | 333 pairs, ~387 calls incl. novel pass and reverse check |
| Full run estimate | ~8 100 calls; ~$3.14 with gpt-4o-mini (prompt ~1 800 / completion ~220 tokens per pair), $0 with a local model; exceeds `MAX_CALLS_PER_RUN = 1500` as shipped |
| Accuracy (Acc_DC, Acc_DE, ...) | **not measured**: 0 verified gold documents |

When verified gold exists: `python -m pipeline.run experiments --split dev` writes `outputs/eval/results.md`
(Table-4-like model table, Figure-3-like ablation table and figure, gate curve, reverse-check yield, error analysis)
and `outputs/eval/errors.csv`; the figures are then copied into `docs/phase_3_report.md`.
