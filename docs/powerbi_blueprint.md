# Power BI blueprint — ZOA proposal indicator report

Paste the sections below into Copilot (Power BI Desktop) one at a time, in order. Section A is done by hand in Power
Query (Copilot cannot create connections or relationships); sections B to D are prompts. The data comes from
`outputs/powerbi/` of the `proposal_investigated` pipeline, refreshed with `python -m pipeline.run export-powerbi`.

---

## A. Data model (set up manually in Power Query, ~15 minutes)

1. **Parameter** `DataFolder` (Text) = the absolute path of `outputs/powerbi/`.
2. Import these five CSVs (UTF-8, comma, first row headers), one query each, using `DataFolder & "\<file>"`:

| Query name | File | Key | Keep as text |
|---|---|---|---|
| `IndiKit` | `indikit_indicators.csv` | `indikit_id` | `indikit_id`, `levels`, `kpi_wording_en`, `kpi_wording_fr`, `url` |
| `Documents` | `document_summary.csv` | `doc_id` | `doc_id`, `relative_path`, `referenced_documents`, `note`, `errors`, `skipped_reason` |
| `DocumentIndicators` | `document_indicators.csv` | `doc_id` + `doc_indicator_id` | `baseline`, `target` (mixed formats: `36%`, `69.200,00`, `n/a`), `evidence_quote`, `alternate_ids`, `nearest_indikit`, `indikit_id` |
| `DisclosureMatrix` | `disclosure_matrix.csv` | `doc_id` + `indikit_id` | `indikit_id`, `doc_indicator_ids`, `sources` |
| `NovelIndicators` | `novel_indicators.csv` | `doc_id` + `doc_indicator_id` | `baseline`, `target`, `nearest_indikit_ids`, `nearest_similarities` |

3. Data types: `needs_review`, `is_indicator`, `export_only`, `on_site`, `in_export`, `indicators_elsewhere`,
   `indicators_also_elsewhere` → True/False. `total_indicators`, `verbatim`, `close`, `none`, `pairs_selected`,
   `pairs_asked`, `pairs_present`, `llm_calls`, `n_document_indicators`, `cluster_id`, `text_similarity` → whole / decimal
   number. `pct_matched`, `cost_usd` → decimal. Everything else text. Replace empty `indikit_id` with null.
4. Add to `Documents` a column `document_name` = the file name part of `relative_path` (text after the last `/`).
5. Add to `DocumentIndicators` a column `matched` = `match_type <> "none"`.
6. Relationships (all one-to-many, single direction, from dimension to fact):
   - `IndiKit[indikit_id]` → `DocumentIndicators[indikit_id]`
   - `IndiKit[indikit_id]` → `DisclosureMatrix[indikit_id]`
   - `Documents[doc_id]` → `DocumentIndicators[doc_id]`
   - `Documents[doc_id]` → `DisclosureMatrix[doc_id]`
   - `Documents[doc_id]` → `NovelIndicators[doc_id]`
   Do **not** load the `portfolio_*.csv` files; the report computes those numbers itself.
7. Mark `IndiKit` and `Documents` as dimension tables; hide the key columns on the fact tables.

Value sets you can rely on: `match_type` ∈ {verbatim, close, none}; `source` ∈ {disclosure, extraction,
reverse_check}; `level` ∈ {output, outcome, impact, unclear}; `extraction_method` ∈ {table_code, text_llm, blank};
`IndiKit[quantity]` ∈ {number, percentage, ratio, score, yes_no, other}; `IndiKit[levels]` is a "; "-separated list.

---

## B. Prompt for Copilot — measures

> Create a measure table called `Measures` with these DAX measures, formatted as stated:
>
> - `Documents processed` = COUNTROWS(FILTER(Documents, Documents[skipped_reason] = BLANK() || Documents[skipped_reason] = "")) — whole number
> - `Documents pointing elsewhere` = CALCULATE(COUNTROWS(Documents), Documents[indicators_elsewhere] = TRUE()) — whole number
> - `Document indicators` = COUNTROWS(DocumentIndicators) — whole number
> - `Matched indicators` = CALCULATE(COUNTROWS(DocumentIndicators), DocumentIndicators[matched] = TRUE()) — whole number
> - `Verbatim matches` = CALCULATE(COUNTROWS(DocumentIndicators), DocumentIndicators[match_type] = "verbatim")
> - `Close matches` = CALCULATE(COUNTROWS(DocumentIndicators), DocumentIndicators[match_type] = "close")
> - `Unmatched indicators` = CALCULATE(COUNTROWS(DocumentIndicators), DocumentIndicators[match_type] = "none")
> - `Share matched` = DIVIDE([Matched indicators], [Document indicators]) — percentage, 0 decimals
> - `Documents using indicator` = DISTINCTCOUNT(DisclosureMatrix[doc_id]) — whole number
> - `Countries using indicator` = DISTINCTCOUNT(DisclosureMatrix[country]) — whole number
> - `IndiKit indicators used` = DISTINCTCOUNT(DisclosureMatrix[indikit_id]) — whole number
> - `IndiKit indicators never used` = CALCULATE(COUNTROWS(IndiKit), IndiKit[on_site] = TRUE()) - [IndiKit indicators used] — whole number
> - `Rows needing review` = CALCULATE(COUNTROWS(DocumentIndicators), DocumentIndicators[needs_review] = TRUE())
> - `Indicators per document` = DIVIDE([Document indicators], [Documents processed]) — decimal, 1 decimal
> - `Novel clusters` = DISTINCTCOUNT(NovelIndicators[cluster_id])
> - `Export-only matches` = CALCULATE(COUNTROWS(DocumentIndicators), DocumentIndicators[export_only] = TRUE())
>
> Hide the measure table's dummy column.

---

## C. Prompt for Copilot — report pages

> Build a report with six pages. Use one consistent palette with these semantic colours: verbatim = dark green,
> close = amber, none = grey, needs review = red. Every page has slicers for `Documents[region]`,
> `Documents[country]`, `Documents[file_type]` and `IndiKit[sector]` in a left-hand panel, synced across pages. Titles
> in sentence case. No 3D, no pie charts, no gauges.
>
> **Page 1 — Overview.** Six cards in a row: `Documents processed`, `Document indicators`, `Share matched`,
> `IndiKit indicators used`, `IndiKit indicators never used`, `Rows needing review`. Below: a stacked bar chart of
> `Document indicators` by `Documents[country]` split by `DocumentIndicators[match_type]`; a 100 % stacked bar of the
> same split by `IndiKit[sector]`; a card with `Documents pointing elsewhere` and the text "documents whose indicators
> live in an annex or logframe file".
>
> **Page 2 — Most used IndiKit indicators.** A horizontal bar chart of `Documents using indicator` by
> `IndiKit[name]`, top 25, sorted descending, tooltip showing `IndiKit[indikit_id]`, `IndiKit[sector]`,
> `Countries using indicator`, `Verbatim matches`, `Close matches`. Next to it a matrix: rows `IndiKit[sector]` then
> `IndiKit[name]`, columns `Documents[country]`, values `Documents using indicator`, with a heat-map colour scale.
> A toggle slicer on `IndiKit[on_site]`.
>
> **Page 3 — Never used IndiKit indicators.** A table of `IndiKit[indikit_id]`, `IndiKit[name]`, `IndiKit[sector]`,
> `IndiKit[levels]`, `IndiKit[quantity]`, filtered to indicators with no rows in `DisclosureMatrix`
> (use a visual-level filter: `Documents using indicator` is blank), with `IndiKit[on_site]` = TRUE by default. A bar
> chart of the count of these indicators by `IndiKit[sector]`.
>
> **Page 4 — Document detail.** A single-select slicer on `Documents[document_name]`. Cards: `Document indicators`,
> `Matched indicators`, `Share matched`. A table with `DocumentIndicators[indicator_text]`, `level`, `baseline`,
> `target`, `match_type` (conditional colour), `indikit_name`, `source`, `needs_review`, `evidence_quote` (word wrap).
> Below it a text card showing `Documents[note]` for documents that point to another file.
>
> **Page 5 — Unmatched indicators.** A table of `NovelIndicators[indicator_text]`, `level`, `baseline`, `target`,
> `country`, `cluster_id`, `nearest_indikit_ids`, sorted by `cluster_id`. A bar chart of the count of novel indicators
> by `cluster_id`, top 20, and a bar chart by `NovelIndicators[level]`. Card: `Novel clusters`.
>
> **Page 6 — Review and data quality.** A table of `DocumentIndicators` rows where `needs_review` = TRUE with
> `document_name`, `indicator_text`, `match_type`, `indikit_name`, `needs_review_reason`, `text_similarity`. Cards:
> `Rows needing review`, `Export-only matches`. A table of `Documents` with `document_name`, `country`,
> `file_type`, `parse_status`, `total_indicators`, `indicators_elsewhere`, `referenced_documents`, `errors`.

---

## D. Prompt for Copilot — finishing

> Add a report-level text box on every page footer: "Source: ZOA proposal indicator pipeline, outputs/powerbi,
> run <run id>". Set page size 16:9. Add drill-through from any IndiKit indicator (pages 2 and 3) to page 4 filtered
> to the documents that use it, and from any document (pages 1 and 6) to page 4. Create a bookmark "Reset filters".
> Name the file `ZOA_IndiKit_Proposals.pbix`.

---

## Refreshing

```bash
python -m pipeline.run export-powerbi          # rewrite outputs/powerbi from the latest run
```

Then **Home → Refresh** in Power BI. Column names and value sets are stable across runs; only the row counts change.
When the pipeline later carries actual progress values, add a `fact_progress.csv` (doc_id, doc_indicator_id, period,
actual) and relate it to `DocumentIndicators`; page 4 then gets a baseline / target / actual visual.
