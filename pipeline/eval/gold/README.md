# Gold set format (paper Section 5.1.1, docs/plan.md Section 6.1)

One CSV per labelled document plus an index. Labels are supplied and verified by a MEAL colleague;
**do not invent labels for real documents**. `_example.csv` is a fictional template.

## `index.csv`

| Column | Values |
|---|---|
| `doc_id` | short stable identifier, also the file name `<doc_id>.csv` |
| `source_file` | path of the document relative to `proposals/` |
| `language` | `en`, `fr`, ... |
| `has_logframe` | `true` / `false` |
| `indicators_in_other_document` | name of the referenced document (e.g. `Annex C - Logical Framework`) or empty |
| `labelled_by` | initials |
| `verified` | `true` / `false` (verified by a second MEAL colleague) |
| `split` | `dev` (prompt tuning allowed) or `test` (held out, never tuned on) |

## `<doc_id>.csv` - one row per **document indicator**

| Column | Values |
|---|---|
| `doc_indicator_id` | unique within the file, e.g. `D01` |
| `indicator_text` | exact wording in the document |
| `level` | `impact`, `outcome`, `output`, `unclear` |
| `baseline` | as written, or empty |
| `target` | as written, or empty |
| `unit` | `%`, `number`, `USD`, ... or empty |
| `location` | page or section label |
| `indikit_id` | IndiKit ID or empty (= no match) |
| `match_type` | `verbatim`, `close`, `none` (`none` if and only if `indikit_id` is empty) |

IndiKit indicators **not listed** for a document count as **absent** for that document. Document indicators
with `match_type=none` feed the novel-indicator metrics.

Validate with:

```
python -m pipeline.run validate-gold                 # index + every listed file + templates
python -m pipeline.run validate-gold --file path.csv # one file
```
