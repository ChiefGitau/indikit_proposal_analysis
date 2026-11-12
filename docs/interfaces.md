# Interfaces Phase 3 relies on (kept stable)

All functions are offline; none calls an LLM. Paths come from `pipeline/config.py`.

## Document registry

```python
from pipeline.preprocess.registry import list_documents, get_document

rows = list_documents()                 # list[ManifestRow] from outputs/kb/manifest.csv, sorted by relative_path
row = get_document("proposals-africa-uganda-large-hope-copy-of-logframe-hope-project-final-1a2b3c4d")
row.parse_status, row.language, row.n_logframe_tables, row.indicators_elsewhere   # 'ok', 'en', 1, False
```

`ManifestRow` fields: `doc_id, relative_path, region, country, file_type, sha256, n_pages_or_sheets, n_chars,
n_tables, language, has_text_layer, parse_status, error, parsed_at, pipeline_version, n_blocks, n_logframe_tables,
n_logframe_sections, n_references, indicators_elsewhere, indicators_also_elsewhere, build_seconds, embedding_model,
n_kb_text, n_kb_outline, n_kb_table, warnings`.

## Knowledge base

```python
from pipeline.preprocess.knowledge_base import load_kb

kb = load_kb(doc_id)                    # KnowledgeBase from outputs/kb/<doc_id>/
kb.entries["table"]                     # list[KBEntry] (entry_kind: row | keyword | summary); "text" (chunk); "outline" (node)
hits = kb.search("table", query_vector, top_n=5)   # [(KBEntry, cosine)]
kb.expand(hits[0][0])                   # outline node -> its chunks + tables; keyword/summary -> rows (or summary if large)
kb.tables_by_id, kb.rows_by_table, kb.blocks, kb.outline, kb.references
```

`KBEntry`: `entry_id, doc_id, kb_type, entry_kind, text_for_embedding, display_text, location, is_logframe, payload_ref, summary`.
`display_text` of a table row stands alone: table title, column header, carried-down result context and the row.

## Retrieval

```python
from pipeline.agent.retrieve import retrieve, retrieve_baseline, RetrievalConfig
from pipeline.metadata.build_metadata import read_jsonl

ind = {r.indikit_id: r for r in read_jsonl()}["5138"]
res = retrieve(doc_id, ind, top_n=5, top_k=4, use_search_terms=True, use_rerank=True)
res.evidence[0].display_text, res.evidence[0].location, res.evidence[0].vector_score, res.evidence[0].rerank_score
res.best_vector_score, res.best_rerank_score, res.n_candidates, res.config_fingerprint

base = retrieve_baseline(doc_id, ind, top_k=4)     # plain-RAG index under outputs/kb_baseline/, same RetrievalResult type
```

For batch work construct one `Retriever(RetrievalConfig(...), cache=EmbeddingCache())` and call `.retrieve(...)`
so the models are loaded once. Ablation flags: `use_search_terms`, `use_rerank`, `kb_types=("text",)`,
`RetrievalConfig(logframe_boost=0.0)`.

`Evidence`: `entry_id, kb_type, entry_kind, display_text, location, is_logframe, vector_score, rerank_score,
boost_applied, expanded_from, table_id, row_id, block_ids`.

## Logframe content (novel-indicator pass)

```python
from pipeline.preprocess.logframe import logframe_content

content = logframe_content(doc_id)
content["tables"]       # [{"table": Table, "rows": [TableRow]}] - logframe tables in reading order, with column_roles
content["blocks"]       # [Block] - blocks inside logframe-titled sections (headings, paragraphs, list items)
content["section_heading_ids"]
```

Phase 3 feeds these (and only these) to the novel-indicator extraction.

## References

```python
from pipeline.preprocess.references import load_references

for r in load_references(doc_id):       # list[Reference]
    r.referenced_document, r.kind, r.confidence, r.quote, r.location   # 'Annex O: Indicator Tracking Table', 'tracker', 0.95, ...
```

## Pair scores (retrieval gate)

`outputs/kb/pair_scores.parquet` (or `.csv` without pyarrow), written by `python -m pipeline.run score-pairs --all`:

| column | meaning |
|---|---|
| `doc_id`, `indikit_id` | the pair |
| `best_vector_score` | best cosine (with logframe boost) over the expanded candidates |
| `best_rerank_score` | best cross-encoder score (empty with `--no-rerank`) |
| `top_entry_id`, `top_kb_type`, `top_is_logframe` | the top evidence entry |

Phase 3 tunes the gate threshold on this file against the gold set.
