"""Command-line entry point of the pipeline.

Phase 1 implements ``build-metadata`` and ``validate-gold``; the other subcommands are placeholders that
name the phase in which they arrive.

    python -m pipeline.run build-metadata [--enrich] [--model NAME]
    python -m pipeline.run validate-gold [--gold-dir DIR] [--file CSV] [--no-metadata-check]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from pipeline import config


def cmd_build_metadata(args: argparse.Namespace) -> int:
    from pipeline.metadata.build_metadata import build

    result = build(enrich=args.enrich, enrich_model=args.model)
    st = result.stats
    print(f"metadata records: {st.total} (both {st.both}, export-only {st.export_only}, site-only {st.site_only}, seed-only {st.seed_only})")
    print(f"conflicts: {len(result.conflicts)} -> {config.MERGE_CONFLICTS_CSV}")
    print(f"written: {config.METADATA_JSONL}, {config.REFERENCE_LIBRARY_CSV}, {config.BUILD_REPORT_MD}")
    if st.overrides_skipped:
        print("overrides skipped: " + "; ".join(f"{i} ({why})" for i, why in st.overrides_skipped))
    if st.extension_exceptions:
        print(f"extension exceptions: {len(st.extension_exceptions)} (see build report)")
    return 0


def cmd_validate_gold(args: argparse.Namespace) -> int:
    from pipeline.eval.gold import load_metadata_ids, validate_gold, validate_gold_file

    metadata_ids: Optional[set[str]] = None if args.no_metadata_check else load_metadata_ids()
    if metadata_ids is not None and not metadata_ids:
        print("note: metadata library not built; indikit_id values are not checked (run build-metadata first)")
        metadata_ids = None
    if args.file:
        rows, errors = validate_gold_file(Path(args.file), metadata_ids)
        checked, n_rows = [Path(args.file).name], len(rows)
    else:
        report = validate_gold(Path(args.gold_dir), metadata_ids)
        checked, n_rows, errors = report.files_checked, report.rows, report.errors
    print(f"checked {len(checked)} file(s), {n_rows} labelled row(s): {', '.join(checked)}")
    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("gold set OK")
    return 0


def cmd_fetch_missing(args: argparse.Namespace) -> int:
    from pipeline.metadata.fetch_missing import main as fetch_main

    return fetch_main([])


def _phase_stub(phase: str, what: str):
    def _cmd(args: argparse.Namespace) -> int:
        raise NotImplementedError(f"{what} arrives in {phase}; see docs/plan.md")

    return _cmd


# --------------------------------------------------------------------------- #
# Phase 2: preprocessing, knowledge bases, retrieval
# --------------------------------------------------------------------------- #
def _select_refs(args: argparse.Namespace):
    from pipeline.preprocess.registry import discover_documents, find_ref

    refs = discover_documents()
    if getattr(args, "doc", None):
        return [find_ref(args.doc, refs)]
    if getattr(args, "all", False):
        return refs
    raise SystemExit("choose --doc PATH|DOC_ID or --all")


def cmd_build_kb(args: argparse.Namespace) -> int:
    import time

    from pipeline.preprocess.embeddings import EmbeddingCache
    from pipeline.preprocess.kb_report import record_run
    from pipeline.preprocess.knowledge_base import build_kb, default_embedder
    from pipeline.preprocess.registry import needs_processing, read_manifest, upsert_manifest

    refs = _select_refs(args)
    manifest = read_manifest()
    todo = [r for r in refs if needs_processing(r, manifest, force=args.force)]
    print(f"{len(refs)} document(s) selected, {len(todo)} to (re)build")
    if not todo:
        return 0
    embedder = default_embedder(args.embedder)
    cache = EmbeddingCache()
    t0 = time.time()
    counts = {"ok": 0, "partial": 0, "failed": 0}
    for i, ref in enumerate(todo, start=1):
        row = build_kb(ref, embedder, cache, ocr=args.ocr)
        upsert_manifest(row)
        counts[row.parse_status] += 1
        print(f"[{i}/{len(todo)}] {row.parse_status:7s} {row.build_seconds:6.1f}s blocks={row.n_blocks} tables={row.n_tables} lf={row.n_logframe_tables}/{row.n_logframe_sections} refs={row.n_references} {ref.relative_path}", flush=True)
    cache.close()
    record_run("build-kb", {"documents": len(todo), "seconds": round(time.time() - t0, 1), "embedder": embedder.name, **counts})
    print(f"done in {time.time() - t0:.1f}s: {counts}")
    return 0 if counts["failed"] == 0 else 1


def cmd_build_baseline_kb(args: argparse.Namespace) -> int:
    import time

    from pipeline.preprocess.baseline_kb import build_baseline_kb, default_baseline_embedder
    from pipeline.preprocess.embeddings import EmbeddingCache
    from pipeline.preprocess.kb_report import record_run

    refs = _select_refs(args)
    embedder = default_baseline_embedder()
    cache = EmbeddingCache()
    t0 = time.time()
    n_fail = 0
    for i, ref in enumerate(refs, start=1):
        meta = build_baseline_kb(ref, embedder, cache)
        n_fail += meta.get("status") != "ok"
        print(f"[{i}/{len(refs)}] {meta.get('status'):6s} {meta.get('build_seconds', 0):6.1f}s chunks={meta.get('n_chunks', 0)} {ref.relative_path}", flush=True)
    cache.close()
    record_run("build-baseline-kb", {"documents": len(refs), "seconds": round(time.time() - t0, 1), "embedder": embedder.name, "failed": n_fail})
    print(f"done in {time.time() - t0:.1f}s, failed: {n_fail}")
    return 0 if n_fail == 0 else 1


def _indicators(ids: Optional[list[str]] = None):
    from pipeline.metadata.build_metadata import read_jsonl

    records = read_jsonl()
    if ids:
        wanted = set(ids)
        records = [r for r in records if r.indikit_id in wanted]
    return records


def cmd_score_pairs(args: argparse.Namespace) -> int:
    from pipeline.agent.retrieve import Retriever, RetrievalConfig, score_pairs, write_pair_scores
    from pipeline.preprocess.embeddings import EmbeddingCache
    from pipeline.preprocess.kb_report import record_run
    from pipeline.preprocess.registry import read_manifest

    manifest = read_manifest()
    rows = [r for r in manifest.values() if r.parse_status != "failed"]
    if args.doc:
        rows = [r for r in rows if r.doc_id == args.doc or r.relative_path == args.doc or args.doc in r.relative_path]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no documents in the manifest; run build-kb first")
    use_rerank = not args.no_rerank
    cfg = RetrievalConfig(use_rerank=use_rerank, **({"reranker": args.reranker} if args.reranker else {}))
    retriever = Retriever(cfg, cache=EmbeddingCache())
    result_rows, stats = score_pairs(retriever, [r.doc_id for r in rows], _indicators(), use_rerank=use_rerank)
    out = write_pair_scores(result_rows, Path(args.out) if args.out else config.PAIR_SCORES_PATH)
    record_run("score-pairs" + ("" if use_rerank else "-no-rerank"), {**stats, "output": str(out)})
    print(f"wrote {len(result_rows)} pair scores to {out} in {stats['seconds']}s ({stats})")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    from pipeline.agent.retrieve import Retriever, RetrievalConfig, retrieve_baseline
    from pipeline.preprocess.embeddings import EmbeddingCache
    from pipeline.preprocess.registry import read_manifest

    manifest = read_manifest()
    doc_id = args.doc
    if doc_id not in manifest:
        matches = [r.doc_id for r in manifest.values() if args.doc in r.relative_path]
        if len(matches) != 1:
            raise SystemExit(f"--doc must be a doc_id or a unique path fragment; matches: {matches[:5]}")
        doc_id = matches[0]
    inds = _indicators([args.indicator])
    if not inds:
        raise SystemExit(f"indicator {args.indicator} not in the metadata")
    ind = inds[0]
    if args.baseline:
        res = retrieve_baseline(doc_id, ind, top_k=args.top_k, cache=EmbeddingCache())
    else:
        cfg = RetrievalConfig(use_rerank=not args.no_rerank, use_search_terms=not args.no_search_terms, top_k=args.top_k, **({"reranker": args.reranker} if args.reranker else {}))
        res = Retriever(cfg, cache=EmbeddingCache()).retrieve(doc_id, ind)
    print(f"document: {doc_id} ({manifest[doc_id].relative_path})\nindicator: {ind.indikit_id} {ind.name}\nquery: {res.query_text[:300]}\n"
          f"candidates: {res.n_candidates}; best vector {res.best_vector_score:.3f}; best rerank {res.best_rerank_score}\nconfig: {res.config_fingerprint}\n")
    for i, e in enumerate(res.evidence, start=1):
        print(f"--- [{i}] {e.kb_type}/{e.entry_kind} {e.location} vector={e.vector_score:.3f} rerank={e.rerank_score} logframe={e.is_logframe}{' (boosted)' if e.boost_applied else ''}{' via ' + e.expanded_from if e.expanded_from else ''}")
        print(e.display_text[:1200])
    return 0


def cmd_kb_report(args: argparse.Namespace) -> int:
    from pipeline.preprocess.kb_report import write_report

    path = write_report()
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeline.run", description="ESGReveal-style IndiKit pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-metadata", help="merge export + site into indikit_metadata.jsonl, reference_library.csv and the build report")
    p.add_argument("--enrich", action="store_true", help="also run the optional LLM enrichment of Knowledge / SearchTerm (makes API calls)")
    p.add_argument("--model", default=None, help="model name for --enrich (default: configured model)")
    p.set_defaults(func=cmd_build_metadata)

    p = sub.add_parser("validate-gold", help="validate the gold-set index and files (pipeline/eval/gold)")
    p.add_argument("--gold-dir", default=str(config.GOLD_DIR))
    p.add_argument("--file", default=None, help="validate a single gold CSV instead of the whole directory")
    p.add_argument("--no-metadata-check", action="store_true", help="skip checking indikit_id values against the metadata library")
    p.set_defaults(func=cmd_validate_gold)

    p = sub.add_parser("fetch-missing", help="fetch the PIP indicator pages missing from the site data (network)")
    p.set_defaults(func=cmd_fetch_missing)

    p = sub.add_parser("build-kb", help="parse documents and build the text / outline / table knowledge bases under outputs/kb/")
    p.add_argument("--doc", default=None, help="one document: path, relative path or doc_id")
    p.add_argument("--all", action="store_true", help="every document under the configured roots (incremental)")
    p.add_argument("--force", action="store_true", help="rebuild even when SHA-256 and pipeline version are unchanged")
    p.add_argument("--ocr", action="store_true", help="OCR pages without a text layer (needs pytesseract + tesseract)")
    p.add_argument("--embedder", default=None, help="embedding model spec (default: config.KB_EMBEDDING_MODEL; 'hashing' for a dependency-free smoke run)")
    p.set_defaults(func=cmd_build_kb)

    p = sub.add_parser("build-baseline-kb", help="plain-RAG baseline index (fixed chunks, all-MiniLM-L6-v2) under outputs/kb_baseline/")
    p.add_argument("--doc", default=None)
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_build_baseline_kb)

    p = sub.add_parser("score-pairs", help="best vector / re-rank score for every document x indicator pair -> outputs/kb/pair_scores.*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--doc", default=None, help="restrict to one document (doc_id or path fragment)")
    p.add_argument("--limit", type=int, default=None, help="first N documents of the manifest")
    p.add_argument("--no-rerank", action="store_true", help="vector scores only (much faster)")
    p.add_argument("--reranker", default=None, help="re-ranker spec (default: config.RERANKER_MODEL)")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_score_pairs)

    p = sub.add_parser("retrieve", help="print the ranked evidence for one document x indicator pair")
    p.add_argument("--doc", required=True, help="doc_id or unique path fragment")
    p.add_argument("--indicator", required=True, help="IndiKit ID")
    p.add_argument("--top-k", type=int, default=config.RETRIEVAL_TOP_K)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-search-terms", action="store_true")
    p.add_argument("--baseline", action="store_true", help="use the plain-RAG baseline index instead")
    p.add_argument("--reranker", default=None)
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("kb-report", help="write outputs/kb/preprocess_report.md")
    p.set_defaults(func=cmd_kb_report)

    for name, phase, what in (
        ("query", "Phase 3", "retrieval and LLM answering per document x indicator"),
        ("novel-pass", "Phase 3", "the novel-indicator pass"),
        ("run-experiments", "Phase 3", "benchmark and ablation experiments"),
    ):
        p = sub.add_parser(name, help=f"[{phase}] {what}")
        p.set_defaults(func=_phase_stub(phase, what))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
