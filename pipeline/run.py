"""Command-line entry point of the pipeline.

Phase 1: ``build-metadata``, ``validate-gold``, ``fetch-missing``. Phase 2: ``build-kb``, ``build-baseline-kb``,
``score-pairs``, ``retrieve``, ``kb-report``. Phase 3: ``select-pairs``, ``estimate``, ``answer``, ``novel-pass``,
``consolidate``, ``run``, ``report``, ``experiments``, ``export-review``, ``import-review``.

Live LLM calls happen only through ``answer``, ``novel-pass``, ``run`` and ``experiments`` and only when a provider
is configured (config.LLM_PROVIDER / env LLM_PROVIDER). ``estimate`` and ``select-pairs`` never call a model.
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


# --------------------------------------------------------------------------- #
# Phase 3: agent, runs, evaluation, review
# --------------------------------------------------------------------------- #
def _metadata_map():
    return {r.indikit_id: r for r in _indicators()}


def _resolve_doc_ids(args: argparse.Namespace, manifest: dict) -> tuple[list[str], str]:
    """(doc ids, mode) from --docs / --pilot / --all."""
    if getattr(args, "all", False):
        return [r.doc_id for r in sorted(manifest.values(), key=lambda r: r.relative_path) if r.region != "template"], "full"
    if getattr(args, "pilot", False):
        from pipeline.agent.orchestrate import pilot_doc_ids
        from pipeline.eval.run_experiments import gold_doc_id_map, gold_documents

        gold = []
        try:
            docs = gold_documents(None, verified_only=False)
            gmap = gold_doc_id_map(docs, manifest)
            gold = [(gmap[d.doc_id], bool(d.entry.verified), d.entry.split) for d in docs if d.doc_id in gmap and d.entry]
        except Exception:
            pass
        return pilot_doc_ids(manifest, gold, size=getattr(args, "pilot_size", None) or config.PILOT_SIZE), "pilot"
    docs = getattr(args, "docs", None) or ([args.doc] if getattr(args, "doc", None) else [])
    if not docs:
        raise SystemExit("choose --docs DOC_ID|PATH-FRAGMENT ... | --pilot | --all --confirm-full-run")
    out = []
    for d in docs:
        if d in manifest:
            out.append(d)
            continue
        matches = [r.doc_id for r in manifest.values() if d in r.relative_path]
        if len(matches) != 1:
            raise SystemExit(f"--docs {d!r} must be a doc_id or a unique path fragment; matches: {matches[:5]}")
        out.append(matches[0])
    return out, "docs"


def _run_config(args: argparse.Namespace, run_id: str, mode: str):
    from pipeline.agent.orchestrate import RunConfig

    kw = dict(run_id=run_id, mode=mode)
    for name, attr in (("provider", "provider"), ("answer_model", "model"), ("extraction_model", "extraction_model"), ("tau", "tau"), ("tau_percentile", "tau_percentile"), ("top_m", "top_m"), ("top_k", "top_k"),
                       ("workers", "workers"), ("max_calls", "max_calls"), ("max_cost_usd", "max_cost")):
        v = getattr(args, attr, None)
        if v is not None:
            kw[name] = v
    if getattr(args, "route_sectors", False):
        kw["route_sectors"] = True
    if getattr(args, "no_export_only", False):
        kw["include_export_only"] = False
    if getattr(args, "no_knowledge", False):
        kw["use_knowledge"] = False
    if getattr(args, "no_search_terms", False):
        kw["use_search_terms"] = False
    if getattr(args, "no_rerank", False):
        kw["use_rerank"] = False
    if getattr(args, "baseline", False):
        kw["baseline_retrieval"] = True
    if getattr(args, "no_novel", False):
        kw["novel_pass"] = False
    if getattr(args, "no_reverse_check", False):
        kw["reverse_check"] = False
    if getattr(args, "force", False):
        kw["force"] = True
    if getattr(args, "allow_unverified", False):
        kw["watermark"] = "UNVERIFIED GOLD"
    return RunConfig(**kw)


def _build_runner(args: argparse.Namespace, run_id: str, mode: str, *, need_provider: bool, need_models: bool = True):
    from pipeline.agent.llm import UsageTracker, get_provider, provider_status
    from pipeline.agent.orchestrate import Runner, run_dir_for
    from pipeline.preprocess.registry import read_manifest

    manifest = read_manifest()
    if not manifest:
        raise SystemExit("no documents in the manifest; run build-kb first")
    cfg = _run_config(args, run_id, mode)
    provider = None
    if need_provider:
        ok, msg = provider_status(cfg.provider)
        if not ok:
            raise SystemExit(f"LLM access is not configured: {msg}. Set LLM_PROVIDER / ANSWER_MODEL (pipeline/config.py or environment) - no live call was made.")
        provider = get_provider(cfg.provider)
        print(f"provider: {msg}; answer model {cfg.answer_model}; extraction model {cfg.extraction_model}; limits {cfg.max_calls} calls / ${cfg.max_cost_usd}")
    retriever = embedder = ecache = None
    if need_models:
        from pipeline.agent.retrieve import RetrievalConfig, Retriever
        from pipeline.preprocess.embeddings import EmbeddingCache

        ecache = EmbeddingCache()
        rcfg = RetrievalConfig(top_n=cfg.top_n, top_k=cfg.top_k, use_search_terms=cfg.use_search_terms, use_rerank=cfg.use_rerank, **({"reranker": args.reranker} if getattr(args, "reranker", None) else {}))
        retriever = Retriever(rcfg, cache=ecache)
        embedder = retriever.embedder
    usage = UsageTracker(cfg.max_calls, cfg.max_cost_usd, path=run_dir_for(run_id) / "usage.json")
    runner = Runner(cfg, provider=provider, metadata=_metadata_map(), manifest=manifest, retriever=retriever, embedder=embedder, embedding_cache=ecache, usage=usage)
    return runner, manifest


def _latest_run_id(args: argparse.Namespace) -> str:
    rid = getattr(args, "run_id", None)
    if rid:
        return rid
    marker = config.LATEST_DIR / "LATEST_RUN.txt"
    if marker.exists():
        return marker.read_text(encoding="utf-8").splitlines()[0].strip()
    runs = sorted(p.name for p in config.RUNS_DIR.glob("*") if p.is_dir()) if config.RUNS_DIR.exists() else []
    if not runs:
        raise SystemExit("no runs under outputs/runs; pass --run-id")
    return runs[-1]


def _print_estimate(est: dict) -> None:
    print(f"documents: {est['documents']}; pairs: {est['pairs']}; novel-pass calls: ~{est['novel_calls']}; reverse-check calls: ~{est['reverse_calls']}")
    print(f"total calls: ~{est['calls']}; estimated cost: ${est['cost_usd']:.2f} ({est['answer_model']} / {est['extraction_model']}); limits: {est['max_calls']} calls, ${est['max_cost_usd']}; within budget: {est['within_budget']}")


def cmd_select_pairs(args: argparse.Namespace) -> int:
    from pipeline.agent.orchestrate import new_run_id

    run_id = args.run_id or new_run_id("select")
    runner, manifest = _build_runner(args, run_id, "docs", need_provider=False, need_models=False)
    doc_ids, mode = _resolve_doc_ids(args, manifest)
    runner.cfg.mode = mode
    sel = runner.select(doc_ids)
    s = sel.summary()
    print(f"run {run_id}: tau={s['tau']} ({s['score_kind']} scores), top_m={s['top_m']}, routing={s['route_sectors']}; {s['pairs_selected']} of {s['pairs_candidates']} pairs kept over {s['documents']} documents (mean {s['mean_pairs_per_document']}/doc); {s['skipped_documents']} documents skipped")
    for d, why in sel.skipped_documents[:20]:
        print(f"  skipped {d}: {why}")
    _print_estimate(runner.estimate(sel, n_docs=len(doc_ids)))
    print(f"wrote {runner.run_dir / 'selected_pairs.csv'}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from pipeline.agent.orchestrate import new_run_id

    runner, manifest = _build_runner(args, args.run_id or new_run_id("estimate"), "docs", need_provider=False, need_models=False)
    doc_ids, mode = _resolve_doc_ids(args, manifest)
    sel = runner.select(doc_ids)
    print(f"gate: tau={sel.tau:.4f} ({sel.score_kind} scores, {'p' + format(runner.cfg.tau_percentile, 'g') if runner.cfg.tau is None else 'fixed'}), top_m={sel.top_m}, routing={sel.route_sectors}; skipped documents: {len(sel.skipped_documents)}")
    _print_estimate(runner.estimate(sel, n_docs=len(doc_ids)))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from pipeline.agent.orchestrate import new_run_id

    run_id = args.run_id or new_run_id("run")
    runner, manifest = _build_runner(args, run_id, "docs", need_provider=True)
    doc_ids, mode = _resolve_doc_ids(args, manifest)
    runner.cfg.mode = mode
    if mode == "full" and not args.confirm_full_run:
        raise SystemExit("a full run over all documents requires --confirm-full-run (use --pilot or --docs otherwise)")
    results, stop = runner.run(doc_ids, confirm_full_run=args.confirm_full_run)
    n_all = sum(1 for r in manifest.values() if r.region != "template" and r.parse_status != "failed")
    out = runner.finalize(copy_to_latest=(mode == "full" and stop is None) or args.copy_latest, extrapolate_to=n_all if mode != "full" else None)
    info = out["info"]
    print(f"run {run_id}: {info['documents']} documents ({info['documents_skipped']} skipped), {info['pairs_asked']} pairs asked, {info['pairs_present']} present, {info['unsupported_dropped']} unsupported dropped, "
          f"{info['usage']['calls']} calls, ${info['usage']['cost_usd']:.4f}, {info['runtime_seconds']}s -> {runner.run_dir}")
    if info.get("extrapolation"):
        e = info["extrapolation"]
        print(f"extrapolation to {e['documents']} documents: ~{e['calls']} calls, ~${e['cost_usd']}, ~{e['hours']} h; within MAX_COST_USD_PER_RUN: {e['within_MAX_COST_USD_PER_RUN']}")
    if stop:
        print(f"STOPPED: {stop} (resume with the same --run-id)")
        return 2
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    """Disclosure step only: answer the selected pairs of a run (resumable)."""
    runner, manifest = _build_runner(args, _latest_run_id(args), "docs", need_provider=True)
    sel = runner.selection()
    stop = None
    for doc_id, g in sel.groupby("doc_id"):
        answers, stop = runner.answer_document(doc_id, g["indikit_id"].tolist())
        print(f"{doc_id}: {sum(1 for a in answers if a.disclosure == 'present')} present of {len(answers)} pairs ({sum(1 for a in answers if a.cached)} cached)")
        if stop:
            print(f"STOPPED: {stop}")
            break
    runner.usage.write()
    return 2 if stop else 0


def cmd_novel_pass(args: argparse.Namespace) -> int:
    runner, manifest = _build_runner(args, _latest_run_id(args), "docs", need_provider=not args.no_llm, need_models=False)
    doc_ids, _ = _resolve_doc_ids(args, manifest) if (args.docs or args.pilot or args.all) else (runner.selection()["doc_id"].unique().tolist(), "docs")
    for d in doc_ids:
        res = runner.novel_document(d)
        print(f"{d}: {len(res.indicators)} document indicators {res.stats}" + (f" errors: {res.errors}" if res.errors else ""))
        if res.budget_stop:
            print(f"STOPPED: {res.budget_stop}")
            return 2
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    """Consolidate answers + novel pass of a run into the output files (reverse check needs the provider unless --no-reverse-check)."""
    runner, manifest = _build_runner(args, _latest_run_id(args), "docs", need_provider=not args.no_reverse_check)
    sel = runner.selection()
    by_doc = {d: g["indikit_id"].tolist() for d, g in sel.groupby("doc_id")}
    for doc_id, ids in by_doc.items():
        r = runner.process_document(doc_id, ids)
        print(f"{doc_id}: {len(r.consolidated)} indicators, reverse check {r.reverse_stats}")
    out = runner.finalize(copy_to_latest=args.copy_latest)
    print(f"wrote {', '.join(str(p) for p in out['paths'].values())}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    import pandas as pd

    from pipeline.agent.outputs import document_report_markdown
    from pipeline.preprocess.registry import read_manifest

    base = config.LATEST_DIR if not args.run_id else config.RUNS_DIR / args.run_id
    if not (base / "document_indicators.csv").exists():
        raise SystemExit(f"{base / 'document_indicators.csv'} not found; run `python -m pipeline.run run ...` first")
    doc_ind = pd.read_csv(base / "document_indicators.csv", dtype=str, keep_default_na=False)
    summary = pd.read_csv(base / "document_summary.csv", dtype=str, keep_default_na=False)
    manifest = read_manifest()
    doc_id = args.doc
    if doc_id not in set(summary["doc_id"]):
        matches = [r.doc_id for r in manifest.values() if args.doc in r.relative_path and r.doc_id in set(summary["doc_id"])]
        if len(matches) != 1:
            raise SystemExit(f"--doc must be a doc_id of the run or a unique path fragment; matches: {matches[:5]}")
        doc_id = matches[0]
    row = summary[summary["doc_id"] == doc_id].iloc[0]
    print(f"# {row['relative_path']}\n")
    print(document_report_markdown(doc_id, doc_ind, summary))
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    from pipeline.agent.orchestrate import RunConfig, Runner, run_dir_for
    from pipeline.agent.llm import UsageTracker, get_provider, provider_status
    from pipeline.eval.run_experiments import EXPERIMENTS, UNVERIFIED_WATERMARK, ablation_figure, error_analysis, error_counts, gold_doc_id_map, gold_documents, run_experiment, write_errors, write_results_table
    from pipeline.preprocess.registry import read_manifest

    manifest = read_manifest()
    metadata = _metadata_map()
    verified = gold_documents(args.split, verified_only=True)
    watermark = ""
    if not verified:
        if not args.allow_unverified:
            raise SystemExit(f"no verified gold documents in split '{args.split}': experiments skipped (docs/phase_3.md Section 2). Use --allow-unverified for a debugging run watermarked '{UNVERIFIED_WATERMARK}'.")
        watermark = UNVERIFIED_WATERMARK
        gold = gold_documents(args.split, verified_only=False)
    else:
        gold = verified
    if len(gold) < 8:
        print(f"note: only {len(gold)} gold document(s) in split '{args.split}': every accuracy figure is preliminary (n = {len(gold)})")
    gmap = gold_doc_id_map(gold, manifest)
    missing = [d.doc_id for d in gold if d.doc_id not in gmap]
    if missing:
        print(f"warning: gold documents without a knowledge base: {missing}")
    names = args.name or list(EXPERIMENTS)
    need_llm = any(n not in ("gate_sensitivity",) for n in names)
    provider = None
    if need_llm:
        ok, msg = provider_status(args.provider)
        if not ok:
            raise SystemExit(f"LLM access is not configured: {msg}")
        provider = get_provider(args.provider)
    from pipeline.agent.retrieve import RetrievalConfig, Retriever
    from pipeline.preprocess.embeddings import EmbeddingCache

    ecache = EmbeddingCache()

    def factory(run_id: str, overrides: dict) -> Runner:
        cfg = RunConfig(run_id=run_id, provider=args.provider or "", watermark=watermark, max_calls=args.max_calls or config.MAX_CALLS_PER_RUN, max_cost_usd=args.max_cost or config.MAX_COST_USD_PER_RUN, **overrides)
        rcfg = RetrievalConfig(use_search_terms=cfg.use_search_terms, use_rerank=cfg.use_rerank)
        retriever = Retriever(rcfg, cache=ecache)
        return Runner(cfg, provider=provider, metadata=metadata, manifest=manifest, retriever=retriever, embedder=retriever.embedder, embedding_cache=ecache, usage=UsageTracker(cfg.max_calls, cfg.max_cost_usd, path=run_dir_for(run_id) / "usage.json"))

    rows = []
    for name in names:
        rows += run_experiment(name, args.split, runner_factory=factory, gold_docs=gold, metadata=metadata, doc_id_map=gmap, watermark=watermark)
    out = write_results_table(rows)
    fig = ablation_figure(rows)
    print(f"wrote {out} and {out.with_suffix('.md')}" + (f" and {fig}" if fig else ""))
    return 0


def cmd_export_review(args: argparse.Namespace) -> int:
    from pipeline.review.export import export_review

    p = export_review(config.LATEST_DIR if not args.run_id else config.RUNS_DIR / args.run_id, Path(args.out) if args.out else None, _metadata_map(), all_rows=args.all_rows)
    print(f"wrote {p}")
    return 0


def cmd_import_review(args: argparse.Namespace) -> int:
    from pipeline.review.import_ import import_review

    res = import_review(Path(args.file), config.LATEST_DIR, _metadata_map(), export_gold_to=(Path(args.gold_dir) if args.export_gold else None), gold_split=args.split)
    print(f"{res['n_decisions']} decision(s): {res['by_decision']}")
    for k, p in res["paths"].items():
        print(f"  {k}: {p}")
    if res["gold_files"]:
        print(f"  gold files: {', '.join(str(p) for p in res['gold_files'])} (verified=true)")
    print("overrides were NOT changed; review the proposal and diff before merging into pipeline/metadata/overrides.yaml")
    return 0


def _add_doc_selection(p: argparse.ArgumentParser, with_all: bool = True) -> None:
    p.add_argument("--docs", nargs="+", default=None, help="doc_ids or unique path fragments")
    p.add_argument("--pilot", action="store_true", help=f"the pilot set ({config.PILOT_SIZE} documents: verified dev gold first, else spread over countries / file types)")
    p.add_argument("--pilot-size", type=int, default=None)
    if with_all:
        p.add_argument("--all", action="store_true", help="every document in the manifest (full run; needs --confirm-full-run for `run`)")


def _add_agent_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-id", default=None)
    p.add_argument("--provider", default=None, help="openai | azure | ollama (default: config.LLM_PROVIDER / env LLM_PROVIDER)")
    p.add_argument("--model", default=None, help="answer model (default: ANSWER_MODEL)")
    p.add_argument("--extraction-model", default=None)
    p.add_argument("--tau", type=float, default=None, help="gate threshold (default: percentile of all pair scores)")
    p.add_argument("--tau-percentile", type=float, default=None)
    p.add_argument("--top-m", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--route-sectors", action="store_true")
    p.add_argument("--no-export-only", action="store_true", help="skip export-only indicators (on_site = false)")
    p.add_argument("--no-knowledge", action="store_true")
    p.add_argument("--no-search-terms", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--baseline", action="store_true", help="plain-RAG retrieval (benchmark)")
    p.add_argument("--no-novel", action="store_true")
    p.add_argument("--no-reverse-check", action="store_true")
    p.add_argument("--reranker", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--max-cost", type=float, default=None)
    p.add_argument("--force", action="store_true", help="ignore cached answers and finished documents")
    p.add_argument("--allow-unverified", action="store_true", help="debugging only: watermark outputs 'UNVERIFIED GOLD'")


def add_phase3_parsers(sub) -> None:
    p = sub.add_parser("select-pairs", help="retrieval gate (+ optional sector routing): write outputs/runs/<run_id>/selected_pairs.csv and print the cost estimate (no LLM)")
    _add_doc_selection(p)
    _add_agent_options(p)
    p.set_defaults(func=cmd_select_pairs)

    p = sub.add_parser("estimate", help="pairs, calls and cost for a document selection without calling the LLM")
    _add_doc_selection(p)
    _add_agent_options(p)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("run", help="full pipeline per document: select -> answer -> novel pass -> consolidate -> reverse check -> outputs (resumable)")
    _add_doc_selection(p)
    _add_agent_options(p)
    p.add_argument("--confirm-full-run", action="store_true", help="required with --all")
    p.add_argument("--copy-latest", action="store_true", help="also copy the outputs to outputs/latest/ (a confirmed full run does this automatically)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("answer", help="disclosure step only for the selected pairs of a run (resumable)")
    _add_agent_options(p)
    p.set_defaults(func=cmd_answer)

    p = sub.add_parser("novel-pass", help="extract the documents' own indicators from their logframe content (code first, LLM for text sections)")
    _add_doc_selection(p)
    _add_agent_options(p)
    p.add_argument("--no-llm", action="store_true", help="code-first extraction only")
    p.set_defaults(func=cmd_novel_pass)

    p = sub.add_parser("consolidate", help="link answers and extracted indicators, reverse check, write the output files of a run")
    _add_agent_options(p)
    p.add_argument("--copy-latest", action="store_true")
    p.set_defaults(func=cmd_consolidate)

    p = sub.add_parser("report", help="print one document's three-column table (document indicator | IndiKit indicator (ID) | match)")
    p.add_argument("--doc", required=True, help="doc_id or unique path fragment")
    p.add_argument("--run-id", default=None, help="default: outputs/latest")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("experiments", help="benchmark / ablations / model comparison / gate sensitivity / reverse-check value on the gold set")
    p.add_argument("--name", nargs="+", default=None, help="experiment names (default: all)")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--provider", default=None)
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--max-cost", type=float, default=None)
    p.add_argument("--allow-unverified", action="store_true")
    p.set_defaults(func=cmd_experiments)

    p = sub.add_parser("export-review", help="write outputs/latest/review.xlsx (one sheet per country, rows with needs_review = true)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--all-rows", action="store_true", help="export every matched row, not only needs_review")
    p.set_defaults(func=cmd_export_review)

    p = sub.add_parser("import-review", help="apply reviewer decisions: review_decisions.csv, document_indicators_reviewed.csv, proposed overrides diff, optional gold export")
    p.add_argument("--file", required=True)
    p.add_argument("--export-gold", action="store_true", help="also write reviewed documents as gold files (verified=true)")
    p.add_argument("--gold-dir", default=str(config.GOLD_DIR))
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.set_defaults(func=cmd_import_review)


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

    add_phase3_parsers(sub)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
