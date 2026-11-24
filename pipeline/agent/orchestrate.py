"""End-to-end run of the agent over documents (docs/phase_3.md Tasks 4-6, 8): pair selection -> LLM answering ->
novel-indicator pass -> consolidation -> reverse check -> outputs. Resumable: every finished pair is appended to
``answers.jsonl``, every finished document is stored as ``docs/<doc_id>.json`` and the usage tracker persists
``usage.json``; a restarted run skips what exists. A failed document never stops the run. Budget limits stop it
cleanly (``BudgetExceeded``). A full run over all documents needs ``confirm_full_run=True``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from pipeline import config
from pipeline.agent.answer import ProgressFile, answer_pairs
from pipeline.agent.consolidate import IndicatorIndex, consolidate, reverse_check
from pipeline.agent.llm import BudgetExceeded, LLMProvider, ResponseCache, UsageTracker, estimate_cost, model_role, provider_name
from pipeline.agent.models import DocumentResult, PairAnswer
from pipeline.agent.novel_pass import run_novel_pass
from pipeline.agent.outputs import copy_latest, write_run_outputs, write_run_report
from pipeline.agent.select import SelectionResult, load_pair_scores, read_selection, select_pairs, write_selection
from pipeline.metadata.models import IndicatorMetadata
from pipeline.preprocess.models import ManifestRow


@dataclass
class RunConfig:
    run_id: str
    provider: str = ""
    answer_model: str = ""
    extraction_model: str = ""
    use_knowledge: bool = True
    use_search_terms: bool = True
    use_rerank: bool = True
    kb_types: tuple[str, ...] = ("text", "outline", "table")
    top_k: int = config.RETRIEVAL_TOP_K
    top_n: int = config.RETRIEVAL_TOP_N
    baseline_retrieval: bool = False
    tau: Optional[float] = config.GATE_TAU
    tau_percentile: float = config.GATE_TAU_PERCENTILE
    top_m: int = config.GATE_TOP_M
    route_sectors: bool = config.ROUTE_SECTORS
    include_export_only: bool = config.INCLUDE_EXPORT_ONLY
    novel_pass: bool = True
    reverse_check: bool = True
    workers: int = config.LLM_MAX_WORKERS
    max_calls: int = config.MAX_CALLS_PER_RUN
    max_cost_usd: float = config.MAX_COST_USD_PER_RUN
    force: bool = False
    mode: str = "docs"            # docs | pilot | full
    prompt_version: str = config.PROMPT_VERSION
    rubric_version: str = config.RUBRIC_VERSION
    watermark: str = ""

    def __post_init__(self):
        self.provider = self.provider or provider_name()
        self.answer_model = self.answer_model or model_role("ANSWER_MODEL")
        self.extraction_model = self.extraction_model or model_role("EXTRACTION_MODEL") or self.answer_model

    def fingerprint(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in ("run_id", "workers", "max_calls", "max_cost_usd", "force", "mode", "watermark")}
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


class RunLock:
    """``RUNNING.pid`` in the run directory: refuses to start a run id that another live process is already working on
    (two processes on one run id write the same files and starve each other for CPU)."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "RUNNING.pid"

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip() or 0)
            except ValueError:
                pid = 0
            if pid and pid != os.getpid() and self._alive(pid):
                raise RuntimeError(f"run '{self.path.parent.name}' is already running in process {pid} (see {self.path}); stop it first "
                                   f"or wait for it - do not start the same run id twice")
        self.path.write_text(str(os.getpid()), encoding="utf-8")
        return self

    def __exit__(self, *exc):
        try:
            if self.path.exists() and self.path.read_text().strip() == str(os.getpid()):
                self.path.unlink()
        except OSError:
            pass
        return False


def run_dir_for(run_id: str, runs_dir: Path = config.RUNS_DIR) -> Path:
    return Path(runs_dir) / run_id


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def pilot_doc_ids(manifest: dict[str, ManifestRow], gold_docs: Optional[Sequence[tuple[str, bool, str]]] = None, size: int = config.PILOT_SIZE) -> list[str]:
    """Prefer verified dev gold documents; otherwise the first ``size`` documents with logframe content spread over
    countries and file types. ``gold_docs`` = (kb doc_id, verified, split)."""
    chosen: list[str] = []
    for doc_id, verified, split in gold_docs or []:
        if verified and split == "dev" and doc_id in manifest and doc_id not in chosen:
            chosen.append(doc_id)
    if len(chosen) >= size:
        return chosen[:size]
    cands = [r for r in sorted(manifest.values(), key=lambda r: r.relative_path) if r.parse_status != "failed" and (r.n_logframe_tables + r.n_logframe_sections) > 0 and r.doc_id not in chosen and r.region != "template"]
    seen_countries: set[str] = set()
    seen_types: set[str] = set()
    pool = list(cands)
    while pool and len(chosen) < size:
        pick = next((r for r in pool if r.country not in seen_countries and r.file_type not in seen_types), None) or next((r for r in pool if r.country not in seen_countries), None) or pool[0]
        chosen.append(pick.doc_id)
        seen_countries.add(pick.country)
        seen_types.add(pick.file_type)
        pool.remove(pick)
    return chosen


class Runner:
    def __init__(self, cfg: RunConfig, *, provider: Optional[LLMProvider], metadata: dict[str, IndicatorMetadata], manifest: dict[str, ManifestRow], retriever=None, retrieve_fn: Optional[Callable] = None,
                 embedder=None, embedding_cache=None, cache: Optional[ResponseCache] = None, usage: Optional[UsageTracker] = None, kb_dir: Path = config.KB_DIR, runs_dir: Path = config.RUNS_DIR,
                 pair_scores=None, references_fn: Optional[Callable[[str], list[str]]] = None):
        self.cfg = cfg
        self.provider = provider
        self.metadata = metadata
        self.manifest = manifest
        self.retriever = retriever
        self.retrieve_fn = retrieve_fn
        self.embedder = embedder if embedder is not None else (getattr(retriever, "embedder", None))
        self.embedding_cache = embedding_cache
        self.kb_dir = Path(kb_dir)
        self.run_dir = run_dir_for(cfg.run_id, runs_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "docs").mkdir(exist_ok=True)
        self.cache = cache if cache is not None else ResponseCache()
        self.usage = usage if usage is not None else UsageTracker(cfg.max_calls, cfg.max_cost_usd, path=self.run_dir / "usage.json")
        if self.provider is not None and getattr(self.provider, "usage", None) is None:
            self.provider.usage = self.usage
        self._pair_scores = pair_scores
        self._index: Optional[IndicatorIndex] = None
        self.references_fn = references_fn
        self.progress = ProgressFile(self.run_dir / "answers.jsonl")
        self.t0 = time.time()
        (self.run_dir / "run_config.json").write_text(json.dumps({**asdict(self.cfg), "fingerprint": self.cfg.fingerprint()}, indent=1, default=str), encoding="utf-8")

    # ----------------------------------------------------------------- selection
    @property
    def pair_scores(self):
        if self._pair_scores is None:
            self._pair_scores = load_pair_scores()
        return self._pair_scores

    def select(self, doc_ids: Sequence[str]) -> SelectionResult:
        sel = select_pairs(self.pair_scores, self.manifest, self.metadata, tau=self.cfg.tau, tau_percentile=self.cfg.tau_percentile, top_m=self.cfg.top_m, route_sectors=self.cfg.route_sectors,
                           include_export_only=self.cfg.include_export_only, doc_ids=doc_ids)
        write_selection(sel, self.run_dir)
        return sel

    def selection(self) -> "pd.DataFrame":  # noqa: F821
        return read_selection(self.run_dir)

    def estimate(self, selection: Optional[SelectionResult] = None, n_docs: Optional[int] = None) -> dict:
        """Pre-run estimate: pairs x average tokens, plus novel-pass and reverse-check calls, without calling the LLM."""
        sel_df = selection.selected if selection is not None else self.selection()
        n_pairs = int(len(sel_df))
        docs = sel_df["doc_id"].unique().tolist() if n_pairs else []
        n_docs = n_docs if n_docs is not None else len(docs)
        n_sections = sum(int(self.manifest[d].n_logframe_sections) + 1 for d in docs if d in self.manifest) if self.cfg.novel_pass else 0
        n_reverse = int(round(n_docs * 8)) if self.cfg.reverse_check else 0       # ~8 novel candidates per document (updated from the pilot)
        est_pairs = estimate_cost(n_pairs, self.cfg.answer_model)
        est_novel = estimate_cost(n_sections, self.cfg.extraction_model, config.EST_PROMPT_TOKENS_PER_EXTRACTION, config.EST_COMPLETION_TOKENS_PER_EXTRACTION)
        est_rev = estimate_cost(n_reverse, self.cfg.answer_model, 1200, 120)
        total_calls = n_pairs + n_sections + n_reverse
        total_cost = round(est_pairs["cost_usd"] + est_novel["cost_usd"] + est_rev["cost_usd"], 4)
        return {"documents": n_docs, "pairs": n_pairs, "novel_calls": n_sections, "reverse_calls": n_reverse, "calls": total_calls, "cost_usd": total_cost, "answer_model": self.cfg.answer_model,
                "extraction_model": self.cfg.extraction_model, "within_budget": total_calls <= self.cfg.max_calls and total_cost <= self.cfg.max_cost_usd, "max_calls": self.cfg.max_calls, "max_cost_usd": self.cfg.max_cost_usd,
                "detail": {"pairs": est_pairs, "novel": est_novel, "reverse": est_rev}}

    # ----------------------------------------------------------------- per-document steps
    def _doc_path(self, doc_id: str) -> Path:
        return self.run_dir / "docs" / f"{doc_id}.json"

    def _novel_path(self, doc_id: str) -> Path:
        p = self.run_dir / "novel"
        p.mkdir(exist_ok=True)
        return p / f"{doc_id}.json"

    def _retrieve_kwargs(self) -> dict:
        from pipeline.agent.retrieve import RetrievalConfig, retrieve_baseline

        kw: dict = dict(use_knowledge=self.cfg.use_knowledge, use_search_terms=self.cfg.use_search_terms, model=self.cfg.answer_model, cache=self.cache, force=self.cfg.force, kb_dir=self.kb_dir, top_k=self.cfg.top_k)
        if self.retrieve_fn is not None:
            kw["retrieve_fn"] = self.retrieve_fn
        elif self.cfg.baseline_retrieval:
            kw["retrieve_fn"] = lambda d, i: retrieve_baseline(d, i, top_k=self.cfg.top_k, embedder=self.embedder if getattr(self.embedder, "name", "") == config.BASELINE_EMBEDDING_MODEL else None, cache=self.embedding_cache)
        else:
            kw["retrieval_cfg"] = RetrievalConfig(top_n=self.cfg.top_n, top_k=self.cfg.top_k, kb_types=self.cfg.kb_types, use_search_terms=self.cfg.use_search_terms, use_rerank=self.cfg.use_rerank,
                                                  **({"embedder": self.retriever.cfg.embedder, "reranker": self.retriever.cfg.reranker} if self.retriever is not None else {}))
            kw["retriever"] = self.retriever
        return kw

    def answer_document(self, doc_id: str, indikit_ids: Sequence[str]) -> tuple[list[PairAnswer], Optional[BudgetExceeded]]:
        pairs = [(doc_id, self.metadata[i]) for i in indikit_ids if i in self.metadata]
        if self.provider is None:
            raise RuntimeError("no LLM provider: configure LLM_PROVIDER or pass a provider")
        return answer_pairs(pairs, provider=self.provider, progress=self.progress, workers=self.cfg.workers, **self._retrieve_kwargs())

    def novel_document(self, doc_id: str):
        p = self._novel_path(doc_id)
        if p.exists() and not self.cfg.force:
            data = json.loads(p.read_text(encoding="utf-8"))
            from pipeline.agent.models import DocIndicator
            from pipeline.agent.novel_pass import NovelPassResult

            return NovelPassResult(doc_id=doc_id, indicators=[DocIndicator.model_validate(d) for d in data["indicators"]], stats=data.get("stats", {}), errors=data.get("errors", []), budget_stop=data.get("budget_stop"))
        res = run_novel_pass(doc_id, provider=self.provider, model=self.cfg.extraction_model, cache=self.cache, kb_dir=self.kb_dir, use_llm=self.provider is not None, force=self.cfg.force)
        if res.budget_stop is None:
            p.write_text(json.dumps({"indicators": [d.model_dump(mode="json") for d in res.indicators], "stats": res.stats, "errors": res.errors, "budget_stop": res.budget_stop}, ensure_ascii=False, indent=1), encoding="utf-8")
        return res

    def indicator_index(self) -> Optional[IndicatorIndex]:
        if self._index is None and self.embedder is not None:
            self._index = IndicatorIndex(self.metadata, self.embedder, self.embedding_cache)
        return self._index

    def _references(self, doc_id: str) -> list[str]:
        if self.references_fn is not None:
            return self.references_fn(doc_id)
        try:
            from pipeline.preprocess.references import load_references

            return [r.referenced_document for r in load_references(doc_id, self.kb_dir) if r.kind in ("logframe", "indicator_table", "tracker")]
        except Exception:
            return []

    def process_document(self, doc_id: str, indikit_ids: Sequence[str], *, skipped_reason: Optional[str] = None) -> DocumentResult:
        t0 = time.time()
        row = self.manifest.get(doc_id)
        res = DocumentResult(doc_id=doc_id, relative_path=row.relative_path if row else "", country=row.country if row else "", region=row.region if row else "", language=row.language if row else "",
                             file_type=row.file_type if row else "", parse_status=row.parse_status if row else "", indicators_elsewhere=bool(row.indicators_elsewhere) if row else False,
                             indicators_also_elsewhere=bool(row.indicators_also_elsewhere) if row else False, referenced_documents=self._references(doc_id), skipped_reason=skipped_reason, pairs_selected=len(indikit_ids))
        if skipped_reason:
            return res
        try:
            answers, stop = self.answer_document(doc_id, indikit_ids)
            res.pairs_asked = sum(1 for a in answers if not a.cached and a.error != "no evidence retrieved")
            res.pairs_cached = sum(1 for a in answers if a.cached)
            res.pairs_present = sum(1 for a in answers if a.disclosure == "present")
            res.pairs_failed = sum(1 for a in answers if a.disclosure == "failed")
            res.unsupported_dropped = sum(len(a.unsupported) for a in answers)
            res.prompt_tokens = sum(a.prompt_tokens for a in answers)
            res.completion_tokens = sum(a.completion_tokens for a in answers)
            res.llm_calls = sum(1 for a in answers if not a.cached and a.prompt_tokens > 0)
            res.cost_usd = self.usage.cost_of(self.cfg.answer_model, res.prompt_tokens, res.completion_tokens)
            res.errors += [f"{a.indikit_id}: {a.error}" for a in answers if a.disclosure == "failed"]
            if stop is not None:
                res.budget_stop = str(stop)
                res.errors.append(f"budget stop: {stop}")
                raise stop
            indicators = []
            if self.cfg.novel_pass:
                nres = self.novel_document(doc_id)
                indicators = nres.indicators
                res.novel_stats = nres.stats
                res.errors += nres.errors
                if nres.budget_stop:
                    res.budget_stop = nres.budget_stop
                    raise BudgetExceeded(nres.budget_stop)
            cons = consolidate(doc_id, answers, indicators, self.metadata)
            if self.cfg.reverse_check:
                idx = self.indicator_index()
                if idx is not None:
                    st = reverse_check(doc_id, cons, idx, provider=self.provider, model=self.cfg.answer_model, cache=self.cache, force=self.cfg.force)
                    res.reverse_stats = {"candidates": st.candidates, "checked": st.checked, "recovered": st.recovered, "calls": st.calls, "cached": st.cached, "failed": st.failed}
                    res.errors += st.errors
                    if st.budget_stop:
                        res.budget_stop = st.budget_stop
                        raise BudgetExceeded(st.budget_stop)
                else:
                    res.errors.append("reverse check skipped: no embedder")
            res.consolidated = cons
            res.export_only_matches = sum(1 for c in cons if c.primary and c.primary.indikit_id in self.metadata and not self.metadata[c.primary.indikit_id].on_site)
        except BudgetExceeded:
            res.seconds = round(time.time() - t0, 1)
            raise
        except Exception as exc:
            res.errors.append(f"document failed: {type(exc).__name__}: {str(exc)[:300]}")
        res.seconds = round(time.time() - t0, 1)
        self._doc_path(doc_id).write_text(res.model_dump_json(indent=1), encoding="utf-8")
        return res

    # ----------------------------------------------------------------- whole run
    def load_results(self) -> list[DocumentResult]:
        out = []
        for p in sorted((self.run_dir / "docs").glob("*.json")):
            try:
                out.append(DocumentResult.model_validate_json(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def run(self, doc_ids: Sequence[str], *, confirm_full_run: bool = False, progress: Optional[Callable[[str], None]] = print, check_estimate: bool = True) -> tuple[list[DocumentResult], Optional[str]]:
        if self.cfg.mode == "full" and not confirm_full_run:
            raise RuntimeError("a full run over all documents requires --confirm-full-run")
        sel = self.select(doc_ids)
        est = self.estimate(sel)
        if progress:
            progress(f"estimate: {est['documents']} documents, {est['pairs']} pairs, ~{est['calls']} calls, ~${est['cost_usd']:.2f} with {est['answer_model']} (limits: {est['max_calls']} calls, ${est['max_cost_usd']})")
        if check_estimate and not est["within_budget"]:
            raise BudgetExceeded(f"estimate exceeds the run limits ({est['calls']} calls / ${est['cost_usd']:.2f} vs {est['max_calls']} / ${est['max_cost_usd']}); reduce the selection or raise the limits in config")
        by_doc = {d: g["indikit_id"].tolist() for d, g in sel.selected.groupby("doc_id")} if len(sel.selected) else {}
        skipped = dict(sel.skipped_documents)
        # finished documents are skipped on resume, except those with failed pairs or a document-level failure:
        # they are processed again (successful answers, novel-pass and reverse-check calls come from the caches)
        done = {r.doc_id for r in self.load_results() if r.skipped_reason or (r.pairs_failed == 0 and not any(e.startswith("document failed") for e in r.errors))} if not self.cfg.force else set()
        results: list[DocumentResult] = []
        stop_reason: Optional[str] = None
        for i, doc_id in enumerate(doc_ids, start=1):
            if doc_id in done:
                continue
            try:
                r = self.process_document(doc_id, by_doc.get(doc_id, []), skipped_reason=skipped.get(doc_id))
                if skipped.get(doc_id):
                    self._doc_path(doc_id).write_text(r.model_dump_json(indent=1), encoding="utf-8")
                results.append(r)
                if progress:
                    progress(f"[{i}/{len(doc_ids)}] {doc_id}: {len(r.consolidated)} indicators, {r.pairs_present} present pairs, {r.llm_calls} calls, {r.seconds}s" + (f" SKIPPED ({r.skipped_reason})" if r.skipped_reason else "") + (f" ERRORS {len(r.errors)}" if r.errors else ""))
            except BudgetExceeded as exc:
                stop_reason = str(exc)
                if progress:
                    progress(f"stopped: {stop_reason}")
                break
        self.usage.write()
        return self.load_results(), stop_reason

    def finalize(self, *, copy_to_latest: bool = False, extrapolate_to: Optional[int] = None, note: str = "") -> dict:
        results = self.load_results()
        embed_fn = (lambda texts: (self.embedding_cache.embed(self.embedder, texts) if self.embedding_cache is not None else self.embedder.embed(texts))) if self.embedder is not None else None
        paths = write_run_outputs(results, self.metadata, self.run_dir, embed_fn=embed_fn, note=note or self.cfg.watermark)
        usage = self.usage.to_dict()
        processed = [r for r in results if not r.skipped_reason]
        pairs_asked = sum(r.pairs_asked for r in processed)
        runtime = round(time.time() - self.t0, 1)
        info = {"run_id": self.cfg.run_id, "config_fingerprint": self.cfg.fingerprint(), "provider": self.cfg.provider or "(none)", "answer_model": self.cfg.answer_model, "extraction_model": self.cfg.extraction_model,
                "prompt_version": self.cfg.prompt_version, "rubric_version": self.cfg.rubric_version, "novel_prompt_version": config.NOVEL_PROMPT_VERSION, "reverse_prompt_version": config.REVERSE_CHECK_PROMPT_VERSION,
                "retrieval": f"{'baseline' if self.cfg.baseline_retrieval else 'enhanced'} top_n={self.cfg.top_n} top_k={self.cfg.top_k} rerank={self.cfg.use_rerank} search_terms={self.cfg.use_search_terms} kb_types={self.cfg.kb_types}",
                "gate": f"tau={self.cfg.tau if self.cfg.tau is not None else f'p{self.cfg.tau_percentile:g}'} top_m={self.cfg.top_m}", "route_sectors": self.cfg.route_sectors, "include_export_only": self.cfg.include_export_only,
                "mode": self.cfg.mode, "documents": len(results), "documents_processed": len(processed), "documents_skipped": sum(1 for r in results if r.skipped_reason), "documents_failed": sum(1 for r in processed if any(e.startswith("document failed") for e in r.errors)),
                "pairs_selected": sum(r.pairs_selected for r in results), "pairs_asked": pairs_asked, "pairs_cached": sum(r.pairs_cached for r in processed), "pairs_present": sum(r.pairs_present for r in processed),
                "pairs_failed": sum(r.pairs_failed for r in processed), "unsupported_dropped": sum(r.unsupported_dropped for r in processed), "novel_llm_calls": sum(int(r.novel_stats.get("llm_calls", 0)) for r in processed),
                "reverse_check_candidates": sum(int(r.reverse_stats.get("candidates", 0)) for r in processed), "reverse_check_recovered": sum(int(r.reverse_stats.get("recovered", 0)) for r in processed),
                "export_only_matches": sum(r.export_only_matches for r in processed), "usage": usage, "runtime_seconds": runtime, "watermark": self.cfg.watermark,
                "seconds_per_pair": round(sum(r.seconds for r in processed) / pairs_asked, 2) if pairs_asked else None, "cost_per_pair_usd": round(usage["cost_usd"] / usage["calls"], 5) if usage["calls"] else None,
                "cost_per_document_usd": round(usage["cost_usd"] / len(processed), 4) if processed else None, "errors": [e for r in results for e in r.errors]}
        if extrapolate_to:
            per_doc_pairs = (sum(r.pairs_selected for r in processed) / len(processed)) if processed else 0
            per_doc_calls = (usage["calls"] / len(processed)) if processed else 0
            per_doc_cost = (usage["cost_usd"] / len(processed)) if processed else 0
            per_doc_sec = (sum(r.seconds for r in processed) / len(processed)) if processed else 0
            info["extrapolation"] = {"documents": extrapolate_to, "pairs": int(round(per_doc_pairs * extrapolate_to)), "calls": int(round(per_doc_calls * extrapolate_to)), "cost_usd": round(per_doc_cost * extrapolate_to, 2),
                                     "hours": round(per_doc_sec * extrapolate_to / 3600, 2), "within_MAX_COST_USD_PER_RUN": per_doc_cost * extrapolate_to <= self.cfg.max_cost_usd}
            if not info["extrapolation"]["within_MAX_COST_USD_PER_RUN"]:
                info["recommendation"] = ("The extrapolated cost exceeds MAX_COST_USD_PER_RUN. Do not raise the limit; choose a cheaper configuration: a stricter gate (higher tau percentile / smaller top-M), "
                                          "sector routing (--route-sectors) or a smaller ANSWER_MODEL, with the expected recall from outputs/eval/gate_sensitivity/, and ask the owner to decide.")
        write_run_report(self.run_dir, info)
        if copy_to_latest:
            copy_latest(self.run_dir)
        return {"paths": paths, "info": info}
