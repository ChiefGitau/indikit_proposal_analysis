"""Evaluation and experiments (paper Section 5.1; docs/phase_3.md Task 7).

Scoring compares the run's predictions with the gold set using the Phase 1 metrics (``acc_dc``, precision / recall /
F1 of "present", ``acc_de``, ``match_type_accuracy``, ``reference_detection_accuracy``) plus novel-pass precision /
recall and extraction recall of document indicators, per document and overall, with n.

Experiments (each writes ``outputs/eval/<experiment>/metrics.json`` + ``predictions.csv``):

| experiment          | configuration                                                            |
|---------------------|--------------------------------------------------------------------------|
| benchmark           | plain-RAG retrieval (flat chunks, MiniLM, no re-rank), no knowledge, no search terms |
| enhanced_rag        | Phase 2 retrieval (three KBs + re-rank), no knowledge, no search terms   |
| plus_knowledge      | enhanced + use_knowledge                                                 |
| plus_search_term    | enhanced + knowledge + search terms (= full method)                      |
| model_comparison    | full method with ANSWER_MODEL vs STRONG_MODEL (and any further models)   |
| gate_sensitivity    | recall of gold-present pairs and pairs asked across tau / M, +/- routing |
| reverse_check_value | matches recovered by the reverse check and their cost                    |

Protocol: tune on dev only; run test once at the end with the frozen configuration. Without verified gold the live
experiments are skipped (``--allow-unverified`` watermarks every output "UNVERIFIED GOLD").
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from pipeline import config
from pipeline.agent.models import DocumentResult
from pipeline.eval.gold import GoldDocument, load_gold
from pipeline.eval.metrics import PairLabel, PairPrediction, ReferenceLabel, ReferencePrediction, acc_dc, acc_de, match_type_accuracy, precision_recall_present, reference_detection_accuracy, values_equal
from pipeline.metadata.models import IndicatorMetadata

EXPERIMENTS: tuple[str, ...] = ("benchmark", "enhanced_rag", "plus_knowledge", "plus_search_term", "model_comparison", "gate_sensitivity", "reverse_check_value")

EXPERIMENT_CONFIGS: dict[str, dict] = {
    "benchmark": dict(baseline_retrieval=True, use_knowledge=False, use_search_terms=False, use_rerank=False),
    "enhanced_rag": dict(baseline_retrieval=False, use_knowledge=False, use_search_terms=False, use_rerank=True),
    "plus_knowledge": dict(baseline_retrieval=False, use_knowledge=True, use_search_terms=False, use_rerank=True),
    "plus_search_term": dict(baseline_retrieval=False, use_knowledge=True, use_search_terms=True, use_rerank=True),
}
PAPER_NAMES = {"benchmark": "Benchmark (basic RAG)", "enhanced_rag": "Enhanced-RAG", "plus_knowledge": "+ expert knowledge", "plus_search_term": "+ search terms (full method)"}

RESULT_COLUMNS: tuple[str, ...] = ("experiment", "model", "split", "n_documents", "n_pairs", "n_present", "acc_dc", "precision", "recall", "f1", "acc_de", "match_type_acc", "reference_detection",
                                   "novel_precision", "novel_recall", "extraction_recall", "llm_calls", "cost_usd", "minutes", "watermark")
ERROR_CATEGORIES = ("retrieval_miss", "gate_miss", "llm_false_negative", "llm_false_positive", "wrong_match_type", "wrong_value", "parse_problem")
UNVERIFIED_WATERMARK = "UNVERIFIED GOLD"


# --------------------------------------------------------------------------- #
# Gold helpers
# --------------------------------------------------------------------------- #
def gold_documents(split: Optional[str] = None, *, verified_only: bool = True, gold_dir: Path = config.GOLD_DIR) -> list[GoldDocument]:
    docs = load_gold(gold_dir)
    out = []
    for d in docs:
        if d.entry is None:
            continue
        if split and d.entry.split != split:
            continue
        if verified_only and not d.entry.verified:
            continue
        out.append(d)
    return out


def gold_doc_id_map(gold_docs: Sequence[GoldDocument], manifest: Mapping[str, object]) -> dict[str, str]:
    """Gold ``doc_id`` -> KB ``doc_id`` via ``source_file`` (relative to ``proposals/``) or an identical id."""
    by_rel = {}
    for row in manifest.values():
        rel = getattr(row, "relative_path", "")
        by_rel[rel] = row.doc_id
        by_rel[rel.split("proposals/", 1)[-1]] = row.doc_id
    out = {}
    for d in gold_docs:
        if d.doc_id in manifest:
            out[d.doc_id] = d.doc_id
        elif d.entry and d.entry.source_file in by_rel:
            out[d.doc_id] = by_rel[d.entry.source_file]
        elif d.entry and f"proposals/{d.entry.source_file}" in by_rel:
            out[d.doc_id] = by_rel[f"proposals/{d.entry.source_file}"]
    return out


def gold_pair_labels(gold_docs: Sequence[GoldDocument], metadata: Mapping[str, IndicatorMetadata], doc_id_map: Mapping[str, str], pair_universe: Optional[set[tuple[str, str]]] = None) -> list[PairLabel]:
    """Present labels from the gold rows; every other (document, metadata indicator) pair is absent. ``pair_universe``
    restricts the absent pairs (e.g. to the selected pairs) so all experiments score the same pairs."""
    labels: dict[tuple[str, str], PairLabel] = {}
    for d in gold_docs:
        kb = doc_id_map.get(d.doc_id)
        if kb is None:
            continue
        for r in d.rows:
            if r.indikit_id and r.indikit_id in metadata:
                key = (kb, r.indikit_id)
                prev = labels.get(key)
                if prev is None or (prev.match_type != "verbatim" and r.match_type == "verbatim"):
                    labels[key] = PairLabel(doc_id=kb, indikit_id=r.indikit_id, present=True, baseline=r.baseline, target=r.target, match_type=r.match_type)
        for iid in metadata:
            key = (kb, iid)
            if key in labels:
                continue
            if pair_universe is not None and key not in pair_universe:
                continue
            labels[key] = PairLabel(doc_id=kb, indikit_id=iid, present=False)
    return list(labels.values())


def gold_present_pairs(gold_docs: Sequence[GoldDocument], doc_id_map: Mapping[str, str]) -> set[tuple[str, str]]:
    return {(doc_id_map[d.doc_id], r.indikit_id) for d in gold_docs if d.doc_id in doc_id_map for r in d.rows if r.indikit_id}


# --------------------------------------------------------------------------- #
# Predictions from run results
# --------------------------------------------------------------------------- #
def predictions_from_results(results: Sequence[DocumentResult]) -> list[PairPrediction]:
    preds: dict[tuple[str, str], PairPrediction] = {}
    for r in results:
        for c in r.consolidated:
            for m in c.matches:
                key = (r.doc_id, m.indikit_id)
                prev = preds.get(key)
                cand = PairPrediction(doc_id=r.doc_id, indikit_id=m.indikit_id, present=True, baseline=m.baseline if m.baseline is not None else c.indicator.baseline, target=m.target if m.target is not None else c.indicator.target, match_type=m.match_type)
                if prev is None or (prev.match_type != "verbatim" and m.match_type == "verbatim"):
                    preds[key] = cand
    return list(preds.values())


def reference_predictions(results: Sequence[DocumentResult]) -> list[ReferencePrediction]:
    return [ReferencePrediction(doc_id=r.doc_id, referenced_document=(r.referenced_documents[0] if r.referenced_documents else None)) for r in results]


def _fuzzy_in(text: str, candidates: Iterable[str], threshold: int = 80) -> bool:
    from pipeline.agent.answer import text_similarity

    return any(text_similarity(text, c) >= threshold for c in candidates)


def novel_and_extraction_metrics(gold_docs: Sequence[GoldDocument], results: Sequence[DocumentResult], doc_id_map: Mapping[str, str]) -> dict:
    """Novel-pass precision / recall (gold rows with match_type none) and extraction recall of all gold document indicators."""
    by_doc = {r.doc_id: r for r in results}
    tp = fp = fn = 0
    found = total = 0
    for d in gold_docs:
        kb = doc_id_map.get(d.doc_id)
        r = by_doc.get(kb)
        if r is None:
            continue
        extracted = [c.indicator.indicator_text for c in r.consolidated if c.indicator.is_indicator]
        pred_novel = [c.indicator.indicator_text for c in r.consolidated if c.indicator.is_indicator and not c.matches]
        gold_novel = [row.indicator_text for row in d.rows if row.match_type == "none"]
        for row in d.rows:
            total += 1
            found += int(_fuzzy_in(row.indicator_text, extracted))
        for g in gold_novel:
            if _fuzzy_in(g, pred_novel):
                tp += 1
            else:
                fn += 1
        for p in pred_novel:
            if not _fuzzy_in(p, gold_novel):
                fp += 1
    return {"novel_precision": tp / (tp + fp) if (tp + fp) else math.nan, "novel_recall": tp / (tp + fn) if (tp + fn) else math.nan, "novel_tp": tp, "novel_fp": fp, "novel_fn": fn,
            "extraction_recall": found / total if total else math.nan, "gold_document_indicators": total}


def score(gold_docs: Sequence[GoldDocument], results: Sequence[DocumentResult], metadata: Mapping[str, IndicatorMetadata], doc_id_map: Mapping[str, str], pair_universe: Optional[set[tuple[str, str]]] = None) -> dict:
    labels = gold_pair_labels(gold_docs, metadata, doc_id_map, pair_universe)
    preds = predictions_from_results(results)
    if pair_universe is not None:
        gold_keys = {l.key for l in labels}
        preds = [p for p in preds if p.key in pair_universe or p.key in gold_keys]
    pr = precision_recall_present(labels, preds)
    overall = {"n_documents": len({l.doc_id for l in labels}), "n_pairs": len(labels), "n_present": sum(1 for l in labels if l.present), "acc_dc": acc_dc(labels, preds), **{k: pr[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
               "acc_de": acc_de(labels, preds), "match_type_acc": match_type_accuracy(labels, preds),
               "reference_detection": reference_detection_accuracy([ReferenceLabel(doc_id=doc_id_map[d.doc_id], referenced_document=d.entry.indicators_in_other_document) for d in gold_docs if d.doc_id in doc_id_map and d.entry], reference_predictions(results)),
               **novel_and_extraction_metrics(gold_docs, results, doc_id_map)}
    per_doc = {}
    for d in gold_docs:
        kb = doc_id_map.get(d.doc_id)
        if kb is None:
            continue
        dl = [l for l in labels if l.doc_id == kb]
        dp = [p for p in preds if p.doc_id == kb]
        dpr = precision_recall_present(dl, dp)
        per_doc[d.doc_id] = {"kb_doc_id": kb, "n_pairs": len(dl), "n_present": sum(1 for l in dl if l.present), "acc_dc": acc_dc(dl, dp), "precision": dpr["precision"], "recall": dpr["recall"], "f1": dpr["f1"], "acc_de": acc_de(dl, dp), "match_type_acc": match_type_accuracy(dl, dp)}
    return {"overall": overall, "per_document": per_doc}


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def error_analysis(gold_docs: Sequence[GoldDocument], results: Sequence[DocumentResult], metadata: Mapping[str, IndicatorMetadata], doc_id_map: Mapping[str, str], selected_pairs: set[tuple[str, str]],
                   evidence_check: Optional[Callable[[str, str, str], bool]] = None, manifest: Optional[Mapping[str, object]] = None) -> list[dict]:
    """One row per wrong pair with a category. ``evidence_check(kb_doc_id, indikit_id, gold_text) -> bool`` says whether the
    gold evidence was retrieved for that pair (``retrieval_miss`` when False)."""
    labels = {l.key: l for l in gold_pair_labels(gold_docs, metadata, doc_id_map)}
    preds = {p.key: p for p in predictions_from_results(results)}
    gold_text = {}
    for d in gold_docs:
        kb = doc_id_map.get(d.doc_id)
        for r in d.rows:
            if r.indikit_id and kb:
                gold_text.setdefault((kb, r.indikit_id), r.indicator_text)
    rows = []
    for key, l in labels.items():
        p = preds.get(key)
        doc, iid = key
        status = (getattr(manifest.get(doc), "parse_status", "ok") if manifest else "ok")
        if l.present and (p is None or not p.present):
            if status != "ok":
                cat = "parse_problem"
            elif key not in selected_pairs:
                cat = "gate_miss"
            elif evidence_check is not None and not evidence_check(doc, iid, gold_text.get(key, "")):
                cat = "retrieval_miss"
            else:
                cat = "llm_false_negative"
            rows.append({"doc_id": doc, "indikit_id": iid, "category": cat, "gold_present": True, "predicted_present": False, "gold_text": gold_text.get(key, ""), "gold_match_type": l.match_type, "predicted_match_type": "", "detail": ""})
        elif l.present and p is not None and p.present:
            if l.match_type != p.match_type:
                rows.append({"doc_id": doc, "indikit_id": iid, "category": "wrong_match_type", "gold_present": True, "predicted_present": True, "gold_text": gold_text.get(key, ""), "gold_match_type": l.match_type, "predicted_match_type": p.match_type, "detail": ""})
            if not (values_equal(l.baseline, p.baseline) and values_equal(l.target, p.target)):
                rows.append({"doc_id": doc, "indikit_id": iid, "category": "wrong_value", "gold_present": True, "predicted_present": True, "gold_text": gold_text.get(key, ""), "gold_match_type": l.match_type, "predicted_match_type": p.match_type,
                             "detail": f"gold baseline/target {l.baseline!r}/{l.target!r} vs {p.baseline!r}/{p.target!r}"})
        elif not l.present and p is not None and p.present:
            rows.append({"doc_id": doc, "indikit_id": iid, "category": "llm_false_positive", "gold_present": False, "predicted_present": True, "gold_text": "", "gold_match_type": "", "predicted_match_type": p.match_type, "detail": ""})
    return rows


def write_errors(rows: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id", "indikit_id", "category", "gold_present", "predicted_present", "gold_text", "gold_match_type", "predicted_match_type", "detail"])
        w.writeheader()
        w.writerows(rows)
    return path


def error_counts(rows: list[dict]) -> dict[str, int]:
    c = {k: 0 for k in ERROR_CATEGORIES}
    for r in rows:
        c[r["category"]] = c.get(r["category"], 0) + 1
    return c


TOP_FIXES = {
    "retrieval_miss": "add search terms / overrides for the missed indicators, raise top-N or top-k, check the logframe boost",
    "gate_miss": "lower tau or raise top-M for the affected sectors (see gate_sensitivity), or enable sector routing with a larger M",
    "llm_false_negative": "tighten the knowledge note (what counts) and add a worked example to the rubric",
    "llm_false_positive": "add 'what does not count' to the knowledge note; strengthen the 'same topic is not enough' rule",
    "wrong_match_type": "add verbatim / close examples for the indicator to the rubric or knowledge note",
    "wrong_value": "check table column mapping (baseline / target) and the prompt instruction to copy values as written",
    "parse_problem": "fix the parser for the document type (or run OCR) before anything else",
}


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def write_metrics(exp_dir: Path, metrics: dict, predictions: Sequence[PairPrediction], watermark: str = "") -> Path:
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    payload = {**metrics, "watermark": watermark}
    (exp_dir / "metrics.json").write_text(json.dumps(payload, indent=1, default=lambda v: None if isinstance(v, float) and math.isnan(v) else str(v)), encoding="utf-8")
    with (exp_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id", "indikit_id", "present", "match_type", "baseline", "target"])
        w.writeheader()
        for p in predictions:
            w.writerow({"doc_id": p.doc_id, "indikit_id": p.indikit_id, "present": p.present, "match_type": p.match_type or "", "baseline": p.baseline or "", "target": p.target or ""})
    return exp_dir / "metrics.json"


def run_experiment(name: str, split: str = "dev", *, runner_factory: Callable[[str, dict], object], gold_docs: Sequence[GoldDocument], metadata: Mapping[str, IndicatorMetadata], doc_id_map: Mapping[str, str],
                   eval_dir: Path = config.EVAL_OUTPUT_DIR, models: Optional[Sequence[str]] = None, watermark: str = "", pair_scores=None) -> list[dict]:
    """Run one experiment. ``runner_factory(run_id, overrides) -> Runner`` builds an orchestrator with the given config
    overrides (baseline_retrieval, use_knowledge, use_search_terms, use_rerank, answer_model). Returns result rows."""
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment '{name}'; choose from {EXPERIMENTS}")
    eval_dir = Path(eval_dir)
    doc_ids = [doc_id_map[d.doc_id] for d in gold_docs if d.doc_id in doc_id_map]
    rows: list[dict] = []
    if name == "gate_sensitivity":
        from pipeline.agent.select import tune_gate

        if pair_scores is None:
            from pipeline.agent.select import load_pair_scores

            pair_scores = load_pair_scores()
        curve, best = tune_gate(pair_scores, gold_present_pairs(gold_docs, doc_id_map), metadata=dict(metadata))
        d = eval_dir / name
        d.mkdir(parents=True, exist_ok=True)
        curve.to_csv(d / "curve.csv", index=False)
        (d / "metrics.json").write_text(json.dumps({"split": split, "best": best, "n_gold_pairs": len(gold_present_pairs(gold_docs, doc_id_map)), "watermark": watermark}, indent=1, default=str), encoding="utf-8")
        return [{"experiment": name, "split": split, "best": best, "watermark": watermark}]
    variants: list[tuple[str, dict]] = []
    if name == "model_comparison":
        models = list(models or [m for m in (config.ANSWER_MODEL, config.STRONG_MODEL) if m])
        for m in models:
            variants.append((m, {**EXPERIMENT_CONFIGS["plus_search_term"], "answer_model": m}))
    elif name == "reverse_check_value":
        variants.append(("", {**EXPERIMENT_CONFIGS["plus_search_term"], "reverse_check": True}))
    else:
        variants.append(("", EXPERIMENT_CONFIGS[name]))
    for model, overrides in variants:
        run_id = f"exp-{name}-{split}" + (f"-{model}" if model else "")
        runner = runner_factory(run_id, overrides)
        import time

        t0 = time.time()
        results, stop = runner.run(doc_ids, progress=None)
        runner.finalize(note=watermark)
        sel = runner.selection()
        universe = set(zip(sel["doc_id"], sel["indikit_id"]))
        sc = score(gold_docs, results, metadata, doc_id_map, pair_universe=universe)
        preds = predictions_from_results(results)
        usage = runner.usage.to_dict()
        row = {"experiment": name, "model": model or runner.cfg.answer_model, "split": split, **{k: sc["overall"].get(k) for k in ("n_documents", "n_pairs", "n_present", "acc_dc", "precision", "recall", "f1", "acc_de", "match_type_acc", "reference_detection", "novel_precision", "novel_recall", "extraction_recall")},
               "llm_calls": usage["calls"], "cost_usd": usage["cost_usd"], "minutes": round((time.time() - t0) / 60, 2), "watermark": watermark, "budget_stop": stop}
        if name == "reverse_check_value":
            row["reverse_recovered"] = sum(int(r.reverse_stats.get("recovered", 0)) for r in results)
            row["reverse_candidates"] = sum(int(r.reverse_stats.get("candidates", 0)) for r in results)
            row["reverse_calls"] = sum(int(r.reverse_stats.get("calls", 0)) for r in results)
            row["reverse_cost_usd"] = usage["per_model"].get(runner.cfg.answer_model, {}).get("cost_usd", 0.0)
        exp_dir = eval_dir / name / (model if model else "") if model else eval_dir / name
        write_metrics(exp_dir, {**row, "per_document": sc["per_document"]}, preds, watermark)
        rows.append(row)
    return rows


def run_all(split: str = "dev", **kw) -> list[dict]:
    rows: list[dict] = []
    for name in EXPERIMENTS:
        rows += run_experiment(name, split, **kw)
    return rows


# --------------------------------------------------------------------------- #
# Results table, figure, error section
# --------------------------------------------------------------------------- #
def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_results_table(rows: Iterable[Mapping[str, object]], path: Path = config.EVAL_OUTPUT_DIR / "results.csv", *, errors: Optional[list[dict]] = None, changelog: str = "") -> Path:
    """``results.csv`` plus ``results.md``: a Table-4-like model table (models x Acc_DE, Acc_DC), the ablation table
    (Figure 3), the gate / reverse-check additions and the error-analysis section."""
    rows = [dict(r) for r in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESULT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in RESULT_COLUMNS})
    md = ["# Evaluation results", ""]
    marks = {r.get("watermark") for r in rows if r.get("watermark")}
    if marks:
        md += [f"**{' / '.join(sorted(marks))}** - accuracy figures below are not based on verified labels.", ""]
    prelim = [r for r in rows if r.get("n_documents") is not None and 0 < int(r.get("n_documents") or 0) < 8]
    if prelim:
        md += ["Figures marked *preliminary (n = ...)* come from fewer than 8 verified documents.", ""]
    model_rows = [r for r in rows if r.get("experiment") in ("model_comparison", "plus_search_term")]
    if model_rows:
        md += ["## Model comparison (full method; paper Table 4)", "", "| model | split | n docs | n pairs | Acc_DE | Acc_DC | precision | recall | F1 | match-type acc | calls | cost USD |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in model_rows:
            n = int(r.get("n_documents") or 0)
            md.append(f"| {r.get('model')} | {r.get('split')} | {n}{' (preliminary)' if 0 < n < 8 else ''} | {r.get('n_pairs')} | {_fmt(r.get('acc_de'))} | {_fmt(r.get('acc_dc'))} | {_fmt(r.get('precision'))} | {_fmt(r.get('recall'))} | {_fmt(r.get('f1'))} | {_fmt(r.get('match_type_acc'))} | {r.get('llm_calls')} | {_fmt(r.get('cost_usd'))} |")
        md.append("")
    abl = [r for r in rows if r.get("experiment") in EXPERIMENT_CONFIGS]
    if abl:
        md += ["## Ablation (paper Figure 3)", "", "| configuration | split | n docs | Acc_DC | Acc_DE | precision | recall | F1 | novel P / R | extraction recall | calls | cost USD |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(abl, key=lambda r: list(EXPERIMENT_CONFIGS).index(r["experiment"])):
            md.append(f"| {PAPER_NAMES.get(r['experiment'], r['experiment'])} | {r.get('split')} | {r.get('n_documents')} | {_fmt(r.get('acc_dc'))} | {_fmt(r.get('acc_de'))} | {_fmt(r.get('precision'))} | {_fmt(r.get('recall'))} | {_fmt(r.get('f1'))} | {_fmt(r.get('novel_precision'))} / {_fmt(r.get('novel_recall'))} | {_fmt(r.get('extraction_recall'))} | {r.get('llm_calls')} | {_fmt(r.get('cost_usd'))} |")
        md += ["", f"Figure: `figures/ablation.png` (written by `ablation_figure`).", ""]
    gate = [r for r in rows if r.get("experiment") == "gate_sensitivity"]
    if gate:
        md += ["## Gate sensitivity", ""] + [f"- cheapest setting with recall >= {config.GATE_TARGET_RECALL}: `{r.get('best')}`" for r in gate] + ["", "Full curve: `gate_sensitivity/curve.csv`.", ""]
    rev = [r for r in rows if r.get("experiment") == "reverse_check_value"]
    if rev:
        md += ["## Reverse-check value", ""] + [f"- recovered {r.get('reverse_recovered')} of {r.get('reverse_candidates')} novel candidates with {r.get('reverse_calls')} calls (model cost in run: {_fmt(r.get('reverse_cost_usd'))} USD)" for r in rev] + [""]
    if errors is not None:
        counts = error_counts(errors)
        md += ["## Error analysis", "", "| category | count | top fix |", "|---|---|---|"] + [f"| {k} | {v} | {TOP_FIXES.get(k, '')} |" for k, v in counts.items()] + ["", "Every wrong pair: `errors.csv`.", ""]
    if changelog:
        md += ["## Changelog of targeted improvements", "", changelog, ""]
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return path


def ablation_figure(rows: Iterable[Mapping[str, object]], path: Path = config.EVAL_OUTPUT_DIR / "figures" / "ablation.png") -> Optional[Path]:
    """Bar chart of Acc_DC / Acc_DE per ablation step, with n and cost in the labels (matplotlib, Agg backend)."""
    rows = [r for r in rows if r.get("experiment") in EXPERIMENT_CONFIGS]
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = sorted(rows, key=lambda r: list(EXPERIMENT_CONFIGS).index(r["experiment"]))
    labels = [f"{PAPER_NAMES.get(r['experiment'], r['experiment'])}\n(n={r.get('n_documents')}, ${_fmt(r.get('cost_usd'))})" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (key, name, color) in enumerate((("acc_dc", "Acc_DC", "#4C72B0"), ("acc_de", "Acc_DE", "#DD8452"), ("f1", "F1 present", "#55A868"))):
        vals = [0.0 if (r.get(key) is None or (isinstance(r.get(key), float) and math.isnan(r.get(key)))) else float(r.get(key)) for r in rows]
        ax.bar(x + (i - 1) * 0.27, vals, width=0.27, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Ablation (paper Figure 3 equivalent)" + (f" - {rows[0].get('watermark')}" if rows[0].get("watermark") else ""))
    ax.legend(loc="lower right")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
