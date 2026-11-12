"""Evaluation metrics (paper Section 5.1.1; docs/plan.md Section 6.2).

All functions take plain lists of Pydantic records, never files. Undefined ratios (empty denominator)
return ``nan`` except precision / recall / F1, which follow the usual ``zero_division=0`` convention.

- ``acc_dc``: disclosure accuracy over all (document, IndiKit indicator) pairs (paper Eq. 3);
- ``precision_recall_present``: precision, recall, F1 of the "present" class;
- ``acc_de``: over truly present pairs, share with correct baseline **and** target after
  normalisation (paper Eq. 4);
- ``match_type_accuracy``: over correctly found pairs, share with the right verbatim / close label;
- ``reference_detection_accuracy``: over documents whose indicators sit in another document, share where
  the predicted referenced document name matches.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Iterable, Literal, Optional

from pydantic import BaseModel

Pair = tuple[str, str]


class PairLabel(BaseModel):
    """Gold label for one (document, IndiKit indicator) pair."""

    doc_id: str
    indikit_id: str
    present: bool
    baseline: Optional[str] = None
    target: Optional[str] = None
    match_type: Optional[Literal["verbatim", "close"]] = None

    @property
    def key(self) -> Pair:
        return (self.doc_id, self.indikit_id)


class PairPrediction(BaseModel):
    doc_id: str
    indikit_id: str
    present: bool
    baseline: Optional[str] = None
    target: Optional[str] = None
    match_type: Optional[Literal["verbatim", "close"]] = None

    @property
    def key(self) -> Pair:
        return (self.doc_id, self.indikit_id)


class ReferenceLabel(BaseModel):
    doc_id: str
    referenced_document: Optional[str] = None


class ReferencePrediction(BaseModel):
    doc_id: str
    referenced_document: Optional[str] = None


# --------------------------------------------------------------------------- #
# Value normalisation
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"^[+-]?[\d.,\s ']+$")


def normalise_value(value: Optional[str]) -> Optional[str]:
    """Canonical form of a baseline / target so that ``"75 %"``, ``"75%"``, ``"75.0"`` compare equal and
    ``"24.397,00"``, ``"24,397"``, ``"24 397"`` all become ``"24397"``.

    Rules: strip whitespace, percent signs and common unit words; if what remains is numeric, resolve the
    separators (last separator with exactly 3 digits after it and no other separator type is a thousands
    separator; otherwise the last separator is the decimal mark) and return the number without trailing
    zeros. Non-numeric values are lower-cased and whitespace-collapsed. Empty / N/A -> None.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("n/a", "na", "none", "null", "-", "tbd", "tbc"):
        return None
    s = s.replace(" ", " ")
    s = re.sub(r"\s*(%|percent|pct|per cent)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*(usd|eur|chf|gbp|\$|€|£)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*(usd|eur|chf|gbp|households?|hh|people|persons|individuals|units?)\s*$", "", s, flags=re.IGNORECASE)
    s = s.strip()
    if _NUM_RE.match(s):
        compact = re.sub(r"[\s']", "", s)
        sign = ""
        if compact and compact[0] in "+-":
            sign, compact = compact[0], compact[1:]
        seps = [c for c in compact if c in ",."]
        kinds = set(seps)
        if len(kinds) == 2:
            # both separators present: the last one is the decimal mark, the other the thousands separator
            dec = seps[-1]
            compact = compact.replace("," if dec == "." else ".", "").replace(dec, ".")
        elif len(kinds) == 1:
            sep = seps[0]
            parts = compact.split(sep)
            if len(parts) > 2:
                # repeated separator (1.234.567): thousands if every later group has 3 digits
                if all(len(g) == 3 for g in parts[1:]):
                    compact = "".join(parts)
                else:
                    return re.sub(r"\s+", " ", s).lower()
            else:
                before, after = parts
                if len(after) == 3 and 1 <= len(before) <= 3 and before != "0":
                    compact = before + after  # 24,397 / 24.397 -> thousands separator
                else:
                    compact = f"{before or '0'}.{after}"  # 0,75 / 0.500 / 1000.5 -> decimal mark
        try:
            d = Decimal(sign + compact)
        except InvalidOperation:
            return re.sub(r"\s+", " ", s).lower()
        d = d.normalize()
        text = format(d, "f")
        return text if text != "-0" else "0"
    return re.sub(r"\s+", " ", s).lower()


def values_equal(a: Optional[str], b: Optional[str]) -> bool:
    return normalise_value(a) == normalise_value(b)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _index(preds: Iterable[PairPrediction]) -> dict[Pair, PairPrediction]:
    return {p.key: p for p in preds}


def acc_dc(gold: list[PairLabel], predictions: list[PairPrediction]) -> float:
    """Share of gold pairs whose predicted present/absent equals the label. Pairs without a prediction count
    as predicted absent. ``nan`` when ``gold`` is empty."""
    if not gold:
        return math.nan
    by_key = _index(predictions)
    correct = 0
    for g in gold:
        p = by_key.get(g.key)
        predicted_present = p.present if p is not None else False
        correct += int(predicted_present == g.present)
    return correct / len(gold)


def precision_recall_present(gold: list[PairLabel], predictions: list[PairPrediction]) -> dict[str, float]:
    """Precision, recall and F1 of the "present" class over the gold pairs (zero_division=0)."""
    by_key = _index(predictions)
    tp = fp = fn = 0
    for g in gold:
        p = by_key.get(g.key)
        predicted_present = p.present if p is not None else False
        if predicted_present and g.present:
            tp += 1
        elif predicted_present and not g.present:
            fp += 1
        elif not predicted_present and g.present:
            fn += 1
    gold_keys = {g.key for g in gold}
    # predictions for pairs absent from the gold list are counted as false positives (gold = absent by definition)
    fp += sum(1 for p in predictions if p.present and p.key not in gold_keys)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def acc_de(gold: list[PairLabel], predictions: list[PairPrediction]) -> float:
    """Over truly present pairs: share predicted present with baseline AND target equal after normalisation.
    ``nan`` when no gold pair is present."""
    present = [g for g in gold if g.present]
    if not present:
        return math.nan
    by_key = _index(predictions)
    correct = 0
    for g in present:
        p = by_key.get(g.key)
        if p is None or not p.present:
            continue
        if values_equal(p.baseline, g.baseline) and values_equal(p.target, g.target):
            correct += 1
    return correct / len(present)


def match_type_accuracy(gold: list[PairLabel], predictions: list[PairPrediction]) -> float:
    """Over pairs correctly predicted present: share with the same verbatim / close label. ``nan`` if none."""
    by_key = _index(predictions)
    found = [(g, by_key[g.key]) for g in gold if g.present and g.key in by_key and by_key[g.key].present]
    if not found:
        return math.nan
    return sum(1 for g, p in found if g.match_type == p.match_type) / len(found)


def _norm_doc_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return s or None


def reference_detection_accuracy(gold: list[ReferenceLabel], predictions: list[ReferencePrediction]) -> float:
    """Over gold documents that reference another document: share where the prediction names the same document
    (case / punctuation insensitive). ``nan`` when no gold document has a reference."""
    with_ref = [g for g in gold if g.referenced_document]
    if not with_ref:
        return math.nan
    by_doc = {p.doc_id: p for p in predictions}
    correct = 0
    for g in with_ref:
        p = by_doc.get(g.doc_id)
        if p is not None and _norm_doc_name(p.referenced_document) == _norm_doc_name(g.referenced_document):
            correct += 1
    return correct / len(with_ref)


def pairs_from_gold_rows(doc_id: str, rows: Iterable) -> list[PairLabel]:
    """Convenience: present-pair labels from the rows of one gold file (rows with an ``indikit_id``)."""
    out: list[PairLabel] = []
    for r in rows:
        if getattr(r, "indikit_id", None):
            out.append(PairLabel(doc_id=doc_id, indikit_id=r.indikit_id, present=True, baseline=r.baseline, target=r.target, match_type=r.match_type))
    return out
