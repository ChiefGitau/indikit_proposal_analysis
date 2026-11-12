"""Metrics with hand-computed expected values, including edge cases."""
from __future__ import annotations

import math

import pytest

from pipeline.eval import metrics as m
from pipeline.eval.run_experiments import EXPERIMENTS, run_experiment, write_results_table


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("75 %", "75"),
        ("75%", "75"),
        ("75.0", "75"),
        (" 75 percent", "75"),
        ("0,75", "0.75"),
        ("0.75", "0.75"),
        ("24.397,00", "24397"),
        ("24,397", "24397"),
        ("24 397", "24397"),
        ("24 397", "24397"),
        ("1,234.5", "1234.5"),
        ("1.234.567", "1234567"),
        ("1,234,567.89", "1234567.89"),
        ("0.500", "0.5"),
        ("1.5", "1.5"),
        ("12 households", "12"),
        ("USD 1,000", "1000"),
        ("$ 2.5", "2.5"),
        ("-3", "-3"),
        ("", None),
        ("   ", None),
        ("N/A", None),
        ("tbd", None),
        (None, None),
        ("approx. 50", "approx. 50"),
        ("Not  Applicable", "not applicable"),
    ],
)
def test_normalise_value(raw, expected):
    assert m.normalise_value(raw) == expected


def test_values_equal():
    assert m.values_equal("75 %", "75%") and m.values_equal("24,397", "24.397,00") and m.values_equal(None, "")
    assert not m.values_equal("0,75", "75%")


def gold_set():
    return [
        m.PairLabel(doc_id="d1", indikit_id="1", present=True, baseline="75%", target="24.397,00", match_type="verbatim"),
        m.PairLabel(doc_id="d1", indikit_id="2", present=True, baseline=None, target="10", match_type="close"),
        m.PairLabel(doc_id="d1", indikit_id="3", present=False),
        m.PairLabel(doc_id="d2", indikit_id="1", present=False),
    ]


def predictions():
    return [
        m.PairPrediction(doc_id="d1", indikit_id="1", present=True, baseline="75 %", target="24,397", match_type="verbatim"),  # correct, values correct
        m.PairPrediction(doc_id="d1", indikit_id="3", present=True, match_type="close"),  # false positive
        m.PairPrediction(doc_id="d2", indikit_id="1", present=False),  # correct absent
        # d1/2 missing -> counts as predicted absent (false negative)
    ]


def test_acc_dc():
    assert m.acc_dc(gold_set(), predictions()) == pytest.approx(2 / 4)
    assert m.acc_dc(gold_set(), []) == pytest.approx(2 / 4)  # empty prediction: all absent -> only the absent pairs are right
    assert math.isnan(m.acc_dc([], predictions()))
    all_absent = [m.PairLabel(doc_id="d", indikit_id=str(i), present=False) for i in range(3)]
    assert m.acc_dc(all_absent, []) == 1.0


def test_precision_recall_present():
    r = m.precision_recall_present(gold_set(), predictions())
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == pytest.approx(0.5)
    # a present prediction for a pair that is not in the gold list is a false positive
    extra = predictions() + [m.PairPrediction(doc_id="d9", indikit_id="9", present=True, match_type="close")]
    assert m.precision_recall_present(gold_set(), extra)["fp"] == 2
    # no present pairs and no predictions -> zeros, not an exception
    all_absent = [m.PairLabel(doc_id="d", indikit_id="1", present=False)]
    assert m.precision_recall_present(all_absent, []) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    assert m.precision_recall_present(gold_set(), [])["recall"] == 0.0


def test_acc_de():
    assert m.acc_de(gold_set(), predictions()) == pytest.approx(1 / 2)  # pair 1 correct after normalisation, pair 2 missed
    wrong_target = [m.PairPrediction(doc_id="d1", indikit_id="1", present=True, baseline="75%", target="24398", match_type="verbatim")]
    assert m.acc_de(gold_set(), wrong_target) == 0.0
    both = predictions() + [m.PairPrediction(doc_id="d1", indikit_id="2", present=True, baseline="", target="10.0", match_type="verbatim")]
    assert m.acc_de(gold_set(), both) == 1.0
    assert math.isnan(m.acc_de([m.PairLabel(doc_id="d", indikit_id="1", present=False)], predictions()))
    assert m.acc_de(gold_set(), []) == 0.0


def test_match_type_accuracy():
    both = predictions() + [m.PairPrediction(doc_id="d1", indikit_id="2", present=True, match_type="verbatim")]  # gold says close
    assert m.match_type_accuracy(gold_set(), both) == pytest.approx(1 / 2)
    assert m.match_type_accuracy(gold_set(), predictions()) == 1.0  # only pair 1 found, label matches
    assert math.isnan(m.match_type_accuracy(gold_set(), []))


def test_reference_detection_accuracy():
    gold = [
        m.ReferenceLabel(doc_id="a", referenced_document="Annex C - Logical Framework"),
        m.ReferenceLabel(doc_id="b", referenced_document="Indicator tracker"),
        m.ReferenceLabel(doc_id="c", referenced_document=None),
    ]
    pred = [
        m.ReferencePrediction(doc_id="a", referenced_document="annex c logical framework"),
        m.ReferencePrediction(doc_id="b", referenced_document="IPTT"),
        m.ReferencePrediction(doc_id="c", referenced_document="something"),  # ignored: gold has no reference
    ]
    assert m.reference_detection_accuracy(gold, pred) == pytest.approx(1 / 2)
    assert m.reference_detection_accuracy(gold, []) == 0.0
    assert math.isnan(m.reference_detection_accuracy([gold[2]], pred))


def test_pairs_from_gold_rows():
    from pipeline.eval.gold import GoldRow

    rows = [
        GoldRow(doc_indicator_id="D1", indicator_text="x", level="output", indikit_id="5138", match_type="close", baseline="1", target="2"),
        GoldRow(doc_indicator_id="D2", indicator_text="y", level="output", match_type="none"),
    ]
    pairs = m.pairs_from_gold_rows("doc", rows)
    assert len(pairs) == 1 and pairs[0].key == ("doc", "5138") and pairs[0].match_type == "close" and pairs[0].target == "2"


def test_run_experiments_stub_and_results_writer(tmp_path):
    assert EXPERIMENTS == ("benchmark", "enhanced_rag", "plus_knowledge", "plus_search_term", "model_comparison", "gate_sensitivity", "reverse_check_value")
    with pytest.raises(ValueError):
        run_experiment("nope", runner_factory=None, gold_docs=[], metadata={}, doc_id_map={})
    path = write_results_table([{"experiment": "benchmark", "model": "m", "split": "test", "n_documents": 9, "acc_dc": 0.5781, "extra": "ignored"}], tmp_path / "results.csv")
    assert path.read_text().splitlines()[1].startswith("benchmark,m,test,9,,,0.578,")
    md = (tmp_path / "results.md").read_text()
    assert "| Benchmark (basic RAG) | test | 9 | 0.578 |" in md
