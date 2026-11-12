"""extraction_utils is a thin shim over the pipeline package."""
from __future__ import annotations

import warnings

import pytest


def test_package_imports_and_cli_lists_subcommands():
    import pipeline
    import pipeline.config as cfg
    from pipeline.run import build_parser

    assert cfg.REPO_ROOT.joinpath("pipeline").is_dir()
    help_text = build_parser().format_help()
    for name in ("build-metadata", "validate-gold", "build-kb", "query", "run-experiments"):
        assert name in help_text
    assert pipeline.__doc__


def test_phase_stubs_raise_not_implemented():
    from pipeline.agent import retrieve
    from pipeline.preprocess import knowledge_base, outline, parse_docx, parse_pdf, parse_xlsx, references
    from pipeline.run import main

    for fn, arg in ((parse_pdf.parse_pdf, "x"), (parse_docx.parse_docx, "x"), (parse_xlsx.parse_xlsx, "x"), (outline.build_outline, []), (references.detect_references, []), (knowledge_base.build_knowledge_base, "x")):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            fn(arg) if fn is not knowledge_base.build_knowledge_base else fn(arg, "y")
    with pytest.raises(NotImplementedError, match="Phase 3"):
        retrieve.build_query(None)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        main(["build-kb"])


def test_shim_reexports_and_forwards_settings():
    import extraction_utils as eu
    from pipeline.agent import answer

    assert eu.LLMClient is __import__("pipeline.agent.novel_pass", fromlist=["LLMClient"]).LLMClient
    assert eu.Indicator is __import__("pipeline.models", fromlist=["Indicator"]).Indicator
    assert eu.CACHE_DIR == answer.CACHE_DIR
    old = answer.FORCE_REFRESH
    try:
        eu.FORCE_REFRESH = True
        assert answer.FORCE_REFRESH is True and eu.FORCE_REFRESH is True
    finally:
        eu.FORCE_REFRESH = old
    assert len([n for n in dir(eu) if n == "LLMClient"]) == 1
    with pytest.raises(AttributeError):
        eu.does_not_exist


def test_deprecated_matching_v1_warns_but_works():
    import extraction_utils as eu

    with pytest.warns(DeprecationWarning):
        t = eu.THRESHOLDS
    with pytest.warns(DeprecationWarning):
        assert eu.classify_match_status(0.96, 92, t) == "exact"
    with pytest.warns(DeprecationWarning):
        assert eu._best_lexical_match("a b", ["a b", "c"])[0] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eu.normalize_text_for_dedupe("x")  # non-deprecated helpers stay silent


def test_load_reference_library_no_dummy_fallback(tmp_path):
    import extraction_utils as eu

    with pytest.raises(FileNotFoundError, match="python -m pipeline.run build-metadata"):
        eu.load_reference_library(str(tmp_path / "missing.csv"))
    with pytest.warns(DeprecationWarning):
        with pytest.raises(FileNotFoundError):
            eu.match_indicators_to_library([eu.Indicator(indicator_text="x")], str(tmp_path / "missing.csv"), embedding_model=None)
