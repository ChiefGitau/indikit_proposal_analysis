"""extraction_utils is a thin shim over the pipeline package."""
from __future__ import annotations

import pytest


def test_package_imports_and_cli_lists_subcommands():
    import pipeline
    import pipeline.config as cfg
    from pipeline.run import build_parser

    assert cfg.REPO_ROOT.joinpath("pipeline").is_dir()
    help_text = build_parser().format_help()
    for name in ("build-metadata", "validate-gold", "build-kb", "build-baseline-kb", "score-pairs", "retrieve", "kb-report", "select-pairs", "estimate", "run", "answer", "novel-pass", "consolidate", "report", "experiments", "export-review", "import-review"):
        assert name in help_text
    assert pipeline.__doc__


def test_phase3_entry_points_exist_and_cli_refuses_live_calls_without_provider(monkeypatch, tmp_path):
    from pipeline import config
    from pipeline.agent import answer, novel_pass
    from pipeline.run import main

    assert callable(answer.answer_pair) and answer.answer_indicator.__doc__ and callable(novel_pass.run_novel_pass)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "")
    with pytest.raises(SystemExit, match="LLM access is not configured"):
        main(["run", "--pilot"])
    with pytest.raises(SystemExit, match="LLM access is not configured"):
        main(["experiments", "--allow-unverified", "--name", "benchmark"])


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




def test_load_reference_library_no_dummy_fallback(tmp_path):
    import extraction_utils as eu

    with pytest.raises(FileNotFoundError, match="python -m pipeline.run build-metadata"):
        eu.load_reference_library(str(tmp_path / "missing.csv"))
    for name in ("match_indicators_to_library", "classify_match_status", "_best_lexical_match", "THRESHOLDS"):
        with pytest.raises(AttributeError):
            getattr(eu, name)
