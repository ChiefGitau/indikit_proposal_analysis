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

    for name, phase, what in (
        ("build-kb", "Phase 2", "document parsing and knowledge-base construction"),
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
