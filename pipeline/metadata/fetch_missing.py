"""Fetch indicator pages that are missing from the local site data (docs/phase_1.md Task 3.6).

Reuses the parsing functions of ``zoa_indikit_app/src/scraper.py`` (same User-Agent, same 1-second
crawl delay). Results go to ``pipeline/metadata/site_supplement.json``; the raw ``scraped.json`` is never
modified. The build picks the supplement up automatically.

Run explicitly (never from tests)::

    python -m pipeline.metadata.fetch_missing            # the seven PIP indicators
    python -m pipeline.metadata.fetch_missing --url https://zoa.indikit.net/indicator/1234-...

Exit code 2 when nothing could be fetched (e.g. no network); the build then falls back to seeded records.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from pipeline import config


def load_scraper():
    """Import ``scraper`` from the IndiKit app without installing it."""
    src = str(config.SCRAPER_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    import scraper  # type: ignore

    return scraper


def fetch(urls: Iterable[str], out: Path = config.SITE_SUPPLEMENT_JSON, delay: Optional[float] = None, keep_existing: bool = True) -> dict:
    scraper = load_scraper()
    delay = scraper.CRAWL_DELAY if delay is None else delay
    out = Path(out)
    existing: dict = {}
    if keep_existing and out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8")).get("records", {})
        except json.JSONDecodeError:
            existing = {}
    records: dict = dict(existing)
    errors: list[dict] = []
    urls = list(urls)
    for i, url in enumerate(urls, start=1):
        rec = scraper.scrape_indicator(url)
        if rec.get("ok"):
            records[str(rec["unique_id"])] = rec
            print(f"[{i}/{len(urls)}] fetched {rec['unique_id']}: {rec.get('name', '')}")
        else:
            errors.append({"url": url, "error": rec.get("error", "unknown")})
            print(f"[{i}/{len(urls)}] FAILED {url}: {rec.get('error')}", file=sys.stderr)
        if i < len(urls):
            time.sleep(delay)
    result = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": scraper.BASE_URL,
        "user_agent": scraper.HEADERS.get("User-Agent"),
        "records": dict(sorted(records.items())),
        "errors": errors,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", action="append", dest="urls", help="indicator page URL (repeatable); default: the seven PIP indicators")
    parser.add_argument("--out", type=Path, default=config.SITE_SUPPLEMENT_JSON)
    parser.add_argument("--delay", type=float, default=None, help="seconds between requests (default: scraper's crawl delay)")
    parser.add_argument("--replace", action="store_true", help="discard previously fetched records in the output file")
    args = parser.parse_args(argv)
    urls = args.urls or list(config.PIP_SEED_URLS)
    try:
        result = fetch(urls, out=args.out, delay=args.delay, keep_existing=not args.replace)
    except Exception as exc:  # network down, import failure ...
        print(f"fetch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    n_ok = len(urls) - len(result["errors"])
    print(f"{n_ok}/{len(urls)} pages fetched; {len(result['records'])} records in {args.out}")
    return 0 if n_ok else 2


if __name__ == "__main__":
    sys.exit(main())
