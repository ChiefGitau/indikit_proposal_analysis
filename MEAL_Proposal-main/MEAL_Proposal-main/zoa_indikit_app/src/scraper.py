"""Scrape live indicator pages from https://zoa.indikit.net and compare them
to the local IndiKit CSV export.

The site is a plain server-rendered HTML app, so BeautifulSoup is enough — no
JS execution needed. Two entry points:

``fetch_sitemap_ids()``   → {unique_id: url} for every live ``/indicator/`` page.
``scrape_indicator(url)`` → dict of the fields we can read off one page.

Scraped results are cached to disk (``data/.../site_scrape/scraped.json``) so a
full crawl (≈300 pages at the site's 1 req/s crawl-delay ≈ 5 min) survives
Streamlit reruns and app restarts. The comparison helpers are pure functions of
the CSV DataFrame + the cached scrape, so they re-run cheaply on every render.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://zoa.indikit.net"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
# robots.txt asks for max one request per second.
CRAWL_DELAY = 1.0
HEADERS = {"User-Agent": "ZOA-IndiKit-Profiler/1.0 (internal MEAL data QA)"}

CACHE_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "indicators"
    / "data_analysis"
    / "site_scrape"
)
CACHE_FILE = CACHE_DIR / "scraped.json"

# Which page block feeds which comparison field. The site uses a typographic
# apostrophe (’) in the headings, so we match on lower-cased keywords instead.
_BLOCK_KEYWORDS = {
    "purpose": "purpose",
    "how_to_collect": "how to collect",
    "disaggregate": "disaggregate",
    "important_comments": "important comments",
}


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def fetch_sitemap_ids() -> dict[str, str]:
    """Return ``{unique_id: full_url}`` for every live indicator page.

    The sitemap is the authoritative list of what currently exists on the site,
    so it doubles as the "is this CSV row still published?" check.
    """
    resp = requests.get(SITEMAP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    urls = re.findall(r"<loc>(.*?)</loc>", resp.text)
    out: dict[str, str] = {}
    for url in urls:
        if "/indicator/" not in url:
            continue
        # A few sitemap entries are nested as /indicator/<sector>/<id-slug> and
        # 404; the canonical page is the flat /indicator/<id-slug>. Take the
        # last path segment (the id-slug) and rebuild the flat URL.
        slug = url.rstrip("/").rsplit("/", 1)[1]
        uid = slug.split("-", 1)[0]
        if uid.isdigit():
            out[uid] = f"{BASE_URL}/indicator/{slug}"
    return out


# Indicator-Level filter value ids are global across sectors (read off the
# sector pages' "Indicator Level" filter). Kept here as the canonical mapping;
# ``fetch_site_levels`` still parses each sector page so a new level added on the
# site is picked up automatically.
LEVEL_VALUES = {"228": "Output", "229": "Outcome", "230": "Impact", "237": "Organisational level"}


def fetch_sector_urls() -> list[str]:
    """Return every ``/sector/`` URL from the sitemap."""
    resp = requests.get(SITEMAP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return [
        u for u in re.findall(r"<loc>(.*?)</loc>", resp.text) if "/sector/" in u
    ]


def _indicator_ids_in(html: str) -> set[str]:
    """All indicator ids linked from a listing page."""
    return set(re.findall(r"/indicator/(\d+)", html))


def _sector_level_options(html: str) -> dict[str, str]:
    """Parse a sector page's Indicator-Level filter → ``{value_id: label}``."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one('.filter__itemWrap[data-id="38"]')
    opts: dict[str, str] = {}
    if body:
        for lab in body.select("label.filter__item"):
            inp = lab.select_one("input")
            span = lab.select_one("span")
            if inp and span and inp.get("value"):
                text = re.sub(r"\s*\(\d+\)\s*$", "", span.get_text()).strip()
                opts[inp.get("value")] = text
    return opts


def fetch_site_levels(*, delay: float = CRAWL_DELAY, progress=None) -> dict[str, list[str]]:
    """Return ``{indicator_id: [level labels]}`` derived from the live site.

    The indicator *detail* pages don't state a level, but each sector listing
    page can be filtered by level via ``?filterValues[]=<id>`` (server-side).
    So for every (sector, level) pair we collect which indicators appear and
    tag them with that level. Multi-level indicators (e.g. Output & Outcome)
    naturally show up under more than one filter. Result is cached to disk.
    """
    sectors = fetch_sector_urls()
    # (sector_url, level_value, level_label) work list.
    tasks: list[tuple[str, str, str]] = []
    for i, sec in enumerate(sectors):
        html = requests.get(sec, headers=HEADERS, timeout=30).text
        for val, label in _sector_level_options(html).items():
            tasks.append((sec, val, label))
        if progress:
            progress(i + 1, len(sectors), f"reading levels for {sec.rsplit('/', 1)[1]}")
        if i < len(sectors) - 1:
            time.sleep(delay)

    levels: dict[str, set[str]] = {}
    for j, (sec, val, label) in enumerate(tasks):
        resp = requests.get(f"{sec}?filterValues[]={val}", headers=HEADERS, timeout=30)
        for uid in _indicator_ids_in(resp.text):
            levels.setdefault(uid, set()).add(label)
        if progress:
            progress(j + 1, len(tasks), f"{sec.rsplit('/', 1)[1]} · {label}")
        if j < len(tasks) - 1:
            time.sleep(delay)

    result = {uid: sorted(labs) for uid, labs in levels.items()}
    cache = load_cache()
    cache["levels"] = result
    cache["levels_at"] = datetime.now().isoformat(timespec="seconds")
    save_cache(cache)
    return result


def _clean(text: str | None) -> str:
    """Collapse whitespace; return '' for None."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def scrape_indicator(url: str) -> dict:
    """Fetch and parse one indicator page into a flat dict of fields.

    Returns ``{"ok": False, "error": ...}`` on any HTTP/parse failure so a
    single bad page never aborts a full crawl.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return {"ok": False, "error": f"HTTP {resp.status_code}", "url": url}
        return _parse_indicator(resp.text, url)
    except requests.RequestException as exc:  # network/timeout
        return {"ok": False, "error": str(exc), "url": url}


def _parse_indicator(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # Slug id from the URL (stable, always present).
    slug = url.rstrip("/").rsplit("/", 1)[1]
    uid = slug.split("-", 1)[0]

    h1 = soup.find("h1")
    name = _clean(h1.get_text()) if h1 else ""

    section, sector, sub_sector = _parse_breadcrumbs(soup, name)
    wording_en, wording_fr = _parse_wording(soup)
    blocks = _parse_blocks(soup)

    return {
        "ok": True,
        "url": url,
        "unique_id": uid,
        "name": name,
        "section": section,
        "sector": sector,
        "sub_sector": sub_sector,
        "wording_en": wording_en,
        "wording_fr": wording_fr,
        "purpose": blocks.get("purpose", ""),
        "how_to_collect": blocks.get("how_to_collect", ""),
        "disaggregate": blocks.get("disaggregate", ""),
        "important_comments": blocks.get("important_comments", ""),
    }


def _parse_breadcrumbs(soup: BeautifulSoup, name: str) -> tuple[str, str, str]:
    """Breadcrumb trail is Home › Section › Sector › [Sub-Sector] › Name.

    We drop 'Home' and the trailing indicator name, then read the remainder
    positionally. Sub-sector is optional.
    """
    crumbs = [
        _clean(li.get_text())
        for li in soup.select(".breadcrumbs__item")
    ]
    crumbs = [c for c in crumbs if c and c.lower() != "home"]
    if crumbs and name and crumbs[-1].rstrip() == name.rstrip():
        crumbs = crumbs[:-1]
    section = crumbs[0] if len(crumbs) > 0 else ""
    sector = crumbs[1] if len(crumbs) > 1 else ""
    sub_sector = crumbs[2] if len(crumbs) > 2 else ""
    return section, sector, sub_sector


def _find_block(soup: BeautifulSoup, keyword: str):
    """Return the content div of the indicatorBlock whose title contains
    ``keyword`` (case-insensitive), or None."""
    for block in soup.select(".indicatorBlock"):
        title_el = block.select_one(".indicatorBlock__title")
        if title_el and keyword in _clean(title_el.get_text()).lower():
            return block.select_one(".indicatorBlock__content")
    return None


def _parse_wording(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract EN / FR wording from the language-tabbed Wording block.

    The block has ``.filter__button[data-id]`` (labelled English/French) and
    matching ``.filter__itemWrap[data-id]`` bodies. We map button label →
    data-id → body text. Falls back to the whole block if no tabs are present.
    """
    content = _find_block(soup, "wording")
    if content is None:
        return "", ""

    lang_by_id: dict[str, str] = {}
    for btn in content.select(".filter__button"):
        did = btn.get("data-id")
        label = _clean(btn.get_text()).lower()
        if did:
            lang_by_id[did] = label

    en = fr = ""
    items = content.select(".filter__itemWrap")
    if items and lang_by_id:
        for item in items:
            did = item.get("data-id")
            label = lang_by_id.get(did, "")
            text = _clean(item.get_text())
            if "french" in label or "fran" in label:
                fr = text
            elif "english" in label or "angl" in label:
                en = text
    else:
        # No language tabs — treat the whole block as the (English) wording.
        en = _clean(content.get_text())
    return en, fr


def _parse_blocks(soup: BeautifulSoup) -> dict[str, str]:
    out: dict[str, str] = {}
    for field, keyword in _BLOCK_KEYWORDS.items():
        content = _find_block(soup, keyword)
        out[field] = _clean(content.get_text()) if content else ""
    return out


# --------------------------------------------------------------------------- #
# Crawl driver + on-disk cache
# --------------------------------------------------------------------------- #
def load_cache() -> dict:
    """Return the cached scrape ``{"scraped_at": iso, "records": {uid: {...}}}``."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"scraped_at": None, "records": {}}


def save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False, default=str))


def crawl(id_url_map: dict[str, str], *, delay: float = CRAWL_DELAY, progress=None) -> dict:
    """Scrape every url in ``id_url_map``, honouring the crawl delay.

    ``progress`` is an optional ``callable(done, total, uid)`` for a UI hook.
    Results are merged into (and persisted to) the on-disk cache as we go, so an
    interrupted crawl keeps whatever it managed to fetch.
    """
    cache = load_cache()
    records = cache.get("records", {})
    total = len(id_url_map)
    for i, (uid, url) in enumerate(id_url_map.items(), start=1):
        records[uid] = scrape_indicator(url)
        if progress:
            progress(i, total, uid)
        if i < total:
            time.sleep(delay)
    cache["records"] = records
    cache["scraped_at"] = datetime.now().isoformat(timespec="seconds")
    save_cache(cache)
    return cache


def store_live_ids(ids) -> dict:
    """Persist the current sitemap indicator-id list into the cache.

    Keeps coverage ("removed from site") accurate across reruns/reloads without
    forcing a re-fetch of the sitemap every time the page renders.
    """
    cache = load_cache()
    cache["live_ids"] = sorted(str(i) for i in ids)
    save_cache(cache)
    return cache


# --------------------------------------------------------------------------- #
# Comparison (pure functions of CSV rows + scraped records)
# --------------------------------------------------------------------------- #
def similarity(a: str, b: str) -> float:
    """Ratio in [0, 1] of two normalised strings (1.0 == identical)."""
    a, b = _clean(a).lower(), _clean(b).lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# CSV column → (scraped record key, output similarity column).
_COMPARE_FIELDS = [
    ("Indicator Name", "name", "name_sim"),
    ("Indicator Wording - ENG", "wording_en", "wording_en_sim"),
    ("Indicator Wording - FRA", "wording_fr", "wording_fr_sim"),
    ("What is its Purpose?", "purpose", "purpose_sim"),
]


def build_comparison(csv_df, ok_records: dict):
    """Return one row per (CSV indicator ∩ scraped page) with per-field diffs.

    ``csv_df`` is any slice of the loaded IndiKit frame (e.g. the filtered one);
    ``ok_records`` is ``{unique_id: scraped_dict}`` for successfully scraped
    pages. Duplicate CSV ids keep their first row. Output columns: ``unique_id``,
    ``name``, the four ``*_sim`` ratios, and exact-match flags for the
    section/sector/sub-sector breadcrumb fields.
    """
    import pandas as pd  # local import keeps the module Streamlit-free to test

    rows = []
    seen = set()
    for _, csv_row in csv_df.iterrows():
        uid = str(csv_row["Unique ID"])
        if uid in seen or uid not in ok_records:
            continue
        seen.add(uid)
        scraped = ok_records[uid]

        rec = {"unique_id": uid, "name": scraped.get("name", "")}
        for csv_col, scr_key, sim_col in _COMPARE_FIELDS:
            rec[sim_col] = round(
                similarity(str(csv_row.get(csv_col, "")), scraped.get(scr_key, "")), 3
            )
        rec["section_match"] = _clean(str(csv_row.get("Section", ""))).lower() == _clean(
            scraped.get("section", "")
        ).lower()
        rec["sector_match"] = _clean(str(csv_row.get("Sector", ""))).lower() == _clean(
            scraped.get("sector", "")
        ).lower()
        rec["sub_sector_match"] = _clean(str(csv_row.get("Sub-Sector", ""))).lower() == _clean(
            scraped.get("sub_sector", "")
        ).lower()
        rows.append(rec)

    return pd.DataFrame(rows)


def _csv_level_set(value: str) -> set[str]:
    """CSV ``Indicator Level`` → a set of atomic levels ('' / '(missing)' → ∅)."""
    v = _clean(value)
    if not v or v.lower() == "(missing)":
        return set()
    return {p.strip() for p in v.split("|") if p.strip()}


def build_level_comparison(csv_df, site_levels: dict):
    """Compare CSV ``Indicator Level`` against the levels read off the live site.

    ``site_levels`` is ``{unique_id: [labels]}`` from :func:`fetch_site_levels`.
    One row per (deduplicated) CSV indicator, with a ``status`` that highlights
    gaps the site can fill:

    * ``site fills gap`` – CSV level missing, site has one → actionable.
    * ``still unknown``  – missing in both.
    * ``match`` / ``differ`` – both present and (dis)agree.
    * ``not on site``    – CSV has a level but the indicator isn't live.
    """
    import pandas as pd

    rows = []
    seen = set()
    for _, csv_row in csv_df.iterrows():
        uid = str(csv_row["Unique ID"])
        if uid in seen:
            continue
        seen.add(uid)

        csv_set = _csv_level_set(str(csv_row.get("Indicator Level", "")))
        site_list = site_levels.get(uid)
        site_set = set(site_list) if site_list else set()

        if site_list is None:
            status = "not on site" if csv_set else "still unknown"
        elif not csv_set and site_set:
            status = "site fills gap"
        elif not csv_set and not site_set:
            status = "still unknown"
        elif not site_set:
            status = "not on site"
        elif csv_set == site_set:
            status = "match"
        else:
            status = "differ"

        rows.append(
            {
                "unique_id": uid,
                "name": _clean(str(csv_row.get("Indicator Name", ""))),
                "sector": _clean(str(csv_row.get("Sector", ""))),
                "csv_level": " | ".join(sorted(csv_set)) or "(missing)",
                "site_level": " | ".join(sorted(site_set)) or "—",
                "status": status,
            }
        )
    return pd.DataFrame(rows)
