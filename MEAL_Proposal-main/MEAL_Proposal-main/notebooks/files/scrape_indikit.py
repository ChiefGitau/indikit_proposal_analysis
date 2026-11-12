"""
IndiKit scraper — works for any IndiKit-based domain:
  https://tdh.indikit.net, https://zoa.indikit.net, https://www.indikit.net

Usage:
  pip install requests beautifulsoup4 lxml pandas
  python scrape_indikit.py https://tdh.indikit.net
  python scrape_indikit.py https://zoa.indikit.net

Output:
  data/{domain}/indicators.json   (full structured data)
  data/{domain}/indicators.csv    (flat table for comparison)
"""

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (research/indicator-comparison; contact: you@example.org)"}
DELAY = 1.0  # be polite: 1 request per second

try:
    import lxml  # noqa: F401
    PARSER = "lxml"
except ImportError:
    PARSER = "html.parser"

# Section headers vary per IndiKit domain (e.g. tdh: "Purpose", "Definition";
# zoa: "Indicator's Wording", "Important Comments"), so capture any h2/h3
# section and only skip site chrome.
SKIP_SECTIONS = {"related indicators", "share", "search"}
MAX_SECTION_TITLE_LEN = 80


def get(session, url):
    time.sleep(DELAY)
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def discover_sectors(session, base_url):
    """Find all /sector/ links on the homepage."""
    soup = BeautifulSoup(get(session, base_url).text, PARSER)
    sectors = {}
    for a in soup.select('a[href*="/sector/"]'):
        href = urljoin(base_url, a["href"])
        name = a.get_text(strip=True)
        if name:
            sectors[href] = name
    return sectors


def discover_indicators_from_sitemap(session, base_url):
    """Try sitemap.xml first — most reliable way to get ALL indicator URLs."""
    for path in ("/sitemap.xml", "/sitemap"):
        try:
            r = get(session, base_url.rstrip("/") + path)
            urls = re.findall(r"<loc>(.*?)</loc>", r.text)
            ind = [u for u in urls if "/indicator/" in u]
            if ind:
                print(f"  sitemap found: {len(ind)} indicator URLs")
                return ind
        except requests.RequestException:
            continue
    return []


def discover_indicators_from_sectors(session, base_url, sectors):
    """Fallback: crawl sector pages. Handles 'See More' pagination by trying ?page=N."""
    found = {}
    for sector_url, sector_name in sectors.items():
        page = 1
        seen_this_sector = set()
        while True:
            url = sector_url if page == 1 else f"{sector_url}?page={page}"
            try:
                soup = BeautifulSoup(get(session, url).text, PARSER)
            except requests.RequestException:
                break
            links = {
                urljoin(base_url, a["href"])
                for a in soup.select('a[href*="/indicator/"]')
            }
            new = links - seen_this_sector
            if not new:
                break
            seen_this_sector |= new
            for u in new:
                found.setdefault(u, sector_name)
            page += 1
            if page > 50:  # safety
                break
        print(f"  {sector_name}: {len(seen_this_sector)} indicators")
    return found


def parse_indicator(session, url, base_url, sector_hint=None):
    soup = BeautifulSoup(get(session, url).text, PARSER)
    data = {
        "url": url,
        "domain": urlparse(base_url).netloc,
        "slug": url.rstrip("/").split("/")[-1],
    }

    h1 = soup.find("h1")
    data["title"] = h1.get_text(strip=True) if h1 else None

    # Breadcrumb → sector
    crumbs = [a.get_text(strip=True) for a in soup.select('a[href*="/sector/"]')]
    # last breadcrumb-like sector link before the h1 is usually the sector;
    # fall back to hint from sector crawl
    data["sector"] = sector_hint or (crumbs[-1] if crumbs else None)

    # Key/value tables (Indicator ID, full statement, Importance, Data sources, ...)
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) == 2:
                key = cells[0].get_text(" ", strip=True)
                val = cells[1].get_text("\n", strip=True)
                if key:
                    data[key] = val

    # Narrative sections under h2/h3 headers
    for header in soup.find_all(["h2", "h3"]):
        title = header.get_text(strip=True)
        if (
            not title
            or len(title) > MAX_SECTION_TITLE_LEN
            or title.lower() in SKIP_SECTIONS
            or title == data["title"]
        ):
            continue
        # Content is either in siblings of the header (tdh template) or in
        # siblings of the header's wrapper div (zoa template:
        # indicatorBlock__header + indicatorBlock__content).
        start = header
        if not header.find_next_siblings():
            start = header.parent
        parts = []
        for sib in start.find_next_siblings():
            if sib.name in ("h2", "h3") or sib.find(["h2", "h3"]) is not None:
                break
            if sib.name == "table":
                continue  # already captured above
            text = sib.get_text("\n", strip=True)
            if text:
                parts.append(text)
        if parts:
            data[f"section:{title}"] = "\n\n".join(parts)

    return data


def main(base_url):
    domain = urlparse(base_url).netloc.replace(".", "_")
    out_dir = Path("data") / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    print(f"[1/3] Discovering sectors on {base_url}")
    sectors = discover_sectors(session, base_url)
    print(f"  {len(sectors)} sectors found")

    print("[2/3] Discovering indicators")
    indicator_urls = {u: None for u in discover_indicators_from_sitemap(session, base_url)}
    if not indicator_urls:
        indicator_urls = discover_indicators_from_sectors(session, base_url, sectors)

    print(f"[3/3] Scraping {len(indicator_urls)} indicator pages")
    records = []
    for i, (url, sector_hint) in enumerate(sorted(indicator_urls.items()), 1):
        try:
            records.append(parse_indicator(session, url, base_url, sector_hint))
            print(f"  [{i}/{len(indicator_urls)}] {url}")
        except Exception as e:
            print(f"  FAILED {url}: {e}")

    (out_dir / "indicators.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(records).to_csv(out_dir / "indicators.csv", index=False)
    print(f"\nSaved {len(records)} indicators to {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1].rstrip("/"))
