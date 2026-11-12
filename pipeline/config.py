"""Central configuration: paths, model names, prompt versions, raw-data locations.

Every path is resolved from this file's location so the package works from any
working directory. Nothing here reads ``.env``; API keys are read by the LLM
client at call time (``pipeline.agent.answer``).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
PIPELINE_DIR: Path = REPO_ROOT / "pipeline"
DOCS_DIR: Path = REPO_ROOT / "docs"
TESTS_DIR: Path = REPO_ROOT / "tests"
PROPOSALS_DIR: Path = REPO_ROOT / "proposals"

OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
API_CACHE_DIR: Path = OUTPUTS_DIR / "api_cache"          # legacy extraction cache (3,409 responses), keep
CACHE_DIR: Path = OUTPUTS_DIR / "cache"                  # new LLM caches (Phase 1: metadata enrichment)
METADATA_CACHE_DIR: Path = CACHE_DIR / "metadata"
KB_DIR: Path = OUTPUTS_DIR / "kb"                        # per-document knowledge bases (Phase 2)
METADATA_REPORT_DIR: Path = OUTPUTS_DIR / "metadata"     # build_report.md, merge_conflicts.csv
EVAL_OUTPUT_DIR: Path = OUTPUTS_DIR / "eval"             # experiment results (Phase 3)

# --------------------------------------------------------------------------- #
# Raw IndiKit data (read-only; see docs/phase_1.md Section 1)
# --------------------------------------------------------------------------- #
INDIKIT_ROOT: Path = REPO_ROOT / "MEAL_Proposal-main" / "MEAL_Proposal-main"
EXPORT_CSV: Path = INDIKIT_ROOT / "data" / "indicators" / "Export indikit.csv"            # latin-1 / cp1252
EXPORT_ENCODING: str = "latin-1"
SCRAPED_JSON: Path = INDIKIT_ROOT / "data" / "indicators" / "data_analysis" / "site_scrape" / "scraped.json"
SITE_CSV: Path = INDIKIT_ROOT / "notebooks" / "data" / "zoa_indikit_net" / "indicators.csv"  # utf-8
SCRAPER_SRC_DIR: Path = INDIKIT_ROOT / "zoa_indikit_app" / "src"                           # scraper.py lives here

# --------------------------------------------------------------------------- #
# Metadata module outputs (Phase 1)
# --------------------------------------------------------------------------- #
METADATA_DIR: Path = PIPELINE_DIR / "metadata"
METADATA_JSONL: Path = METADATA_DIR / "indikit_metadata.jsonl"
OVERRIDES_YAML: Path = METADATA_DIR / "overrides.yaml"
SITE_SUPPLEMENT_JSON: Path = METADATA_DIR / "site_supplement.json"   # written by fetch_missing.py
REFERENCE_LIBRARY_CSV: Path = REPO_ROOT / "reference_library.csv"    # legacy format used by the notebook
BUILD_REPORT_MD: Path = METADATA_REPORT_DIR / "build_report.md"
MERGE_CONFLICTS_CSV: Path = METADATA_REPORT_DIR / "merge_conflicts.csv"

# Seven PIP indicators that exist only on the live site (July 2026), see docs/phase_1.md Task 3.
PIP_SEED_URLS: tuple[str, ...] = (
    "https://zoa.indikit.net/indicator/5889-pip-of-households-who-are-implementing-their-pip-plans",
    "https://zoa.indikit.net/indicator/5890-pip-of-people-undertaking-actions-to-manage-and-protect-natural-resources",
    "https://zoa.indikit.net/indicator/5891-pip-of-farmer-households-who-are-trained-through-peer-to-peer-learning",
    "https://zoa.indikit.net/indicator/5892-pip-of-households-with-increased-income",
    "https://zoa.indikit.net/indicator/5893-pip-of-farm-plans-pip-developed-jointly-by-the-household",
    "https://zoa.indikit.net/indicator/5894-pip-of-community-members-participating-in-collective-activities-in-the-community-as-defined-in-the-village-plan",
    "https://zoa.indikit.net/indicator/5895-pip-farm-productivity",
)
PIP_SEED_SECTOR: str = "Food Security and Livelihoods"
PIP_SEED_SECTION: str = "Sectoral"

# Canonical IndiKit result levels, in display order.
LEVELS: tuple[str, ...] = ("Output", "Outcome", "Impact", "Organisational level")

# --------------------------------------------------------------------------- #
# Evaluation (Phase 1 scaffold)
# --------------------------------------------------------------------------- #
GOLD_DIR: Path = PIPELINE_DIR / "eval" / "gold"
GOLD_INDEX_CSV: Path = GOLD_DIR / "index.csv"

# --------------------------------------------------------------------------- #
# Models and prompt versions
# --------------------------------------------------------------------------- #
# Provider for the legacy extraction client and the metadata enrichment call.
# "openai" | "azure" | "ollama" (see pipeline.agent.answer.configure_llm).
DEFAULT_LLM_PROVIDER: str = "ollama"
DEFAULT_MODEL_NAME: str = "llama3.2"
OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
LEGACY_CACHE_MODEL: str = "gpt-4o-mini"   # model that produced the pre-existing outputs/api_cache entries

# Embedding model used by the legacy matcher. Phase 2 switches to a multilingual model.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Prompt versions: bump when a prompt's wording changes so caches are not reused.
PROMPT_VERSION: str = "disclosure-v1"          # five-part disclosure prompt (pipeline.agent.prompts)
EXTRACTION_PROMPT_VERSION: str = "v1"          # legacy per-chunk extraction prompt (novel pass)
LINK_PROMPT_VERSION: str = "link-v1"           # legacy Present/Different/Missing linking prompt
METADATA_ENRICH_PROMPT_VERSION: str = "enrich-v1"  # optional LLM enrichment of Knowledge/SearchTerm
