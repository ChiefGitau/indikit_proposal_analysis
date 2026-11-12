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
PROMPT_VERSION: str = "disclosure-v2"          # five-part disclosure prompt (pipeline.agent.prompts); v2 = multi-match schema (Phase 3)
EXTRACTION_PROMPT_VERSION: str = "v1"          # legacy per-chunk extraction prompt (novel pass)
LINK_PROMPT_VERSION: str = "link-v1"           # legacy Present/Different/Missing linking prompt
METADATA_ENRICH_PROMPT_VERSION: str = "enrich-v1"  # optional LLM enrichment of Knowledge/SearchTerm

# --------------------------------------------------------------------------- #
# Phase 2 — preprocessing, knowledge bases, retrieval
# --------------------------------------------------------------------------- #
PIPELINE_VERSION: str = "kb-v1"   # bump when parsing / chunking / KB layout changes (forces a rebuild)

# Document sources. ``proposals/`` is the corpus; the template folder of the IndiKit app is used only for the
# integration sanity checks (docs/phase_2.md Task 8) and is registered under the "template" region.
DOCUMENT_ROOTS: tuple[tuple[str, Path], ...] = (
    ("proposals", PROPOSALS_DIR),
    ("template", INDIKIT_ROOT / "data" / "proposals" / "template"),
)
DOCUMENT_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".xlsx")

KB_BASELINE_DIR: Path = OUTPUTS_DIR / "kb_baseline"
MANIFEST_CSV: Path = KB_DIR / "manifest.csv"
PAIR_SCORES_PATH: Path = KB_DIR / "pair_scores.parquet"      # falls back to pair_scores.csv without pyarrow
QUERIES_DIR: Path = KB_DIR / "_queries"
EMBEDDING_CACHE_DB: Path = KB_DIR / "_cache" / "embeddings.sqlite"
PREPROCESS_REPORT_MD: Path = KB_DIR / "preprocess_report.md"

# Embedding models. Names are "<backend>:<model>" or a bare model name (backend chosen automatically:
# fastembed first - ONNX, no PyTorch, works on Intel Macs - then sentence-transformers). "hashing" is the
# deterministic dependency-free embedder used by the tests.
KB_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # documents are EN + FR
BASELINE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"                  # paper's plain-RAG benchmark
TEST_EMBEDDING_MODEL: str = "hashing"

# Re-rankers. Default: multilingual cross-encoder available through fastembed (ONNX). The brief's first choice
# BAAI/bge-reranker-v2-m3 needs PyTorch via sentence-transformers and is selectable as "st:BAAI/bge-reranker-v2-m3".
RERANKER_MODEL: str = "jinaai/jina-reranker-v2-base-multilingual"
RERANKER_MODEL_FAST: str = "Xenova/ms-marco-MiniLM-L-6-v2"   # small English-only alternative for quick runs
TEST_RERANKER_MODEL: str = "lexical"

# Parsing limits (warn, never silently truncate)
MAX_CELL_CHARS: int = 4000
MAX_BLOCK_CHARS: int = 20000
MAX_TABLE_ROWS: int = 2000

# Text KB chunking: section-aware, target words per chunk (never crossing a top-level heading)
CHUNK_TARGET_WORDS: int = 250
CHUNK_MAX_WORDS: int = 350
# Table KB: tables with at most this many rows are expanded whole on a keyword/summary hit
SMALL_TABLE_ROWS: int = 12

# Baseline (plain RAG): fixed-size character chunks
BASELINE_CHUNK_CHARS: int = 1000
BASELINE_CHUNK_OVERLAP: int = 200

# Retrieval defaults (paper 3.4.1: top-N per KB, re-rank to top-k)
RETRIEVAL_TOP_N: int = 5
RETRIEVAL_TOP_K: int = 4
LOGFRAME_BOOST: float = 0.05      # added to the vector score of logframe evidence before re-ranking

# --------------------------------------------------------------------------- #
# Phase 3 — LLM agent, pair selection, novel pass, runs, evaluation, review
# --------------------------------------------------------------------------- #
# Provider and model roles. The repository owner sets these (or the environment variables LLM_PROVIDER,
# ANSWER_MODEL, STRONG_MODEL, EXTRACTION_MODEL, which take precedence). Providers: "openai" | "azure" | "ollama"
# (local, OpenAI-compatible) | "mock" (tests only). An empty provider means "not configured": the CLI then
# refuses every live call (docs/phase_3.md Section 2 and 3). Keys are read from the environment / .env at call
# time only (pipeline.agent.llm); nothing here reads .env.
LLM_PROVIDER: str = ""                                   # owner decision: which service may process ZOA documents
ANSWER_MODEL: str = DEFAULT_MODEL_NAME                   # disclosure answering (default: the repository's existing default model)
STRONG_MODEL: str = ""                                   # model comparison (owner fills in, e.g. "gpt-4o")
EXTRACTION_MODEL: str = DEFAULT_MODEL_NAME               # novel-indicator pass (text sections, unmapped tables)
AZURE_OPENAI_API_VERSION_DEFAULT: str = "2024-10-21"

# Budget (enforced in code by pipeline.agent.llm.UsageTracker). Conservative defaults; the owner raises them.
MAX_CALLS_PER_RUN: int = 1500
MAX_COST_USD_PER_RUN: float = 5.0
# USD per 1M tokens (prompt, completion). Unknown models are priced at the most expensive entry (fail safe).
MODEL_PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    "llama3.2": (0.0, 0.0),          # local
}
UNKNOWN_MODEL_PRICE_USD_PER_M: tuple[float, float] = (2.50, 10.00)
# Averages used for the pre-run estimate (pairs x average tokens); updated from the pilot in the run report.
EST_PROMPT_TOKENS_PER_PAIR: int = 1800
EST_COMPLETION_TOKENS_PER_PAIR: int = 220
EST_PROMPT_TOKENS_PER_EXTRACTION: int = 1500
EST_COMPLETION_TOKENS_PER_EXTRACTION: int = 500

LLM_MAX_WORKERS: int = 4                # bounded concurrency
LLM_RATE_LIMIT_PER_MIN: int = 120       # calls per minute across workers
LLM_MAX_RETRIES: int = 3                # transport retries with exponential backoff
LLM_TIMEOUT_S: float = 120.0
LLM_MAX_COMPLETION_TOKENS: int = 1500

AGENT_CACHE_DIR: Path = CACHE_DIR / "agent"              # one JSON per (doc sha, indicator, evidence fingerprint, prompt version, model)
RUNS_DIR: Path = OUTPUTS_DIR / "runs"
LATEST_DIR: Path = OUTPUTS_DIR / "latest"
PILOT_SIZE: int = 5

# Prompt / rubric versions (bump when wording changes; part of every cache key)
RUBRIC_VERSION: str = "rubric-v1"
RUBRIC_MD: Path = DOCS_DIR / "rubric.md"
NOVEL_PROMPT_VERSION: str = "extract-v2"               # LLM extraction for logframe text sections / unmapped tables
REVERSE_CHECK_PROMPT_VERSION: str = "reverse-v1"

# Pair selection (docs/phase_3.md Task 4). Defaults apply when no verified gold exists.
GATE_TOP_M: int = 60                    # always keep each document's top-M indicators by score
GATE_TAU: float | None = None           # absolute threshold; None = the GATE_TAU_PERCENTILE-th percentile of all pair scores
GATE_TAU_PERCENTILE: float = 90.0
GATE_TARGET_RECALL: float = 0.95        # tuning target on the dev split
ROUTE_SECTORS: bool = False             # sector routing off until evaluated
SECTOR_ROUTE_TOP_N: int = 1             # a sector is "covered" when the mean of its top-N pair scores >= the sector threshold
SECTOR_ROUTE_PERCENTILE: float = 75.0   # sector threshold = this percentile of all pair scores (independent of tau)
INCLUDE_EXPORT_ONLY: bool = True        # ask export-only indicators (on_site = false) too; marked in the outputs

# Post-validation of answers (guards against invented indicators)
QUOTE_MATCH_THRESHOLD: int = 88         # rapidfuzz partial_ratio on normalised text (0-100)
VERBATIM_SIMILARITY_WARN: int = 60      # verbatim matches below this token-set similarity to the IndiKit wording get needs_review
LINK_TEXT_SIMILARITY: int = 80          # linking disclosure matches to extracted document indicators

# Reverse check (Task 5C) and novel clustering (Task 6)
REVERSE_CHECK_TOP: int = 10
NOVEL_CLUSTER_THRESHOLD: float = 0.80   # cosine similarity for greedy clustering of novel indicators

# Review workflow (Task 9)
REVIEW_DECISIONS: tuple[str, ...] = ("accept", "change_to_verbatim", "change_to_close", "reject", "other_id")
