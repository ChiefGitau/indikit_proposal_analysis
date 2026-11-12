"""Shared text utilities for the parsers: whitespace, hyphenation, language heuristic, ids, limits."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Optional

_WS_RE = re.compile(r"[ \t   ]+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalise_whitespace(text: str, keep_newlines: bool = False) -> str:
    """Collapse runs of spaces; optionally keep single newlines. Accents and symbols stay intact."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    if keep_newlines:
        text = "\n".join(line.strip() for line in text.split("\n"))
        return _MULTI_NL_RE.sub("\n\n", text).strip()
    return re.sub(r"\s*\n\s*", " ", text).strip()


def fix_hyphenation(text: str) -> str:
    """Join words hyphenated at a line break ("house-\\nholds" -> "households") before whitespace collapsing."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text or "")


def clean_block_text(text: str) -> str:
    return normalise_whitespace(fix_hyphenation(text or ""))


def cap_length(text: str, limit: int, what: str, warnings: list[str], label: str = "") -> str:
    """Hard limit with a recorded warning instead of silent truncation."""
    if len(text) > limit:
        warnings.append(f"{what} {label} truncated from {len(text)} to {limit} characters".strip())
        return text[:limit]
    return text


# --------------------------------------------------------------------------- #
# Language heuristic (no dependency)
# --------------------------------------------------------------------------- #
_EN_STOP = {"the", "and", "of", "to", "in", "for", "with", "that", "will", "are", "is", "by", "on", "be", "this", "as", "from", "at", "or", "an", "which", "their", "has", "have", "not", "was", "were", "it", "its", "project", "through"}
_FR_STOP = {"le", "la", "les", "des", "du", "de", "et", "en", "pour", "dans", "une", "un", "que", "qui", "sur", "par", "au", "aux", "est", "sont", "avec", "ce", "cette", "ces", "pas", "plus", "leur", "leurs", "projet", "été", "sera", "ont", "nous", "afin"}


def detect_language(text: str, min_tokens: int = 20) -> str:
    """'en', 'fr' or 'mixed' from stop-word counts. Returns '' when there is too little text."""
    tokens = re.findall(r"[a-zà-ÿ']+", (text or "").lower())
    if len(tokens) < min_tokens:
        return ""
    en = sum(1 for t in tokens if t in _EN_STOP)
    fr = sum(1 for t in tokens if t in _FR_STOP)
    if en == 0 and fr == 0:
        return ""
    share_en = en / (en + fr)
    if share_en >= 0.8:
        return "en"
    if share_en <= 0.2:
        return "fr"
    return "mixed"


# --------------------------------------------------------------------------- #
# Ids and hashes
# --------------------------------------------------------------------------- #
def slugify(text: str, max_len: int = 80) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:max_len].rstrip("-") or "doc"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def looks_like_number(text: str) -> bool:
    return bool(re.fullmatch(r"[\s\-+]?[\d.,%\s]+", text or "")) and any(ch.isdigit() for ch in text)


def page_number_line(line: str) -> bool:
    s = (line or "").strip().lower()
    return bool(re.fullmatch(r"(page\s*)?\d{1,4}(\s*(/|of|de|sur)\s*\d{1,4})?", s)) or bool(re.fullmatch(r"[-–]\s*\d{1,4}\s*[-–]", s))


def heading_number_prefix(text: str) -> Optional[str]:
    """'3.2 Logical framework' -> '3.2'; 'Annex C' -> 'Annex C'; None otherwise."""
    m = re.match(r"^\s*((?:\d{1,2}\.){0,3}\d{1,2}\.?|[A-Z]\.|(?:annex|annexe|appendix|chapter|section|part)\s+[A-Z0-9]{1,3})\s*[\s:.\-–)]+\s*\S", text or "", re.IGNORECASE)
    return m.group(1).strip() if m else None
