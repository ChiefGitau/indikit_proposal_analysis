"""
extraction_utils.py - compatibility shim.

The reusable parts of the original module moved into the ``pipeline`` package in Phase 1
(see docs/plan.md Section 5 and docs/phase_1.md Task 2):

- ``pipeline.models``            DocumentChunk, Indicator, level/confidence normalisers
- ``pipeline.agent.answer``      provider config, cache helpers, _call_openai_with_retries,
                                 _extract_json_from_response, CacheMiss
- ``pipeline.agent.novel_pass``  chunk scoring/filtering, deduplication, LLMClient

Everything is re-exported here so the notebook and the existing tests keep working.
Runtime settings that used to be module globals here (``LLM_PROVIDER``, ``MODEL_NAME``,
``CACHE_DIR``, ``FORCE_REFRESH`` ...) now live in ``pipeline.agent.answer``; reading or
assigning them through this module is forwarded there, so ``extraction_utils.FORCE_REFRESH = True``
and ``monkeypatch.setattr(extraction_utils, "CACHE_DIR", ...)`` still take effect.

What stays here (legacy, replaced in later phases):
- the simple docx/PDF parsers (Phase 2 replaces them with layout-aware parsers);
- the IndiKit CSV -> reference_library.csv converter and ``load_reference_library``;
- matching v1 (``match_indicators_to_library`` and friends) - **deprecated**, replaced by the agent;
- the embedding shortlist + LLM linking v2 (``shortlist_reference_candidates``,
  ``link_indicator_to_reference``) - retired by the Phase 3 agent, kept for the notebook.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import types
import warnings
from typing import Dict, List, Optional, Tuple

from pipeline import config as _config
from pipeline.agent import answer as _answer
from pipeline.agent.answer import (  # noqa: F401  re-exported functions (settings are forwarded, see bottom)
    CacheMiss,
    SUPPORTED_PROVIDERS,
    _chunk_hash,
    _extract_json_from_response,
    _make_client,
    cache_response_for_chunk,
    configure_llm,
    llm_available,
    load_cached_response,
)
from pipeline.agent.novel_pass import (  # noqa: F401
    DISCARD_KEYWORDS,
    INDICATOR_KEYWORDS,
    TABLE_HEADER_PATTERNS,
    LLMClient,
    _build_extraction_prompt,
    chunk_discard_reason,
    deduplicate_indicators,
    filter_indicator_chunks,
    is_table_like_text,
    normalize_text_for_dedupe,
    score_chunk_for_indicators,
)
from pipeline.models import (  # noqa: F401
    CONFIDENCE_LEVELS,
    INDICATOR_LEVELS,
    DocumentChunk,
    Indicator,
    normalize_confidence,
    normalize_indicator_level,
)

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - import guard for environments missing rapidfuzz
    fuzz = None


# Names whose reads and writes are forwarded to pipeline.agent.answer (single source of truth).
_FORWARDED = frozenset(
    {
        "LLM_PROVIDER",
        "MODEL_NAME",
        "OLLAMA_BASE_URL",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "LEGACY_CACHE_MODEL",
        "FORCE_REFRESH",
        "CACHE_DIR",
        "_call_openai_with_retries",
        "_call_llm_with_retries",
    }
)

# Deprecated matching-v1 thresholds. Accessing ``THRESHOLDS`` warns (see module __getattr__).
_THRESHOLDS = {
    "exact_embedding": 0.95,
    "exact_lexical": 90,
    "close_embedding": 0.80,
    "related_embedding": 0.65,
}

_DEPRECATION_MSG = (
    "{name} belongs to the retired matching v1 (fuzzy + embedding thresholds). "
    "It is replaced by the Phase 3 LLM agent (pipeline.agent); see docs/plan.md Section 5."
)


def _deprecated(name: str) -> None:
    warnings.warn(_DEPRECATION_MSG.format(name=name), DeprecationWarning, stacklevel=3)


# ------------------ Deprecated matching v1 ------------------

def classify_match_status(embedding_score: float, lexical_score: float, thresholds: Optional[Dict] = None) -> str:
    """Deprecated. Classify a combined match status per the v1 thresholds.

    lexical_score is 0-100; embedding_score is 0-1
    """
    _deprecated("classify_match_status")
    thresholds = thresholds or _THRESHOLDS
    if embedding_score >= thresholds["exact_embedding"] and lexical_score >= thresholds["exact_lexical"]:
        return "exact"
    if embedding_score >= thresholds["close_embedding"]:
        return "close"
    if embedding_score >= thresholds["related_embedding"]:
        return "related"
    return "novel"


# ------------------ Deterministic parse helpers (replaced in Phase 2) ------------------

def _serialize_docx_table(table) -> str:
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    sep_cells = ["---" if cell else "" for cell in table.rows[0].cells]
    sep_line = "| " + " | ".join(sep_cells) + " |"
    return "\n".join([rows[0], sep_line] + rows[1:])


def simple_parse_docx(path: str) -> List[DocumentChunk]:
    try:
        from docx import Document
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("python-docx is required to parse .docx files") from exc

    doc = Document(path)
    chunks: List[DocumentChunk] = []
    paragraph_texts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    for idx, p in enumerate(paragraph_texts):
        chunks.append(
            DocumentChunk(
                doc_id=os.path.basename(path),
                source_file=path,
                chunk_type="text",
                page_or_section=f"para_{idx + 1}",
                content=p.strip(),
                char_count=len(p.strip()),
            )
        )

    for idx, table in enumerate(doc.tables):
        table_text = _serialize_docx_table(table)
        if table_text:
            chunks.append(
                DocumentChunk(
                    doc_id=os.path.basename(path),
                    source_file=path,
                    chunk_type="table",
                    page_or_section=f"table_{idx + 1}",
                    content=table_text,
                    char_count=len(table_text),
                )
            )
    return chunks


def simple_parse_pdf(path: str) -> List[DocumentChunk]:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pymupdf (fitz) is required to parse PDFs") from exc

    doc = fitz.open(path)
    chunks: List[DocumentChunk] = []
    for pnum in range(len(doc)):
        page = doc.load_page(pnum)
        text = page.get_text("text")
        chunk_type = "table" if is_table_like_text(text) else "text"
        chunks.append(
            DocumentChunk(
                doc_id=os.path.basename(path),
                source_file=path,
                chunk_type=chunk_type,
                page_or_section=f"page_{pnum + 1}",
                content=text.strip(),
                char_count=len(text.strip()),
            )
        )
    return chunks


def extract_chunks_from_file(path: str) -> Tuple[List[DocumentChunk], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            chunks = simple_parse_pdf(path)
        elif ext == ".docx":
            chunks = simple_parse_docx(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        if not chunks:
            errors.append({"doc_id": os.path.basename(path), "source_file": path, "error": "no content extracted"})
        return chunks, errors
    except Exception as exc:
        errors.append({"doc_id": os.path.basename(path), "source_file": path, "error": str(exc)})
        return [], errors


def simple_parse_docx_text(path: str) -> List[DocumentChunk]:
    return simple_parse_docx(path)


def simple_parse_pdf_text(path: str) -> List[DocumentChunk]:
    return simple_parse_pdf(path)


# ------------------ Reference library (legacy CSV format) ------------------

def _split_indikit_wording(wording: str) -> Tuple[str, str]:
    """Split an IndiKit "Indicator's Wording" block into (english, french).

    The block is normally "English\\nFrench\\n<english wording>\\n<french wording>", with occasional
    hard line breaks (\\r\\n) inside a sentence. Returns empty strings when nothing usable is present.
    Same logic as ``pipeline.metadata.build_metadata.split_indikit_wording``; kept here so the legacy
    converter below has no dependency on the metadata module.
    """
    if not wording or not isinstance(wording, str):
        return "", ""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in wording.replace("\r\n", " ").split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return "", ""
    if len(lines) >= 3 and lines[0].lower() == "english" and lines[1].lower() == "french":
        return lines[2], " ".join(lines[3:])
    return lines[0], ""


def build_reference_library_from_zoa(zoa_path: str, out_path: str):
    """Convert the scraped IndiKit site CSV into the reference_library.csv format the matcher expects.

    Superseded by ``python -m pipeline.run build-metadata``, which writes the same file from the merged
    metadata (export + site). Kept because the notebook and a test still call it.

    Output columns: indicator_id (IndiKit slug), indicator_text (English wording, or title when wording is
    missing), indicator_text_fr, title, sector, level (blank: IndiKit has no result level), disaggregation,
    url, text_source ("wording" | "title"). Returns the DataFrame that was written.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to build the reference library") from exc

    zoa = pd.read_csv(zoa_path)
    wording_col = next((c for c in zoa.columns if c.startswith("section:Indicator") and "Wording" in c), None)
    disagg_col = next((c for c in zoa.columns if c.startswith("section:Disaggregate")), None)

    rows = []
    for _, r in zoa.iterrows():
        english, french = _split_indikit_wording(r[wording_col] if wording_col and pd.notna(r[wording_col]) else "")
        title = str(r.get("title", "")).strip()
        text = english or title
        disagg = str(r[disagg_col]).replace("\r\n", " ").replace("\n", " ").strip() if disagg_col and pd.notna(r[disagg_col]) else ""
        rows.append(
            {
                "indicator_id": str(r.get("slug", "")).strip(),
                "indicator_text": text,
                "indicator_text_fr": french,
                "title": title,
                "sector": str(r.get("sector", "")).strip(),
                "level": "",
                "disaggregation": re.sub(r"\s+", " ", disagg),
                "url": str(r.get("url", "")).strip(),
                "text_source": "wording" if english else "title",
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["indicator_text"].astype(str).str.strip() != ""]
    df = df.sort_values("indicator_id").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def load_reference_library(path: str):
    """Load ``reference_library.csv`` (columns indicator_id, indicator_text, sector, level).

    Returns ``(DataFrame, False)``; the second element used to flag a dummy fallback library, which no
    longer exists. A missing file raises FileNotFoundError instead of silently substituting dummy rows.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to load the reference library") from exc

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reference library not found at {path}. Generate it from the IndiKit metadata with: "
            "python -m pipeline.run build-metadata"
        )
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ["indicator_id", "indicator_text", "sector", "level"]:
        if col not in df.columns:
            df[col] = ""
    return df, False


def _best_lexical_match(query: str, choices: List[str]) -> Tuple[int, int]:
    """Deprecated (matching v1)."""
    _deprecated("_best_lexical_match")
    if fuzz is None:
        raise RuntimeError("rapidfuzz is required for lexical matching")
    best_score = -1
    best_idx = -1
    for idx, choice in enumerate(choices):
        score = fuzz.token_sort_ratio(query, choice)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


# Embedding backends, tried in order. fastembed runs the same MiniLM model through ONNX Runtime and needs
# no PyTorch, which matters on Intel Macs where PyTorch stopped at 2.2.2 (incompatible with numpy 2 and
# transformers 5). sentence-transformers is kept as a fallback for machines where PyTorch works.
EMBEDDING_BACKENDS = ("fastembed", "sentence-transformers")
_EMBEDDING_MODEL_CACHE: Dict[Tuple[str, str], object] = {}
_EMBEDDING_BACKEND_USED: Optional[str] = None


def _fastembed_model_name(model_name: str) -> str:
    """fastembed registers Sentence-Transformers models under their full Hugging Face id."""
    if "/" in model_name:
        return model_name
    return f"sentence-transformers/{model_name}"


def _load_embedding_model(backend: str, model_name: str):
    key = (backend, model_name)
    if key in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[key]
    if backend == "fastembed":
        from fastembed import TextEmbedding

        model = TextEmbedding(_fastembed_model_name(model_name))
    elif backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
    _EMBEDDING_MODEL_CACHE[key] = model
    return model


def _compute_embeddings(texts: List[str], model_name: str = _config.EMBEDDING_MODEL):
    """Return L2-normalised embeddings (numpy array, shape n x dim) using the first backend that works.

    Raises RuntimeError listing every backend failure when none is usable.
    """
    global _EMBEDDING_BACKEND_USED
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for embedding similarity") from exc

    failures = []
    backends = (_EMBEDDING_BACKEND_USED,) if _EMBEDDING_BACKEND_USED else EMBEDDING_BACKENDS
    for backend in backends:
        try:
            model = _load_embedding_model(backend, model_name)
            if backend == "fastembed":
                embeddings = np.asarray(list(model.embed(list(texts))), dtype="float32")
            else:
                embeddings = np.asarray(
                    model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False),
                    dtype="float32",
                )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
                raise RuntimeError(f"{backend} returned embeddings of shape {embeddings.shape} for {len(texts)} texts")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            if _EMBEDDING_BACKEND_USED != backend:
                _EMBEDDING_BACKEND_USED = backend
                print(f"Embeddings: using {backend} backend with model {model_name}")
            return embeddings
        except Exception as exc:
            failures.append(f"{backend}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
    raise RuntimeError("no embedding backend available (" + "; ".join(failures) + ")")


def _best_embedding_matches(query_embeddings, reference_embeddings):
    try:
        import numpy as np  # noqa: F401
    except Exception as exc:
        raise RuntimeError("numpy is required for embedding similarity") from exc
    query = query_embeddings.reshape(1, -1)
    scores = reference_embeddings.dot(query.T).flatten().tolist()
    best_idx = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else -1
    return best_idx, float(scores[best_idx]) if best_idx >= 0 else 0.0


def match_indicators_to_library(indicators: List[Indicator], reference_path: str, embedding_model: Optional[str] = _config.EMBEDDING_MODEL):
    """Deprecated (matching v1): best lexical + embedding match per indicator with threshold classes.

    Returns ``(DataFrame, False)``. The second element used to flag the dummy library, which was removed;
    a missing reference file now raises FileNotFoundError.
    """
    _deprecated("match_indicators_to_library")
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for matching indicators") from exc

    reference_df, dummy_library = load_reference_library(reference_path)
    ref_texts = reference_df["indicator_text"].astype(str).tolist()
    match_rows = []

    if not indicators:
        return pd.DataFrame(match_rows), dummy_library

    extracted_texts = [ind.indicator_text for ind in indicators]

    # Lexical matches (always available)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lex_results = [_best_lexical_match(q, ref_texts) for q in extracted_texts]

    # Embedding matches (optional)
    use_embeddings = bool(embedding_model)
    ref_embeddings = None
    query_embeddings = None
    if use_embeddings:
        try:
            ref_embeddings = _compute_embeddings(ref_texts, embedding_model)
            query_embeddings = _compute_embeddings(extracted_texts, embedding_model)
        except Exception as exc:
            # If embeddings fail, fall back to lexical-only, but say so loudly
            print(f"WARNING: embedding similarity unavailable ({type(exc).__name__}: {exc}). Falling back to lexical matching only.")
            use_embeddings = False
            ref_embeddings = None
            query_embeddings = None

    for idx, ind in enumerate(indicators):
        lex_idx, lex_score = lex_results[idx]
        emb_idx, emb_score = -1, 0.0
        if use_embeddings and query_embeddings is not None and ref_embeddings is not None:
            emb_idx, emb_score = _best_embedding_matches(query_embeddings[idx], ref_embeddings)
        ref_lex = reference_df.iloc[lex_idx] if lex_idx >= 0 else None
        ref_emb = reference_df.iloc[emb_idx] if emb_idx >= 0 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            status = classify_match_status(emb_score, lex_score)
        match_rows.append(
            {
                "doc_id": ind.doc_id,
                "indicator_text": ind.indicator_text,
                "level": ind.level,
                "baseline": ind.baseline,
                "target": ind.target,
                "means_of_verification": ind.means_of_verification,
                "sector": ind.sector,
                "disaggregation": ind.disaggregation,
                "unit_or_type": ind.unit_or_type,
                "source_confidence": ind.source_confidence,
                "reference_id_lexical": ref_lex["indicator_id"] if ref_lex is not None else "",
                "reference_text_lexical": ref_lex["indicator_text"] if ref_lex is not None else "",
                "lexical_score": lex_score,
                "reference_id_embedding": ref_emb["indicator_id"] if ref_emb is not None else "",
                "reference_text_embedding": ref_emb["indicator_text"] if ref_emb is not None else "",
                "embedding_score": emb_score,
                "match_status": status,
                "dummy_library": dummy_library,
            }
        )

    return pd.DataFrame(match_rows), dummy_library


# ------------------ Linking v2: embedding shortlist + LLM judgement (retired by the Phase 3 agent) ------------------

LINK_PROMPT_VERSION = _config.LINK_PROMPT_VERSION
LINK_STATUSES = ("Present", "Different", "Missing")


def shortlist_reference_candidates(
    indicator_texts: List[str],
    reference_df,
    k: int = 5,
    embedding_model: Optional[str] = _config.EMBEDDING_MODEL,
) -> List[List[Dict[str, object]]]:
    """For each proposal indicator, return the k most similar reference indicators.

    Uses sentence embeddings (cosine similarity) over the whole library; falls back to rapidfuzz
    token_sort_ratio if embeddings are unavailable. Each candidate is a dict with indicator_id,
    indicator_text, sector, score (0-1) and method ("embedding" | "lexical").
    """
    ref_ids = reference_df["indicator_id"].astype(str).tolist()
    ref_texts = reference_df["indicator_text"].astype(str).tolist()
    ref_sectors = reference_df["sector"].astype(str).tolist() if "sector" in reference_df.columns else [""] * len(ref_texts)
    if not indicator_texts or not ref_texts:
        return [[] for _ in indicator_texts]
    k = max(1, min(k, len(ref_texts)))

    def _cand(j: int, score: float, method: str) -> Dict[str, object]:
        return {"indicator_id": ref_ids[j], "indicator_text": ref_texts[j], "sector": ref_sectors[j], "score": round(float(score), 4), "method": method}

    if embedding_model:
        try:
            import numpy as np

            ref_emb = _compute_embeddings(ref_texts, embedding_model)
            q_emb = _compute_embeddings(indicator_texts, embedding_model)
            sims = q_emb @ ref_emb.T
            out = []
            for i in range(len(indicator_texts)):
                top = np.argsort(-sims[i])[:k]
                out.append([_cand(int(j), sims[i][j], "embedding") for j in top])
            return out
        except Exception as exc:
            print(f"WARNING: embedding shortlist unavailable ({type(exc).__name__}: {exc}). Falling back to lexical shortlist.")

    if fuzz is None:
        raise RuntimeError("rapidfuzz is required for the lexical shortlist fallback")
    out = []
    for q in indicator_texts:
        scored = sorted(((fuzz.token_sort_ratio(q, t) / 100.0, j) for j, t in enumerate(ref_texts)), reverse=True)[:k]
        out.append([_cand(j, s, "lexical") for s, j in scored])
    return out


def _build_link_prompt(indicator_text: str, candidates: List[Dict[str, object]]) -> str:
    cand_lines = "\n".join(
        f"  {n}. id={c['indicator_id']} | sector={c['sector']} | {c['indicator_text']}" for n, c in enumerate(candidates, 1)
    )
    return (
        "You are an M&E specialist comparing a project proposal indicator with the ZOA reference indicator library.\n\n"
        f"PROPOSAL INDICATOR:\n\"{indicator_text}\"\n\n"
        f"CLOSEST ZOA REFERENCE INDICATORS (pre-selected by similarity):\n{cand_lines}\n\n"
        "TASK: decide how the proposal indicator relates to the reference library and answer with ONE status:\n"
        "- \"Present\": a candidate measures the same thing. Differences only in target group, location, numbers, or minor wording.\n"
        "- \"Different\": a candidate covers the same topic, but the proposal indicator measures it in a materially different way "
        "(different unit, scope, calculation, or population definition), so it would need adapting to fit the reference.\n"
        "- \"Missing\": none of the candidates cover the topic of the proposal indicator.\n\n"
        "Respond with JSON only, no extra text, using exactly these keys:\n"
        "{\"reference_id\": \"<id of the best candidate, or null when Missing>\", "
        "\"status\": \"Present\" | \"Different\" | \"Missing\", "
        "\"confidence\": <number between 0 and 1>, "
        "\"reason\": \"<one sentence>\"}"
    )


def link_indicator_to_reference(
    indicator_text: str,
    candidates: List[Dict[str, object]],
    model_name: Optional[str] = None,
    prompt_version: str = LINK_PROMPT_VERSION,
    cache_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Ask the LLM whether a proposal indicator is Present, Different or Missing relative to the shortlisted candidates.

    Responses are cached on disk keyed by a hash of (prompt, model, prompt_version). Returns a dict with
    reference_id, reference_text, status, confidence, reason, and cached (bool). On API failure status is "Error".
    """
    if not candidates:
        return {"reference_id": None, "reference_text": None, "status": "Missing", "confidence": 1.0, "reason": "No reference candidates available.", "cached": False}

    model_name = model_name or _answer.MODEL_NAME
    cache_dir = cache_dir or _answer.CACHE_DIR
    prompt = _build_link_prompt(indicator_text, candidates)
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(model_name.encode("utf-8"))
    h.update(prompt_version.encode("utf-8"))
    cache_path = os.path.join(cache_dir, f"link_{h.hexdigest()}.json")

    raw = None
    cached = False
    if not _answer.FORCE_REFRESH and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh).get("response")
                cached = raw is not None
        except Exception:
            raw = None

    if raw is None:
        try:
            text = _answer._call_openai_with_retries(prompt, model_name, response_format={"type": "json_object"}, temperature=0.0, max_tokens=300)
        except Exception as exc:
            return {"reference_id": None, "reference_text": None, "status": "Error", "confidence": 0.0, "reason": f"{type(exc).__name__}: {exc}", "cached": False}
        raw = _extract_json_from_response(text)
        if not isinstance(raw, dict):
            raw = {"reference_id": None, "status": "Error", "confidence": 0.0, "reason": f"Unparseable response: {text[:200]}"}
        else:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump({"prompt_version": prompt_version, "model": model_name, "indicator_text": indicator_text, "response": raw}, fh, ensure_ascii=False, indent=2)
            except Exception:
                pass

    # Normalise and validate against the candidate list so the model cannot invent a reference id.
    by_id = {str(c["indicator_id"]): c for c in candidates}
    status = str(raw.get("status", "")).strip().capitalize()
    if status not in LINK_STATUSES:
        status = "Error" if status.lower() == "error" else "Missing"
    ref_id = raw.get("reference_id")
    ref_id = str(ref_id).strip() if ref_id not in (None, "", "null") else None
    if ref_id is not None and ref_id not in by_id:
        # tolerate the model echoing the text instead of the id
        text_match = next((c for c in candidates if str(c["indicator_text"]).strip().lower() == str(ref_id).strip().lower()), None)
        ref_id = str(text_match["indicator_id"]) if text_match else None
    if status == "Missing":
        ref_id = None
    if status in ("Present", "Different") and ref_id is None:
        ref_id = str(candidates[0]["indicator_id"])
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "reference_id": ref_id,
        "reference_text": by_id[ref_id]["indicator_text"] if ref_id else None,
        "status": status,
        "confidence": confidence,
        "reason": str(raw.get("reason", "")).strip(),
        "cached": cached,
    }


# ------------------ Attribute forwarding (keeps old notebook code and tests working) ------------------

class _ShimModule(types.ModuleType):
    """Module type that forwards the runtime settings to ``pipeline.agent.answer`` and warns on THRESHOLDS."""

    def __getattr__(self, name: str):  # only called when normal lookup fails
        if name in _FORWARDED:
            return getattr(_answer, name)
        if name == "THRESHOLDS":
            _deprecated("THRESHOLDS")
            return _THRESHOLDS
        raise AttributeError(f"module 'extraction_utils' has no attribute '{name}'")

    def __setattr__(self, name: str, value) -> None:
        if name in _FORWARDED:
            setattr(_answer, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ShimModule

# End of module
