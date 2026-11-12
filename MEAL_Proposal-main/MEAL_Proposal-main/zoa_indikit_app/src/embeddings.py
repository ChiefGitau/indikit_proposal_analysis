"""Embedding + 2D-projection cache, with graceful fallbacks.

Quality path (when installed):
    sentence-transformers  →  semantic embeddings
    umap-learn             →  UMAP projection

Fallback path (core deps only — sklearn, numpy):
    TF-IDF vectors         →  lexical "embeddings"
    PCA                    →  2D projection

Each function returns ``(array, method_label)`` so pages can tell the user
which path was used. Heavy imports are lazy so cold start stays fast.
"""

from pathlib import Path

import numpy as np
import streamlit as st

EMB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings.npy"
# Shared with the pipeline (pipeline/config.py KB_EMBEDDING_MODEL): the multilingual model, because the documents
# and part of the IndiKit wording are French as well as English. The on-disk cache is keyed on this name.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# The pipeline's metadata file (one record per unique IndiKit ID, with Knowledge / SearchTerm extensions).
METADATA_JSONL = None
try:
    from src.config import METADATA_JSONL  # noqa: F401
except Exception:
    pass


@st.cache_resource
def get_model(model_name: str = DEFAULT_MODEL):
    """fastembed (ONNX, no PyTorch) first - the backend the pipeline uses - then sentence-transformers."""
    try:
        from fastembed import TextEmbedding

        class _Wrapped:
            def __init__(self, name):
                self._m = TextEmbedding(model_name=name)

            def encode(self, texts, **kw):
                vecs = np.asarray(list(self._m.embed(list(texts))), dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return vecs / norms

        return _Wrapped(model_name)
    except Exception:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)


def _tfidf(texts):
    """Lexical fallback: L2-normalised TF-IDF (cosine-ready)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", max_features=512)
    mat = vec.fit_transform(list(texts))
    arr = mat.toarray().astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


@st.cache_data(show_spinner="Embedding indicators…")
def embed_indicators(texts: tuple, model_name: str = DEFAULT_MODEL):
    """Return ``(vectors, method)``. Tries sentence-transformers, else TF-IDF."""
    use_disk = model_name == DEFAULT_MODEL
    if use_disk and EMB_PATH.exists():
        cached = np.load(EMB_PATH)
        if len(cached) == len(texts):
            return cached, "embeddings (cached)"

    try:
        vecs = get_model(model_name).encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if use_disk:
            np.save(EMB_PATH, vecs)
        return vecs, DEFAULT_MODEL.split("/")[-1]
    except ImportError:
        return _tfidf(texts), "TF-IDF (fastembed / sentence-transformers not installed)"


@st.cache_data(show_spinner="Projecting to 2D…")
def project_2d(vecs: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1):
    """Return ``(xy, method)``. Tries UMAP, else PCA."""
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=min(n_neighbors, max(2, len(vecs) - 1)),
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(vecs), "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(vecs), "PCA (umap not installed)"
