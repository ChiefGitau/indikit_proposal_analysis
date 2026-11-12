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
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def get_model(model_name: str = DEFAULT_MODEL):
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
            return cached, "sentence-transformers (cached)"

    try:
        vecs = get_model(model_name).encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if use_disk:
            np.save(EMB_PATH, vecs)
        return vecs, "sentence-transformers"
    except ImportError:
        return _tfidf(texts), "TF-IDF (sentence-transformers not installed)"


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
