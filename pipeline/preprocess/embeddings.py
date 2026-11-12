"""Embedders, vector index and embedding cache (paper Section 3.3.2 vectors; 3.4.1 cosine retrieval).

``Embedder.embed(texts) -> np.ndarray`` returns L2-normalised rows. Implementations:

- ``HashingEmbedder``: deterministic hashed word + character n-grams, no dependency (tests, smoke runs);
- ``FastembedEmbedder``: ONNX models through fastembed (default backend; no PyTorch, works on Intel Macs);
- ``SentenceTransformerEmbedder``: sentence-transformers / PyTorch.

``get_embedder("hashing" | "fastembed:<model>" | "st:<model>" | "<model>")`` is the factory. ``VectorIndex`` is a
NumPy inner-product index (FAISS is used when ``faiss`` is importable) persisted as ``.npy`` + id map.
``EmbeddingCache`` memoises vectors by (model, text hash) in a SQLite file so re-runs cost nothing.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from pipeline import config


def l2_normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float32")
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


class Embedder(ABC):
    name: str = "embedder"
    dim: int = 0
    batch_size: int = 64

    @abstractmethod
    def _embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        out = [self._embed(texts[i : i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
        return l2_normalise(np.vstack(out))


class HashingEmbedder(Embedder):
    """Deterministic bag of hashed word unigrams, bigrams and character trigrams (dependency-free)."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hashing-{dim}"

    @staticmethod
    def _features(text: str) -> Iterable[str]:
        words = re.findall(r"\w+", text.lower())
        yield from words
        yield from (f"{a}_{b}" for a, b in zip(words, words[1:]))
        for w in words:
            if len(w) >= 5:
                yield from (f"#{w[i:i + 3]}" for i in range(len(w) - 2))

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for f in self._features(t):
                h = int(hashlib.blake2b(f.encode("utf-8"), digest_size=8).hexdigest(), 16)
                mat[i, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return mat


class FastembedEmbedder(Embedder):
    def __init__(self, model_name: str = config.KB_EMBEDDING_MODEL, batch_size: int = 64):
        from fastembed import TextEmbedding

        self.name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name)
        self.dim = int(self._embed(["dimension probe"]).shape[1])

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(list(texts), batch_size=self.batch_size)), dtype="float32")


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = config.KB_EMBEDDING_MODEL, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension() or self._embed(["probe"]).shape[1])

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False), dtype="float32")


def get_embedder(spec: Optional[str] = None) -> Embedder:
    """``"hashing"`` / ``"hashing:512"`` / ``"fastembed:<model>"`` / ``"st:<model>"`` / ``"<model>"`` (fastembed first, then
    sentence-transformers)."""
    spec = spec or config.KB_EMBEDDING_MODEL
    if spec.startswith("hashing"):
        dim = int(spec.split(":", 1)[1]) if ":" in spec else 256
        return HashingEmbedder(dim)
    if spec.startswith("fastembed:"):
        return FastembedEmbedder(spec.split(":", 1)[1])
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec.split(":", 1)[1])
    errors = []
    for cls in (FastembedEmbedder, SentenceTransformerEmbedder):
        try:
            return cls(spec)
        except Exception as exc:  # model not supported by this backend / backend missing
            errors.append(f"{cls.__name__}: {type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError(f"no embedding backend could load '{spec}': " + "; ".join(errors))


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #
class EmbeddingCache:
    """SQLite-backed (model, sha256(text)) -> vector cache. ``None`` path disables caching."""

    def __init__(self, path: Optional[Path] = config.EMBEDDING_CACHE_DB):
        self.path = Path(path) if path else None
        self._conn: Optional[sqlite3.Connection] = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
            self._conn.execute("CREATE TABLE IF NOT EXISTS vectors (model TEXT, sha TEXT, dim INTEGER, vec BLOB, PRIMARY KEY (model, sha))")
            self._conn.commit()

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, embedder.dim), dtype="float32")
        if self._conn is None:
            return embedder.embed(texts)
        shas = [self._sha(t) for t in texts]
        rows = {}
        for i in range(0, len(shas), 500):
            chunk = shas[i : i + 500]
            q = f"SELECT sha, dim, vec FROM vectors WHERE model=? AND sha IN ({','.join('?' * len(chunk))})"
            for sha, dim, vec in self._conn.execute(q, [embedder.name, *chunk]):
                rows[sha] = np.frombuffer(vec, dtype="float32").reshape(dim)
        missing = [i for i, s in enumerate(shas) if s not in rows]
        if missing:
            new = embedder.embed([texts[i] for i in missing])
            for i, vec in zip(missing, new):
                rows[shas[i]] = vec
                self._conn.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)", (embedder.name, shas[i], int(vec.shape[0]), vec.astype("float32").tobytes()))
            self._conn.commit()
        return l2_normalise(np.vstack([rows[s] for s in shas]))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# --------------------------------------------------------------------------- #
# Vector index
# --------------------------------------------------------------------------- #
class VectorIndex:
    """Inner-product search over L2-normalised vectors (cosine). NumPy brute force; FAISS when available."""

    def __init__(self, ids: list[str], vectors: np.ndarray):
        self.ids = list(ids)
        self.vectors = l2_normalise(vectors) if len(ids) else np.zeros((0, 0), dtype="float32")
        self._faiss = None
        try:
            import faiss  # type: ignore

            if len(ids):
                idx = faiss.IndexFlatIP(self.vectors.shape[1])
                idx.add(self.vectors)
                self._faiss = idx
        except Exception:
            self._faiss = None

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: np.ndarray, top_n: int) -> list[tuple[str, float]]:
        if not self.ids or top_n <= 0:
            return []
        q = l2_normalise(np.asarray(query, dtype="float32"))
        top_n = min(top_n, len(self.ids))
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, top_n)
            return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
        scores = self.vectors @ q[0]
        order = np.argsort(-scores, kind="stable")[:top_n]
        return [(self.ids[i], float(scores[i])) for i in order]

    def save(self, prefix: Path) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(prefix.with_suffix(".npy")), self.vectors)
        prefix.with_suffix(".ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, prefix: Path) -> "VectorIndex":
        prefix = Path(prefix)
        ids = json.loads(prefix.with_suffix(".ids.json").read_text(encoding="utf-8"))
        vectors = np.load(str(prefix.with_suffix(".npy"))) if ids else np.zeros((0, 0), dtype="float32")
        return cls(ids, vectors)
