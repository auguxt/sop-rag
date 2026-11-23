"""Embedding models.

Two back-ends are supported:

* ``huggingface`` - real semantic embeddings (BAAI/bge-small-en-v1.5) through
  ``llama-index-embeddings-huggingface``. Recommended. The model (~130 MB) is
  downloaded once on the first run and cached.
* ``local``       - an offline, dependency-free *hashing* embedding written
  for this project. It needs no download and no internet, so the project
  always runs (e.g. in an exam lab without internet).

``auto`` (the default) tries huggingface first and silently falls back to
local if the package or the model download is not available.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import List, Tuple

from llama_index.core.base.embeddings.base import BaseEmbedding

from .text_utils import GENERIC_DOMAIN_WORDS, STOPWORDS, stem, tokenize

log = logging.getLogger(__name__)


class LocalHashEmbedding(BaseEmbedding):
    """Deterministic feature-hashing embedding (no training, no download).

    Features: word stems (weight 1.0), word bigrams (0.5) and character
    4-grams of the stems (0.25, gives robustness to spelling variants).
    Each feature is hashed with MD5 (stable between runs, unlike Python's
    built-in ``hash``) into a fixed-size vector which is then L2-normalised,
    so cosine similarity behaves like a weighted keyword overlap.
    """

    dim: int = 1024

    def __init__(self, dim: int = 1024, **kwargs):
        super().__init__(dim=dim, model_name="local-hash-embedding", **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "LocalHashEmbedding"

    # -- feature extraction -------------------------------------------------
    def _features(self, text: str) -> List[Tuple[str, float]]:
        stems = []
        for tok in tokenize(text):
            if tok in STOPWORDS:
                continue
            weight = 0.35 if tok in GENERIC_DOMAIN_WORDS else 1.0
            stems.append((stem(tok), weight))
        feats: List[Tuple[str, float]] = []
        for s, w in stems:
            feats.append(("w:" + s, w))
            padded = f"#{s}#"
            for i in range(len(padded) - 3):
                feats.append(("c:" + padded[i: i + 4], 0.25 * w))
        for (a, wa), (b, wb) in zip(stems, stems[1:]):
            feats.append((f"b:{a}_{b}", 0.5 * min(wa, wb)))
        return feats

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for feat, weight in self._features(text):
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 64) & 1 else -1.0
            vec[idx] += sign * weight
        if not any(vec):          # empty / punctuation-only text: avoid a zero vector (NaN cosine)
            vec[0] = 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # -- BaseEmbedding interface ---------------------------------------------
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)


def get_embed_model(backend: str = "auto", hf_model: str = "BAAI/bge-small-en-v1.5") -> Tuple[BaseEmbedding, str]:
    """Return ``(embedding_model, human_readable_name)`` for the requested back-end."""
    backend = (backend or "auto").lower().strip()
    if backend in ("auto", "huggingface", "hf"):
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            model = HuggingFaceEmbedding(model_name=hf_model)
            model.get_text_embedding("warm-up")  # forces the download / load now
            return model, f"huggingface:{hf_model}"
        except Exception as exc:  # package missing, no internet, ...
            if backend != "auto":
                log.warning("HuggingFace embeddings unavailable (%s); using local hashing embeddings.", exc)
            else:
                log.info("HuggingFace embeddings unavailable (%s); using local hashing embeddings.", exc)
    elif backend != "local":
        log.warning("Unknown embed backend %r; using local hashing embeddings.", backend)
    return LocalHashEmbedding(), "local-hash-embedding"
