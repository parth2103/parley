"""Hybrid retrieval combining Okapi BM25 and Dense Vector Scoring via Reciprocal Rank Fusion.

Uses standard Python and NumPy for zero external dependencies.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Sequence

import numpy as np

from backend.rag.corpus import CORPUS, PolicyChunk


def tokenize(text: str) -> list[str]:
    """Lowercase word, alphanumeric, and currency tokenization."""
    return re.findall(r"\$?[a-zA-Z0-9-]+", text.lower())


class BM25Index:
    """Okapi BM25 index with k1=1.5 and b=0.75."""

    def __init__(self, corpus: Sequence[PolicyChunk], k1: float = 1.5, b: float = 0.75):
        self.corpus = list(corpus)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(f"{c.title} {c.content}") for c in self.corpus]
        self.doc_lengths = np.array([len(tokens) for tokens in self.doc_tokens], dtype=float)
        self.avg_dl = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) > 0 else 1.0

        # Inverted document frequency
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n_docs = len(self.corpus)
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        scores = np.zeros(len(self.corpus), dtype=float)
        for term in q_tokens:
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            for idx, d_tokens in enumerate(self.doc_tokens):
                tf = d_tokens.count(term)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[idx] / self.avg_dl))
                    scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores


class DenseVectorIndex:
    """Subword hashed n-gram dense vector index with cosine similarity.

    Encodes tokens and character 3-grams into normalized dense vector space,
    capturing morphological variants, compound terms, and numeric patterns.
    """

    def __init__(self, corpus: Sequence[PolicyChunk], dim: int = 512):
        self.corpus = list(corpus)
        self.dim = dim
        self.vectors = np.zeros((len(self.corpus), self.dim), dtype=float)
        for idx, c in enumerate(self.corpus):
            self.vectors[idx] = self._encode(f"{c.title} {c.content}")

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=float)
        tokens = tokenize(text)
        for t in tokens:
            h = zlib.crc32(t.encode("utf-8")) % self.dim
            vec[h] += 1.0
            # Also hash character 3-grams for subword similarity
            if len(t) >= 3:
                for j in range(len(t) - 2):
                    sub = t[j : j + 3]
                    h_sub = zlib.crc32(sub.encode("utf-8")) % self.dim
                    vec[h_sub] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def score(self, query: str) -> np.ndarray:
        q_vec = self._encode(query)
        if np.linalg.norm(q_vec) == 0:
            return np.zeros(len(self.corpus), dtype=float)
        return np.dot(self.vectors, q_vec)


class HybridRetriever:
    """Combines BM25 and dense vectors via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        corpus: Sequence[PolicyChunk] = CORPUS,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        self.corpus = list(corpus)
        self.bm25 = BM25Index(self.corpus)
        self.dense = DenseVectorIndex(self.corpus)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[PolicyChunk, float]]:
        bm25_scores = self.bm25.score(query)
        dense_scores = self.dense.score(query)

        # Compute ranks (0-indexed rank from highest score)
        bm25_ranked = np.argsort(-bm25_scores)
        dense_ranked = np.argsort(-dense_scores)

        bm25_ranks = {doc_idx: rank for rank, doc_idx in enumerate(bm25_ranked)}
        dense_ranks = {doc_idx: rank for rank, doc_idx in enumerate(dense_ranked)}

        rrf_scores = np.zeros(len(self.corpus), dtype=float)
        for idx in range(len(self.corpus)):
            bm25_rrf = self.bm25_weight / (self.rrf_k + bm25_ranks[idx] + 1)
            dense_rrf = self.dense_weight / (self.rrf_k + dense_ranks[idx] + 1)
            rrf_scores[idx] = bm25_rrf + dense_rrf

        top_indices = np.argsort(-rrf_scores)[:top_k]
        return [(self.corpus[i], float(rrf_scores[i])) for i in top_indices]
