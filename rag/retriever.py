from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

__all__ = [
    "CorpusEntry",
    "RetrieverHit",
    "TfidfRetriever",
]


@dataclass(frozen=True)
class CorpusEntry:
    """Canonical representation of an item indexed by a retriever."""

    id: str
    content: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class RetrieverHit:
    """Result returned by a retriever search."""

    entry: CorpusEntry
    score: float


class TfidfRetriever:
    """Thin wrapper around a TF-IDF vectoriser + cosine similarity search."""

    def __init__(
        self,
        corpus: Sequence[CorpusEntry],
        *,
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_range: tuple[int, int] = (1, 2),
        lowercase: bool = True,
    ) -> None:
        if not corpus:
            raise ValueError("The corpus must contain at least one entry.")

        self._corpus = list(corpus)
        self._vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=lowercase,
            strip_accents="unicode",
        )
        self._matrix = self._vectorizer.fit_transform(entry.content for entry in self._corpus)

    @property
    def corpus(self) -> Iterable[CorpusEntry]:
        return iter(self._corpus)

    def search(self, query: str, *, top_k: int = 3) -> list[RetrieverHit]:
        """Return the `top_k` closest entries to the query."""

        if not query.strip():
            return []

        top_k = max(1, top_k)
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix).astype(np.float32)[0]
        if np.isnan(similarities).any():
            similarities = np.nan_to_num(similarities)

        indices = np.argsort(similarities)[::-1][:top_k]
        return [
            RetrieverHit(entry=self._corpus[idx], score=float(similarities[idx]))
            for idx in indices
        ]
