from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from src.index.inverted_index import InvertedIndex
from src.utils.text import tokenize


@dataclass(frozen=True)
class SearchResult:
    book_id: str
    score: float
    display: dict[str, str]
    record: Any

    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id,
            "score": self.score,
            "display": dict(self.display),
            "record": self.record,
        }


def search(
    index: InvertedIndex,
    query: str,
    *,
    top_k: int = 10,
) -> list[SearchResult]:
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = _score_tokens(index, tokens)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results: list[SearchResult] = []
    for doc_id, score in ranked:
        results.append(
            SearchResult(
                book_id=doc_id,
                score=score,
                display=index.doc_display.get(doc_id, {}),
                record=index.doc_records.get(doc_id),
            )
        )
    return results


def _score_tokens(index: InvertedIndex, tokens: Iterable[str]) -> dict[str, float]:
    query_counts = Counter(tokens)
    query_norm = 0.0
    query_weights: dict[str, float] = {}
    for term, tf in query_counts.items():
        idf = index.idf.get(term)
        if idf is None:
            continue
        tf_weight = 1.0 + math.log(tf)
        weight = tf_weight * idf
        query_weights[term] = weight
        query_norm += weight * weight
    query_norm = math.sqrt(query_norm) if query_norm > 0 else 0.0
    if query_norm == 0:
        return {}

    scores: dict[str, float] = {}
    for term, query_weight in query_weights.items():
        postings = index.postings.get(term, {})
        idf = index.idf.get(term, 0.0)
        for doc_id, tf in postings.items():
            tf_weight = 1.0 + math.log(tf)
            doc_weight = tf_weight * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + query_weight * doc_weight

    for doc_id, score in list(scores.items()):
        denom = index.doc_norms.get(doc_id, 0.0) * query_norm
        if denom > 0:
            scores[doc_id] = score / denom
        else:
            scores[doc_id] = 0.0
    return scores
