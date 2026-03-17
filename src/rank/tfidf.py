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
    matched_terms: list[str]
    why_matched: str

    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id,
            "score": self.score,
            "display": dict(self.display),
            "record": self.record,
            "matched_terms": list(self.matched_terms),
            "why_matched": self.why_matched,
        }


def search(
    index: InvertedIndex,
    query: str,
    *,
    top_k: int = 10,
    preferred_genres: list[str] | None = None,
    preferred_authors: list[str] | None = None,
    preferred_moods: list[str] | None = None,
    preferred_time: str | None = None,
    genre_boost: float = 0.2,
    author_boost: float = 0.3,
    mood_boost: float = 0.2,
    time_boost: float = 0.2,
    user_prefs: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> list[SearchResult]:
    """Search contract used by UI and personalization modules.

    Baseline retrieval is TF-IDF + cosine similarity. A second-stage rerank hook
    is applied after baseline scoring so personalization/context logic can evolve
    without modifying the core retrieval math.
    """
    tokens = tokenize(query)
    if not tokens:
        return []
    baseline_scores = _score_tokens(index, tokens)
    # Backward-compatible bridge: legacy arguments are translated into user_prefs.
    merged_prefs = dict(user_prefs or {})
    if preferred_genres:
        merged_prefs["preferred_genres"] = preferred_genres
    if preferred_authors:
        merged_prefs["preferred_authors"] = preferred_authors
    if preferred_moods:
        merged_prefs["preferred_moods"] = preferred_moods
    if preferred_time:
        merged_prefs["preferred_time"] = preferred_time

    reranked_scores = apply_rerank(
        index,
        baseline_scores,
        query_tokens=tokens,
        user_prefs=merged_prefs,
        context=context,
        genre_boost=genre_boost,
        author_boost=author_boost,
        mood_boost=mood_boost,
        time_boost=time_boost,
    )
    ranked = sorted(reranked_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results: list[SearchResult] = []
    for doc_id, score in ranked:
        matched_terms = _matched_query_terms(index, doc_id, tokens)
        why_matched = _build_why_matched(matched_terms, user_prefs=merged_prefs, context=context)
        results.append(
            SearchResult(
                book_id=doc_id,
                score=score,
                display=index.doc_display.get(doc_id, {}),
                record=index.doc_records.get(doc_id),
                matched_terms=matched_terms,
                why_matched=why_matched,
            )
        )
    return results


def apply_rerank(
    index: InvertedIndex,
    scores: dict[str, float],
    *,
    query_tokens: list[str],
    user_prefs: dict[str, Any] | None,
    context: dict[str, Any] | None,
    genre_boost: float = 0.2,
    author_boost: float = 0.3,
    mood_boost: float = 0.2,
    time_boost: float = 0.2,
) -> dict[str, float]:
    """Second-stage ranking hook for Person C integration.

    Baseline relevance comes from TF-IDF. This layer applies user preference and
    context boosts (genre, author, mood, time) without changing retrieval math.
    """
    _ = (query_tokens, context)
    reranked = dict(scores)
    if not user_prefs:
        return reranked

    preferred_genres = _normalize_pref_list(user_prefs.get("preferred_genres"))
    preferred_authors = _normalize_pref_list(user_prefs.get("preferred_authors"))
    preferred_moods = _normalize_pref_list(user_prefs.get("preferred_moods"))
    preferred_time = _normalize_pref_value(
        user_prefs.get("preferred_time") or (context or {}).get("reading_time")
    )

    for doc_id in reranked:
        display = index.doc_display.get(doc_id, {})
        if preferred_genres:
            categories = _normalize_pref_list(display.get("categories", ""))
            if any(genre in categories for genre in preferred_genres):
                reranked[doc_id] += genre_boost

        if preferred_authors:
            authors = _normalize_pref_list(display.get("authors", ""))
            if any(author in authors for author in preferred_authors):
                reranked[doc_id] += author_boost

        if preferred_moods and _matches_mood(display, preferred_moods):
            reranked[doc_id] += mood_boost

        if preferred_time and _matches_time_pref(display, preferred_time):
            reranked[doc_id] += time_boost
    return reranked


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


def _matched_query_terms(index: InvertedIndex, doc_id: str, query_tokens: list[str]) -> list[str]:
    term_freqs = index.doc_term_freqs.get(doc_id, {})
    unique_query_terms = list(dict.fromkeys(query_tokens))
    return [term for term in unique_query_terms if term in term_freqs]


def _build_why_matched(
    matched_terms: list[str],
    *,
    user_prefs: dict[str, Any] | None,
    context: dict[str, Any] | None,
) -> str:
    if not matched_terms:
        base = "Matched overall query semantics."
    else:
        base = "Matched query terms: " + ", ".join(matched_terms[:5]) + "."

    # Keep explanation contract stable even before reranking is implemented.
    if user_prefs or context:
        return base + " Personalization/context rerank hook available."
    return base


def _normalize_pref_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        source = value.replace(";", ",").split(",")
    elif isinstance(value, list):
        source = value
    else:
        return []
    return [str(item).strip().lower() for item in source if str(item).strip()]


def _normalize_pref_value(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    aliases = {
        "short": "short",
        "quick": "short",
        "low": "short",
        "medium": "medium",
        "mid": "medium",
        "normal": "medium",
        "long": "long",
        "deep": "long",
        "high": "long",
    }
    return aliases.get(normalized, normalized)


def _matches_time_pref(display: dict[str, str], preferred_time: str) -> bool:
    num_pages = _parse_num_pages(display.get("num_pages", ""))
    if num_pages is None:
        return False
    if preferred_time == "short":
        return num_pages <= 200
    if preferred_time == "medium":
        return 200 < num_pages <= 400
    if preferred_time == "long":
        return num_pages > 400
    if preferred_time.isdigit():
        # Optional minutes input: approximate 1 page/minute.
        minutes = int(preferred_time)
        return num_pages <= minutes
    return False


def _parse_num_pages(value: str) -> int | None:
    if not value:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _matches_mood(display: dict[str, str], preferred_moods: list[str]) -> bool:
    searchable_text = " ".join(
        [
            str(display.get("categories", "")),
            str(display.get("subjects", "")),
            str(display.get("bookshelves", "")),
            str(display.get("description", "")),
            str(display.get("title", "")),
        ]
    ).lower()
    if not searchable_text.strip():
        return False

    for mood in preferred_moods:
        mood_keywords = _MOOD_KEYWORDS.get(mood, [mood])
        if any(keyword in searchable_text for keyword in mood_keywords):
            return True
    return False


_MOOD_KEYWORDS: dict[str, list[str]] = {
    "relaxing": ["relaxing", "calm", "poetry", "classic", "nature"],
    "fun": ["fun", "humor", "comedy", "adventure", "fantasy"],
    "learning": ["history", "science", "biography", "philosophy", "education"],
    "serious": ["war", "politics", "law", "ethics", "society"],
}
