from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.utils.text import tokenize


@dataclass
class InvertedIndex:
    postings: dict[str, dict[str, int]] = field(default_factory=dict)
    doc_freqs: dict[str, int] = field(default_factory=dict)
    doc_term_freqs: dict[str, dict[str, int]] = field(default_factory=dict)
    doc_lengths: dict[str, int] = field(default_factory=dict)
    doc_norms: dict[str, float] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    doc_display: dict[str, dict[str, str]] = field(default_factory=dict)
    doc_records: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        records: Iterable[Any],
        *,
        token_fields: Iterable[str] | None = None,
    ) -> "InvertedIndex":
        index = cls()
        for record in records:
            doc_id = _get_doc_id(record)
            if not doc_id:
                continue
            tokens = _get_tokens(record, token_fields=token_fields)
            if not tokens:
                continue
            term_freq = Counter(tokens)
            index.doc_term_freqs[doc_id] = dict(term_freq)
            index.doc_lengths[doc_id] = sum(term_freq.values())
            index.doc_records[doc_id] = record
            display = _get_display(record)
            if display:
                index.doc_display[doc_id] = display
            for term, freq in term_freq.items():
                postings = index.postings.setdefault(term, {})
                postings[doc_id] = freq
        index._finalize()
        return index

    def _finalize(self) -> None:
        doc_count = len(self.doc_term_freqs)
        self.doc_freqs = {term: len(postings) for term, postings in self.postings.items()}
        self.idf = {
            term: math.log((doc_count + 1) / (df + 1)) + 1
            for term, df in self.doc_freqs.items()
        }
        self.doc_norms = {}
        for doc_id, term_freqs in self.doc_term_freqs.items():
            norm = 0.0
            for term, tf in term_freqs.items():
                tf_weight = 1.0 + math.log(tf)
                tfidf = tf_weight * self.idf.get(term, 0.0)
                norm += tfidf * tfidf
            self.doc_norms[doc_id] = math.sqrt(norm) if norm > 0 else 0.0


def _get_doc_id(record: Any) -> str:
    if isinstance(record, dict):
        return str(record.get("book_id") or record.get("id") or "")
    return str(getattr(record, "book_id", "") or getattr(record, "id", "") or "")


def _get_tokens(record: Any, *, token_fields: Iterable[str] | None) -> list[str]:
    tokens: list[str] = []
    source = None
    if isinstance(record, dict):
        source = record.get("tokens")
    else:
        source = getattr(record, "tokens", None)
    if isinstance(source, dict):
        fields = token_fields or source.keys()
        for field in fields:
            field_tokens = source.get(field, [])
            tokens.extend([token for token in field_tokens if token])
        return tokens

    if isinstance(record, dict):
        text = " ".join(str(record.get(key, "")) for key in _fallback_fields())
    else:
        text = " ".join(str(getattr(record, key, "")) for key in _fallback_fields())
    return tokenize(text)


def _get_display(record: Any) -> dict[str, str]:
    if isinstance(record, dict):
        display = record.get("display")
        if isinstance(display, dict):
            return {str(k): str(v) for k, v in display.items()}
        return {}
    display = getattr(record, "display", None)
    if isinstance(display, dict):
        return {str(k): str(v) for k, v in display.items()}
    return {}


def _fallback_fields() -> list[str]:
    return ["title", "subtitle", "authors", "author", "subjects", "bookshelves", "categories"]
