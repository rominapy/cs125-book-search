from __future__ import annotations

"""PG catalog ingestion and logical view creation.

Logical view:
- Searchable tokens: title, author, subjects, bookshelves.
- Display fields: title, authors, subjects, bookshelves, language.
- Filter fields: language (English by default).
"""

import csv
import os
import re
from dataclasses import dataclass
from typing import Iterator, Optional

from src.utils.text import normalize_text, tokenize


_DEFAULT_CANDIDATES = {
    "id": ["id", "book_id", "text#", "text_id", "pg_id", "gutenberg_id"],
    "title": ["title"],
    "author": ["author", "authors"],
    "subjects": ["subjects", "subject", "subject_headings"],
    "bookshelves": ["bookshelves", "bookshelf", "shelves"],
    "language": ["language", "lang"],
}


@dataclass(frozen=True)
class BookRecord:
    book_id: str
    title: str
    authors: list[str]
    subjects: list[str]
    bookshelves: list[str]
    language: str
    display: dict[str, str]
    tokens: dict[str, list[str]]

    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id,
            "title": self.title,
            "authors": self.authors,
            "subjects": self.subjects,
            "bookshelves": self.bookshelves,
            "language": self.language,
            "display": dict(self.display),
            "tokens": {key: list(value) for key, value in self.tokens.items()},
        }


def find_pg_catalog(root: str) -> Optional[str]:
    if os.path.isfile(root) and root.endswith(".csv"):
        return root
    if not os.path.isdir(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower() == "pg_catalog.csv":
                return os.path.join(dirpath, filename)
    return None


def load_pg_catalog(
    csv_path: str,
    *,
    filter_english: bool = True,
    min_title_length: int = 1,
) -> list[BookRecord]:
    records = []
    for row in iter_pg_catalog_rows(csv_path):
        logical = build_logical_view(
            row,
            filter_english=filter_english,
            min_title_length=min_title_length,
        )
        if logical is not None:
            records.append(logical)
    return records


def iter_pg_catalog_rows(csv_path: str) -> Iterator[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key.strip(): (value or "").strip() for key, value in row.items()}


def build_logical_view(
    row: dict[str, str],
    *,
    filter_english: bool = True,
    min_title_length: int = 1,
) -> Optional[BookRecord]:
    fields = _extract_fields(row)
    title = fields["title"]
    book_id = fields["id"]

    if not title or len(title) < min_title_length:
        return None
    if not book_id:
        return None

    language = fields["language"]
    if filter_english and not _is_english(language):
        return None

    authors = _split_multi(fields["author"])
    subjects = _split_multi(fields["subjects"])
    bookshelves = _split_multi(fields["bookshelves"])

    display = {
        "title": title,
        "authors": "; ".join(authors),
        "subjects": "; ".join(subjects),
        "bookshelves": "; ".join(bookshelves),
        "language": language,
    }

    tokens = {
        "title": tokenize(title),
        "author": tokenize(" ".join(authors)),
        "subjects": tokenize(" ".join(subjects)),
        "bookshelves": tokenize(" ".join(bookshelves)),
    }

    return BookRecord(
        book_id=str(book_id),
        title=title,
        authors=authors,
        subjects=subjects,
        bookshelves=bookshelves,
        language=language,
        display=display,
        tokens=tokens,
    )


def _extract_fields(row: dict[str, str]) -> dict[str, str]:
    normalized = {key.strip().lower(): value for key, value in row.items()}

    def pick(field: str) -> str:
        for candidate in _DEFAULT_CANDIDATES[field]:
            if candidate in normalized and normalized[candidate]:
                return normalized[candidate]
        return ""

    return {
        "id": pick("id"),
        "title": pick("title"),
        "author": pick("author"),
        "subjects": pick("subjects"),
        "bookshelves": pick("bookshelves"),
        "language": pick("language"),
    }


def _split_multi(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;|]", value)
    cleaned = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        for sub in part.split("--"):
            sub = sub.strip()
            if sub:
                cleaned.append(sub)
    return cleaned


def _is_english(language: str) -> bool:
    if not language:
        return False
    normalized = normalize_text(language)
    return normalized in {"en", "eng", "english"}
