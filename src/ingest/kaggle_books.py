from __future__ import annotations

"""Kaggle 7k books dataset ingestion and logical view creation.

Logical view:
- Searchable tokens: title, subtitle, authors, categories, description.
- Display fields: title, subtitle, authors, categories, description, thumbnail,
  published_year, average_rating, ratings_count, num_pages, language.
- Filter fields: language, published_year, categories.
"""

import csv
import os
import re
from dataclasses import dataclass
from typing import Iterator, Optional

from src.utils.text import tokenize


_DEFAULT_CANDIDATES = {
    "id": [
        "id",
        "book_id",
        "bookid",
        "isbn13",
        "isbn_13",
        "isbn10",
        "isbn_10",
    ],
    "title": ["title", "book_title", "name"],
    "subtitle": ["subtitle", "sub_title"],
    "title_and_subtitle": ["title_and_subtitle"],
    "authors": ["authors", "author", "author_name"],
    "description": ["description", "book_description", "summary", "tagged_description"],
    "categories": [
        "categories",
        "category",
        "subjects",
        "subject",
        "genres",
        "simple_categories",
    ],
    "language": ["language", "language_code", "lang"],
    "published_year": ["published_year", "publication_year", "year"],
    "average_rating": ["average_rating", "avg_rating", "rating"],
    "ratings_count": ["ratings_count", "num_ratings", "ratings"],
    "num_pages": ["num_pages", "page_count", "pages"],
    "thumbnail": ["thumbnail", "image_url", "cover_image", "cover_url"],
}


@dataclass(frozen=True)
class KaggleBookRecord:
    book_id: str
    title: str
    subtitle: str
    authors: list[str]
    categories: list[str]
    description: str
    language: str
    published_year: str
    average_rating: str
    ratings_count: str
    num_pages: str
    thumbnail: str
    display: dict[str, str]
    tokens: dict[str, list[str]]
    filters: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "authors": list(self.authors),
            "categories": list(self.categories),
            "description": self.description,
            "language": self.language,
            "published_year": self.published_year,
            "average_rating": self.average_rating,
            "ratings_count": self.ratings_count,
            "num_pages": self.num_pages,
            "thumbnail": self.thumbnail,
            "display": dict(self.display),
            "tokens": {key: list(value) for key, value in self.tokens.items()},
            "filters": dict(self.filters),
        }


def find_books_csv(root: str) -> Optional[str]:
    if os.path.isfile(root) and root.endswith(".csv"):
        return root
    if not os.path.isdir(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower() in {"books.csv", "book_dataset.csv"}:
                return os.path.join(dirpath, filename)
    return None


def load_books_csv(csv_path: str, *, min_title_length: int = 1) -> list[KaggleBookRecord]:
    records = []
    for row in iter_books_rows(csv_path):
        logical = build_logical_view(row, min_title_length=min_title_length)
        if logical is not None:
            records.append(logical)
    return records


def iter_books_rows(csv_path: str) -> Iterator[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key.strip(): (value or "").strip() for key, value in row.items()}


def build_logical_view(
    row: dict[str, str],
    *,
    min_title_length: int = 1,
) -> Optional[KaggleBookRecord]:
    fields = _extract_fields(row)
    title, subtitle = _derive_title(fields)
    book_id = fields["id"] or _fallback_id(title, fields["authors"])

    if not title or len(title) < min_title_length:
        return None

    authors = _split_multi(fields["authors"])
    categories = _split_multi(fields["categories"])
    description = fields["description"]

    display = {
        "title": title,
        "subtitle": subtitle,
        "authors": "; ".join(authors),
        "categories": "; ".join(categories),
        "description": description,
        "thumbnail": fields["thumbnail"],
        "published_year": fields["published_year"],
        "average_rating": fields["average_rating"],
        "ratings_count": fields["ratings_count"],
        "num_pages": fields["num_pages"],
        "language": fields["language"],
    }

    tokens = {
        "title": tokenize(" ".join([title, subtitle]).strip()),
        "authors": tokenize(" ".join(authors)),
        "categories": tokenize(" ".join(categories)),
        "description": tokenize(description),
    }

    filters = {
        "language": fields["language"],
        "published_year": fields["published_year"],
        "categories": "; ".join(categories),
    }

    return KaggleBookRecord(
        book_id=book_id,
        title=title,
        subtitle=subtitle,
        authors=authors,
        categories=categories,
        description=description,
        language=fields["language"],
        published_year=fields["published_year"],
        average_rating=fields["average_rating"],
        ratings_count=fields["ratings_count"],
        num_pages=fields["num_pages"],
        thumbnail=fields["thumbnail"],
        display=display,
        tokens=tokens,
        filters=filters,
    )


def _extract_fields(row: dict[str, str]) -> dict[str, str]:
    normalized = {key.strip().lower(): value for key, value in row.items()}

    def pick(field: str) -> str:
        for candidate in _DEFAULT_CANDIDATES[field]:
            if candidate in normalized and normalized[candidate]:
                return normalized[candidate]
        return ""

    return {key: pick(key) for key in _DEFAULT_CANDIDATES}


def _derive_title(fields: dict[str, str]) -> tuple[str, str]:
    title = fields["title"]
    subtitle = fields["subtitle"]
    if title:
        return title, subtitle
    title_and_subtitle = fields["title_and_subtitle"]
    if not title_and_subtitle:
        return "", subtitle
    parts = [part.strip() for part in title_and_subtitle.split(":", 1)]
    title = parts[0] if parts else ""
    if len(parts) > 1 and not subtitle:
        subtitle = parts[1]
    return title, subtitle


def _fallback_id(title: str, authors_value: str) -> str:
    if not title and not authors_value:
        return ""
    base = re.sub(r"\s+", "-", f"{title}-{authors_value}".strip())
    base = re.sub(r"[^a-zA-Z0-9-]", "", base)
    return base.lower()


def _split_multi(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;|]", value)
    cleaned = []
    for part in parts:
        for sub in part.split(","):
            sub = sub.strip()
            if sub:
                cleaned.append(sub)
    return cleaned
