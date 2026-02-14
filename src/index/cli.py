from __future__ import annotations

import argparse
import sys

from src.index import InvertedIndex
from src.rank import search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build index and run TF-IDF search.")
    parser.add_argument(
        "--dataset",
        choices=["pg", "kaggle"],
        default="pg",
        help="Dataset type to ingest.",
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Path to dataset root or CSV file.",
    )
    parser.add_argument("--query", default="", help="Query string.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results.")
    args = parser.parse_args(argv)

    if args.dataset == "pg":
        from src.ingest import find_pg_catalog, load_pg_catalog

        csv_path = find_pg_catalog(args.path)
        if not csv_path:
            print("pg_catalog.csv not found. Provide --path to the CSV or its folder.")
            return 1
        records = load_pg_catalog(csv_path, filter_english=True)
    else:
        from src.ingest.kaggle_books import find_books_csv, load_books_csv

        csv_path = find_books_csv(args.path)
        if not csv_path:
            print("Books CSV not found. Provide --path to the CSV or its folder.")
            return 1
        records = load_books_csv(csv_path)

    index = InvertedIndex.build(records)
    if not args.query:
        print("Index built. Provide --query to search.")
        return 0

    results = search(index, args.query, top_k=args.top_k)
    if not results:
        print("No results found.")
        return 0

    for rank, result in enumerate(results, start=1):
        title = result.display.get("title") or ""
        authors = result.display.get("authors") or ""
        print(f"{rank}. {title} | {authors} | score={result.score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
