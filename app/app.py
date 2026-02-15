from flask import Flask, render_template, request
from src.index import InvertedIndex
from src.ingest import find_pg_catalog, load_pg_catalog
from src.ingest.kaggle_books import find_books_csv, load_books_csv
from src.rank import search

dataset_root = '.'

pg_path = find_pg_catalog(dataset_root)
kaggle_path = find_books_csv(dataset_root)

records = []
if pg_path:
    records = load_pg_catalog(pg_path, filter_english=True)
    dataset_name = 'pg_catalog'
elif kaggle_path:
    records = load_books_csv(kaggle_path)
    dataset_name = 'kaggle_books'
else:
    dataset_name = None

index = InvertedIndex.build(records)

def personalize(results, preferred_genres):
    boosted = []

    preferred_genres = [g.strip().lower() for g in preferred_genres]
    for r in results:
        score = r.score

        book_genres = [g.strip().lower() for g in r.display.get("categories", "").split(",") if g]

        score = r.score
        if any(g in book_genres for g in preferred_genres):
            score += 0.2

        boosted.append({
            "book_id": r.book_id,
            "title": r.display.get("title", ""),
            "authors": r.display.get("authors", ""),
            "genres": book_genres,
            "score": score
        })

    return sorted(boosted, key=lambda x: x["score"], reverse=True)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    if request.method == "POST":
        # query = request.form["query"]
        # results = search(index, query, top_k=10)
        # print([(r.book_id, r.display.get('title', ''), r.display.get('authors', ''), r.score) for r in results])

        query = request.form.get("query", "")
        genres = request.form.getlist("genres") 

        raw_results = search(index, query, top_k = 10)
        # print(raw_results[0].display)

        results = personalize(raw_results, genres)

    return render_template("index.html", results=results)

app.run(debug=True)
