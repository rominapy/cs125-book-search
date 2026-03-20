from flask import Flask, render_template, request, redirect, url_for
from src.index import InvertedIndex
from src.ingest import find_pg_catalog, load_pg_catalog
from src.ingest.kaggle_books import find_books_csv, load_books_csv
from src.rank import search
from flask import session


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

app = Flask(__name__)

app.secret_key = "secret"

def get_recommendations(index, bookmarks, k=10):
    if not bookmarks:
        return []

    genres = []
    authors = []
    bookmarked_ids = {b["book_id"] for b in bookmarks}

    for b in bookmarks:
        if b.get("categories"):
            genres.extend(b["categories"].split(","))
        if b.get("authors"):
            authors.extend(b["authors"].split(","))

    preferred_genres = [g.strip().lower() for g in genres if g]
    preferred_authors = [a.strip().lower() for a in authors if a]

    # Use top genres as query or fallback
    query = " ".join(preferred_genres[:3]) or "book"

    # Run search with personalization
    results = search(
        index,
        query,
        top_k=k*2,  # fetch extra to allow filtering
        preferred_genres=preferred_genres,
        preferred_authors=preferred_authors
    )

    # Filter out already bookmarked books
    filtered_results = [r for r in results if r.book_id not in bookmarked_ids]

    return filtered_results[:k]

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    curr_query = ""

    bookmarks = session.get("bookmarks", [])
    recommended = get_recommendations(index, bookmarks, k=5)


    if request.method == "POST":
        curr_query = request.form.get("query", "")
    
        # preferences 
        genres_input = request.form.get("genres", "")
        authors_input = request.form.get("authors", "")
        mood_input = request.form.get("moods", "")
        time_input = request.form.get("time", "")

        if genres_input:
            session["genres"] = genres_input
        if authors_input:
            session["authors"] = authors_input
        if mood_input:
            session["moods"] = mood_input
        if time_input:
            session["time"] = time_input

        # context (NOT stored long term)

        preferred_genres_list = [
            g.strip().lower()
            for g in session.get("genres", "").split(",") if g
        ]

        preferred_authors_list = [
            a.strip().lower()
            for a in session.get("authors", "").split(",") if a
        ]

        preferred_moods_list = [
            m.strip().lower()
            for m in session.get("moods", "").split(",") if m
        ]
        

        results = search(
            index,
            curr_query,
            preferred_genres=preferred_genres_list,
            preferred_authors=preferred_authors_list,
            preferred_moods=preferred_moods_list,
            preferred_time=session.get("time","")
        )
        # print(results)

    return render_template(
        "index.html",
        results=results,
        recommended=recommended,
        current_query=curr_query,
        current_moods=session.get("moods", ""),
        current_time=session.get("time", "")
    )



@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        genres = request.form.get("genres", "")
        authors = request.form.get("authors", "")

        if genres:
            session["genres"] = genres
        if authors:
            session["authors"] = authors

    return render_template(
        "profile.html",
        current_genres=session.get("genres", ""),
        current_authors=session.get("authors", ""),
        bookmarks=session.get("bookmarks", [])
    )

@app.route("/bookmark", methods=["POST"])
def bookmark():
    bookmarks = session.get("bookmarks", [])

    book = {
        "book_id": request.form.get("book_id"),
        "title": request.form.get("title"),
        "authors": request.form.get("authors"),
        "categories": request.form.get("categories"),
    }

    if "bookmarks" not in session:
        session["bookmarks"] = []

    bookmarks = session["bookmarks"]

    # prevent duplicates
    if not any(b["book_id"] == book["book_id"] for b in bookmarks):
        bookmarks.append(book)
        session["bookmarks"] = bookmarks

    return redirect(url_for("home"))

@app.route("/remove_bookmark", methods=["POST"])
def remove_bookmark():
    book_id = request.form.get("book_id")

    bookmarks = session.get("bookmarks", [])

    # remove matching book
    bookmarks = [b for b in bookmarks if b["book_id"] != book_id]

    session["bookmarks"] = bookmarks

    return redirect("/profile") 

app.run(debug=True)