
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

app = Flask(__name__)

# ==========================
# LOAD DATA & EMBEDDINGS
# ==========================

movie_df = pd.read_csv("movie_df.csv")
movie_embeddings = np.load("movie_embeddings.npy")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Movie data & embeddings loaded:", movie_embeddings.shape)

# ==========================
# SEARCH FUNCTION
# ==========================

MIN_SCORE = 0.35  # below this, a match is too weak to trust

def recommend_movies(query, top_n=5, min_score=MIN_SCORE):
    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, movie_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            continue  # skip weak matches instead of showing them as confident results

        row = movie_df.iloc[idx]
        results.append({
            "title": row['Title'],
            "year": int(row['Year']),
            "genre": row['Genre'],
            "description": row['Description'],
            "rating": row['Rating'],
            "score": round(score, 3)
        })
    return results

# ==========================
# ROUTES
# ==========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")
    results = recommend_movies(query, top_n=5)
    return jsonify({
        "results": results,
        "message": None if results else "No strong matches found in this dataset for that request."
    })

# ==========================
import os

if _name_ == "_main_":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
