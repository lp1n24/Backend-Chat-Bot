import os
import json
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Paths for the artifacts created by index.py
index_dir = os.path.join("data", "index")
tfidf_path = os.path.join(index_dir, "tfidf_vectorizer.joblib")
matrix_path = os.path.join(index_dir, "tfidf_matrix.joblib")
docs_path = os.path.join(index_dir, "docs.json")

class Retriever:
    # Load the artifacts once and reuse later to avoid recomputing TF-IDF every query
    def __init__(self):
        self.vectorizer = joblib.load(tfidf_path)
        self.matrix = joblib.load(matrix_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

    # Return top-k matches for a query
    def search(self, query, k=5):

        # Vectorize query (1 x n_terms) to make sure query and indexed documents are in the same TF-IDF space
        query_vec = self.vectorizer.transform([query])

        # Apply cosine similarity to every document (return 1 x n_docs) to measure how close the query and indexed documents are
        scores = cosine_similarity(query_vec, self.matrix)[0]

        # Choose top-k by score
        top_index = np.argsort(scores)[::-1][:k]

        # Build simple result objects to show score and citations to the document used for transparency
        results = []
        for i in top_index:
            d = self.docs[i]
            results.append({
                "score": float(scores[i]),
                "source": d["source"],
                "page": d["page"],
                "type": d["type"],
                "text": d["text"]
            })
        return results