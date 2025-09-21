import os
import json
import joblib
import numpy as np
import re
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

        # Make the system prefer passages from source file that have overlapping tokens with the query
        query_tokens = [t for t in (query.lower().replace("(", " ").replace(")", " ").split()) if len(t) >= 3]
        if query_tokens:
            boost = []
            for d in self.docs:
                source = d["source"].lower()
                found = any(t in source for t in query_tokens)
                boost.append(1.25 if found else 1.00)
            boost = np.array(boost)
            scores = scores * boost

        query = (query or "").lower()

        # Pick a year if one is mentioned in the query
        year = re.search(r"\b(19|20)\d{2}\b", query)
        query_year = year.group(0) if year else None

        # Extract main author name from the retrieved source
        def author_surname(source: str) -> str:
            s = (source or "").lower()
            s = s.split(" - ")[0]                 
            if "et al." in s:                     
                s = s.split("et al.")[0].strip()
            parts = s.split()
            return parts[-1] if parts else ""

        # Create a set of surnames present across our index and see which one is mentioned in the query
        surnames = { author_surname(d.get("source", "")) for d in self.docs if d.get("source") }
        mentioned = { name for name in surnames if name and name in query }

        if mentioned or query_year:
            source_boost = []
            for d in self.docs:
                source = (d.get("source") or "").lower()
                author = author_surname(source)
                boost = 1.0
                if author in mentioned:
                    boost *= 1.6
                if query_year and query_year in source:
                    boost *= 1.2
                source_boost.append(boost)
            scores = scores * np.array(source_boost)

        # Choose top-k by score
        top_index = np.argsort(scores)[::-1][:k]

        # Show retrieved result
        results = []
        for i in top_index:
            d = self.docs[i]
            results.append({
                "score": float(scores[i]),
                "source": d["source"],
                "page": d["page"],
                "type": d["type"],
                "text": d["text"],
            })
        return results

    