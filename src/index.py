import os
import csv
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Use input data from the CSV file produced by ingest.py
csv_path = os.path.join("data", "processed", "pages.csv")

index_dir = os.path.join("data", "index")
tfidf_path = os.path.join(index_dir, "tfidf_vectorizer.joblib")
matrix_path = os.path.join(index_dir, "tfidf_matrix.joblib")
docs_path = os.path.join(index_dir, "docs.json")

# Read rows from CSV and turn them into a list of dicts
def load_rows(csv_path):
    docs = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # This keep schema consistent with what we have in ingest.py
            docs.append({
                "source": r.get("source",""),
                "page": int(r.get("page", 0) or 0),
                "type": r.get("type", "page"),
                "text": r.get("text", "")
                })
    return docs

def main():
    os.makedirs(index_dir, exist_ok=True)

    docs = load_rows(csv_path)

    # We tried with caption before but got worse result, so we will go with page only
    docs = [d for d in docs if d.get("type") == "page"]

    texts = [d["text"] for d in docs]

    # Build a simple TF-IDF that drop common english words and ignore words that appear in >85% of docs
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.85, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)

    # Save artifacts for fast loading at query time
    joblib.dump(vectorizer, tfidf_path)
    joblib.dump(matrix, matrix_path)
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    # Short summary for easier validation
    print(f"docs: {len(docs)}, matrix shape: {matrix.shape}")
    print(f"saved: {tfidf_path}")
    print(f"saved: {matrix_path}")
    print(f"saved: {docs_path}")

if __name__ == "__main__":
    main()

