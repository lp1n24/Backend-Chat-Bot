from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.search import Retriever

# The object that represents the web service
app = FastAPI(title="Chat-With-PDFs", version="0.2.0", description="integrated search into FastAPI")

# Load the retriever class once and reuse later to avoid recomputing TF-IDF every query
retriever = None
def get_retriever():
    global retriever
    if retriever is None:
        try:
            retriever = Retriever()
        except Exception as e:
            # In case we forgot to create artifacts
            raise HTTPException(status_code=500, detail=f"Index not ready. Please run 'python src\\index.py'. Error: {e}")
    return retriever

# This class define a shape for when client sends user question and initialize session_id
class Ask(BaseModel):
    session_id: str
    question: str

# This class helps clear the session using session_id
class Clear(BaseModel):
    session_id: str

# For checking server status
@app.get("/status")
def status():
    return {"status": "ok"}

# For asking question (we will implement RAG + LLM later in the next version)
@app.post("/ask")
def ask(body: Ask):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    
    retriever = get_retriever()
    retrieve = retriever.search(body.question, k=5) # Retrieve top-5 chunks based on the score

    # Keep the response simple and transparent
    return {
        "session_id": body.session_id,
        "question": body.question,
        "answer": [
            {
                "snippet": r["text"][:500],
                "source": r["source"],
                "page": r["page"],
                "type": r["type"],
                "score": round(r["score"], 4)
            }
            for r in retrieve
        ]
    }

# For showing that the session memory has been cleared
@app.post("/clear")
def clear(body: Clear):
    return {"cleared": True}