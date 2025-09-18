from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.search import Retriever
from src.memory import SessionMemory
from src.router import route

# The object that represents the web service
app = FastAPI(title="Chat-With-PDFs", version="0.2.1", description="integrated memory and router into FastAPI")

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

# in-memory session history
memory = SessionMemory(max_turns=20)

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

# For asking question (store user turn -> route -> respond -> store agent turn)
@app.post("/ask")
def ask(body: Ask):
    query = (body.question or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question must not be empty")
    
    # Store user turn
    memory.add_user(body.session_id, query)

    # Fetch history for routing context
    history = memory.get(body.session_id)
    
    # Plan action using scores got from router
    retrieve = get_retriever()
    decision = route(question=query, retriever=retrieve, history=history, k=5)
    action = decision.get("action")

    if action == "retrieve_pdfs":
        found = decision.get("found", [])
        answer = []
        for f in found:
            answer.append({
                "snippet": f["text"][:500],
                "source": f["source"],
                "page": f["page"],
                "type": f["type"],
                "score": round(float(f["score"]), 4),
            })
        agent_text = f"Found {len(answer)} passages in PDFs (context={decision.get('used_context', 'raw')})."
        memory.add_agent(body.session_id, agent_text)

        return {
            "session_id": body.session_id,
            "question": query,
            "router": {
                "action": action,
                "used_context": decision.get("used_context", "raw"),
                "reason": decision.get("reason", ""),
                "scores": decision.get("scores", []),
            },
            "answer": answer,
            "history": memory.get(body.session_id),
        }
    
    if action == "clarify":
        message = decision.get("message", "Could you be more specific?")
        memory.add_agent(body.session_id, message)
        return {
            "session_id": body.session_id,
            "question": query,
            "router": {
                "action": action,
                "used_context": decision.get("used_context", "raw"),
                "reason": decision.get("reason", "ambiguous"),
                "scores": decision.get("scores", []),
            },
            "message": message,
            "history": memory.get(body.session_id),
        }
    
    if action == "web_search":
        message = "This looks outside the provided PDFs"
        memory.add_agent(body.session_id, message)
        return {
            "session_id": body.session_id,
            "question": query,
            "router": {
                "action": action,
                "used_context": decision.get("used_context", "raw"),
                "reason": decision.get("reason", "low similarity to the PDFs"),
                "scores": decision.get("scores", []),
                "query": decision.get("query", query),
            },
            "message": message,
            "history": memory.get(body.session_id),
        }       

# For showing that the session memory has been cleared
@app.post("/clear")
def clear(body: Clear):
    memory.clear(body.session_id)
    return {"cleared": True}