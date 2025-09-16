from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# The object that represents the web service
app = FastAPI(title="Chat-With-PDFs", version="0.1.2", description="index builder and search class implemented")

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
    raise HTTPException(status_code=501) # Using 501 just as a placeholder for testing for now

# For showing that the session memory has been cleared
@app.post("/clear")
def clear(body: Clear):
    return {"cleared": True}