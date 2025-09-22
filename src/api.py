import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import run_graph
from src.memory import add_user, clear as clear_session
from src.log import setup as setup_logging

setup_logging()
logger = logging.getLogger("api")

# The object that represents the web service
app = FastAPI(title="Backend-Chat-Bot", version="0.4.0", description="Completed prototype")

# Request bodies
class Ask(BaseModel):
    session_id: str
    question: str

class Clear(BaseModel):
    session_id: str

# For checking API status
@app.get("/status")
def status():
    return {"status": "ok"}

# For posting question (main function)
@app.post("/ask")
def ask(body: Ask):
    query = (body.question or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    # Store user turn in session memory
    add_user(body.session_id, query)

    logger.info("ASK start session=%s", body.session_id)

    try:
        # Run the LangGraph (router -> pdf/clarify/web agent -> answer)
        payload = run_graph(body.session_id, query)
    except Exception:
        logger.exception("ASK failed session=%s", body.session_id)
        raise HTTPException(status_code=500, detail="Internal Error")
    
    # Logging summary
    router = payload.get("router", {})
    answer = payload.get("answer", {})
    logger.info("ASK done session=%s action=%s mode=%s", body.session_id, router.get("action", ""), answer.get("mode", ""))

    # Return the response that with predefined shape (router, answer, history)
    return payload

@app.post("/clear")
def clear(body: Clear):
    clear_session(body.session_id)
    logger.info("session cleared session=%s", body.session_id)
    return {"cleared": True}