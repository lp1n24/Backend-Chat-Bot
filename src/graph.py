import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.memory import add_agent, get as get_history
from src.router import route
from src.search import Retriever
from src.llm import llm_answer, llm_clarify, llm_web_answer
from src.log import setup as setup_logging

setup_logging()
logger = logging.getLogger("graph")


# This is the router agent. It decides the next action using the retriever + history
def router_node(state):
    session_id = state["session_id"]
    question = state["question"]
    logger.info(f"[Router Node] Session={session_id}, Question={question}")
    retriever = Retriever()
    decision = route(question, retriever=retriever, history=get_history(session_id))
    logger.info(f"[Router Node] Decision={decision}")
    state["decision"] = decision
    return state

# This is the PDF agent. It retrieves passages from PDFs and composes an answer with citations
def pdf_node(state):
    session_id = state["session_id"]
    question = state["question"]
    logger.info(f"[PDF Node] Session={session_id}, Question={question}")
    retriever = Retriever()
    found = retriever.search(question, k=5)
    logger.info(f"[PDF Node] Retrieved {len(found)} passages")
    state["found"] = found

    result = llm_answer(question, found)
    text = (result or {}).get("answer", "")

    add_agent(session_id, text)
    state["answer"] = {
        "text": text,
        "citations": (result or {}).get("citations", []),
        "mode": (result or {}).get("mode", "llm"),
        "evidence": (result or {}).get("evidence", []),
    }
    state["history"] = get_history(session_id)
    logger.info(f"[PDF Node] Retrieved {len(found)} passages")
    return state

# This is the clarify agent. It generates a clarification/follow-up question when the user query is vague
def clarify_node(state):
    session_id = state["session_id"]
    question = state["question"]
    logger.info(f"[Clarify Node] Session={session_id}, Question={question}")

    result = llm_clarify(question)
    text = (result or {}).get("answer", "")

    add_agent(session_id, text)
    state["answer"] = {
        "text": text,
        "citations": [],
        "mode": (result or {}).get("mode", "llm"),
    }
    state["history"] = get_history(session_id)
    logger.info(f"[Clarify Node] Clarification={text}")
    return state

# This is the web agent. It performs web search and composes an answer from external sources
def web_node(state):
    session_id = state["session_id"]
    question = state["question"]
    logger.info(f"[Web Node] Session={session_id}, Question={question}")

    result = llm_web_answer(question)
    text = (result or {}).get("answer", "")

    add_agent(session_id, text)
    state["answer"] = {
        "text": text,
        "citations": (result or {}).get("citations", []),
        "mode": (result or {}).get("mode", "llm"),
        "evidence": (result or {}).get("evidence", []),
    }
    state["history"] = get_history(session_id)
    logger.info(f"[Web Node] Answer length={len(text)}")
    return state


# This function handles decision making using action string from router node. 
# It maps the router's action string into which agent node to run next
def branch_by_action(state):
    action = (state.get("decision") or {}).get("action", "")
    logger.info(f"[Branch] Deciding route for action={action}")
    if action == "retrieve_pdfs":
        return "pdf_agent"
    if action == "clarify":
        return "clarify_agent"
    if action == "web_search":
        return "web_agent"
    return "pdf_agent"


# This is where the graph is defined. It registers nodes, sets entry point, 
# addes conditional edges, and compiles with langgraph's memory saver
builder = StateGraph(dict)
builder.add_node("router", router_node)
builder.add_node("pdf_agent", pdf_node)
builder.add_node("clarify_agent", clarify_node)
builder.add_node("web_agent", web_node)

builder.set_entry_point("router")
builder.add_conditional_edges(
    "router",
    branch_by_action,
    {
        "pdf_agent": "pdf_agent",
        "clarify_agent": "clarify_agent",
        "web_agent": "web_agent",
    },
)
builder.add_edge("pdf_agent", END)
builder.add_edge("clarify_agent", END)
builder.add_edge("web_agent", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# This function is like an adapter between langgraph and the api.
# It runs the graph when session started and also return output as JSON schema
def run_graph(session_id, question):
    logger.info(f"[Run Graph] Start session={session_id}, question={question}")
    state = {"session_id": session_id, "question": question}
    final = graph.invoke(state, config={"configurable": {"thread_id": session_id}})
    logger.info(f"[Run Graph] Finished with action={(final.get('decision') or {}).get('action', '')}")
    return {
        "session_id": session_id,
        "question": question,
        "router": {
            "action": (final.get("decision") or {}).get("action", ""),
            "used_context": (final.get("decision") or {}).get("used_context", ""),
            "reason": (final.get("decision") or {}).get("reason", ""),
            "scores": (final.get("decision") or {}).get("scores", []),
            "query": (final.get("decision") or {}).get("query", question),
        },
        "answer": final.get("answer", {}),
        "history": final.get("history", []),
    }
