import os
import re
import logging
from textwrap import dedent
from urllib.parse import quote_plus
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.log import setup as setup_logging

setup_logging()
logger = logging.getLogger("llm")

# For creating a compact evidence string for prompting the LLM
def format_evidence(found, max_items=8, max_chars=None):
    grouped = {}
    for f in found:
        source = f.get("source", "unknown")
        page = f.get("page", "?")
        key = (source, page)
        text = (f.get("text") or "").strip()
        if not text:
            continue
        grouped.setdefault(key, []).append(text)

    items = []
    for (source, page), texts in grouped.items():
        text = " ".join(texts)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        items.append((source, page, text))

    items = items[:max_items]

    lines = []
    for i, (source, page, text) in enumerate(items, 1):
        lines.append(f"[{i}] {source} p.{page}: {text}")
    return "\n".join(lines)

# Return citations for the API response
def citations(found):
    output = []
    for f in found:
        output.append({
            "source": f.get("source", "unknown"),
            "page": f.get("page", "?"),
            "score": float(f.get("score", 0.0)),
        })
    return output

# Offline: return full evidence with labels (need optimization later for better response)
def local_answer(found, total_limit=None):
    parts = []
    used = []
    total = 0

    for f in found:
        text = (f.get("text") or "").strip()
        if not text:
            continue

        piece = (("\n\n" if parts else "") + text)

        if isinstance(total_limit, int):
            if total + len(piece) > total_limit:
                room = max(total_limit - total, 0)
                piece = piece[:room].rstrip()
                if not piece:
                    break

        parts.append(piece)
        used.append(f)
        total += len(piece)

        if isinstance(total_limit, int) and total >= total_limit:
            break

    return {
        "answer": "".join(parts),
        "citations": citations(used),
        "mode": "local",
    }

# Offline: return with predetermined answer for vague questions
def local_clarify():
    message = (
            "Could you share a bit more detail so I can be more precise? "
            "Which dataset or scope? "
            "Any specific PDF or research papers? "
            "Any constraints (model, prompt style, time)?"
    )
    return {"answer": message, "citations": [], "mode": "local"}

# Offline: return with predetermined answer for out of scope questions
def local_web():
    return {"answer": "This looks outside the PDFs, please re-check on the web.",
            "citations": [], "mode": "local"}

# Return a text block that contain content from last few turns
# We keep it short and not pull all past turn to not overload the system and to decrease hallucination
def history_block(question, history, max_recent=6, top_k=4, max_chars=1200):
    if not history:
        return ""

    # Take the last few turns, words by words
    recent = list(history)[-max_recent:]

    # Score older turns by token overlap with the current question
    def tokens(s):
        import re
        return {t for t in re.findall(r"[A-Za-z0-9-]+", (s or "").lower()) if len(t) >= 3}

    question_set = tokens(question or "")
    earlier = list(history)[:-max_recent] if len(history) > max_recent else []
    scored = []
    if question_set and earlier:
        for role, text in earlier:
            t = (text or "").strip()
            if not t:
                continue
            overlap = len(question_set & tokens(t))
            if overlap > 0:
                scored.append((overlap, role, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [(role, t) for _, role, t in scored[:top_k]]
    else:
        top = []

    # Combine top turns and specify roles for context
    combined = top + recent
    lines = [f"{role}: {(text or '').strip()}" for role, text in combined if (text or "").strip()]
    block = "\n".join(lines)

    if len(block) > max_chars:
        block = block[-max_chars:] 

    return block

# Build a clean snippet list for the API + a readable block for the LLM
def web_snippets(query, max_results = 5):
    search = DuckDuckGoSearchAPIWrapper(region="us-en", time="m", safesearch="moderate")
    results = search.results(query, max_results=max_results)

    items = []
    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        link = (r.get("link") or "").strip()
        snippet = (r.get("snippet") or r.get("content") or "").strip()
        if not (title or snippet):
            continue
        items.append({"title": title, "link": link, "snippet": snippet})
        line = f"[{i}] {title} — {snippet}"
        if link:
            line += f"\n({link})"
        lines.append(line)

    return items, "\n\n".join(lines)

# This function allow the agent to request search.
# The llm agent can ask for search by replies with 'SEARCH: <query>'
# We set a cap for number of search per query to avoid infinite loop of searching (llm may spam SEARCH)
def self_search(llm, *, system_rules, question, context="", evidence="", max_rounds: int = None, results_per_round: int = 5,
):
    # Define the maximum limit for consecutive search per query
    if max_rounds is None:
        try:
            max_rounds = int(os.getenv("SELF_SEARCH_ROUNDS", "10"))
        except Exception:
            max_rounds = 10

    combined_block = (evidence or "").strip()
    gathered_items = []

    # Remember queries to avoid infinite loops
    seen_queries = set()

    def ask_with(block: str) -> str:
        prompt = dedent(f"""
            {system_rules}

            Question: {question}

            Context:
            {context or "none"}

            Evidence:
            {block or "none"}

            If you need more up-to-date information or external facts not present here,
            reply with exactly this one line: SEARCH: <your short search query>
            Otherwise, write the final answer in plain English paragraphs now in 2–5 sentences.

            IMPORTANT: Do not print the word SEARCH unless you truly need a web query.
            Answer:
        """)
        response = llm.invoke([HumanMessage(content=prompt)])
        return (getattr(response, "content", "") or "").strip()

    # Ask with what we already have (round 0)
    reply = ask_with(combined_block)

    for round_index in range(max_rounds):
        m = re.match(r"(?is)^[`\s>]*search\s*:\s*(.+)$", reply)
        if not m:
            return reply, gathered_items

        query = m.group(1).strip().strip('"').strip("'")

        # Loop guards
        normal_query = query.lower()
        if not normal_query:
            break
        if normal_query in seen_queries:
            break
        seen_queries.add(normal_query)

        # Do web search and add snippets
        items, block = web_snippets(query, max_results=results_per_round)

        gathered_items.extend(items)
        combined_block = (combined_block + ("\n\n" if combined_block else "") + block).strip()

        # Ask again with bigger evidence pools
        reply = ask_with(combined_block)

    # Force answer if we see SEARCH query after we existing the loop (another safe guard)
    if re.match(r"(?is)^[`\s>]*search\s*:", reply):
        final_prompt = dedent(f"""
            {system_rules}

            Question: {question}

            Context:
            {context or "none"}

            Evidence:
            {combined_block or "none"}

            You must provide the final answer in plain English paragraphs now in 2–5 sentences using the evidence above.
            Do not request another search and do not output SEARCH in your final answer.
            Answer:
        """)
        response = llm.invoke([HumanMessage(content=final_prompt)])
        forced = (getattr(response, "content", "") or "").strip()
        if forced and not re.match(r"(?is)^[`\s>]*search\s*:", forced):
            return forced, gathered_items

    # Final safe guard. We return the best snippets if the agent still cannot find an answer
    return ("Here are the most relevant snippets I found:" + (combined_block or "")), gathered_items


# Normal PDF retrieval path. Uses langchain-openai if OPENAI_API_KEY is set, otherwise falls back to local
def llm_answer(question, found, history=None):
    logger.info("llm_answer called", extra={"question": question})
    try:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            logger.warning("No API key found, falling back to local_answer")
            return local_answer(found)

        evidence = format_evidence(found)
        context = history_block(question, history)

        system_rules = dedent(f"""
            You are a research assistant. Answer using only the evidence provided unless you explicitly request a web search.

            Rules:
            - Give a clear, concise answer in plain English (2–5 sentences).
            - If the evidence is partial or mixed, explain that briefly or compare if appropriate.
            - Always cite sources in brackets like [1], [2] based on the evidence list.
            - If the evidence contains tables, schemas, figures, or code, do not copy them. Instead, describe their meaning in natural language.
            - Answer in simple paragraphs.

            If you need more up-to-date information or facts that not present here then reply with
            exactly this one line: SEARCH: <your short search query>
        """)

        llm = ChatOpenAI(
            model="gpt-5",   
            temperature=0,
            api_key=api_key,
        )

        text, web_items = self_search(
            llm,
            system_rules=system_rules,
            question=question,
            context=context or "none",
            evidence=evidence
        )

        logger.info("llm_answer success", extra={"used_evidence": bool(evidence)})
        return {"answer": text, "citations": citations(found), "mode": "llm", "evidence": web_items}

    except Exception as e:
        logger.info("llm_answer success", extra={"used_evidence": bool(evidence)})
        return local_answer(found)

# An entrypoint the API calls when router decided its PDF retrieval path
def compose_answer(question, found, history=None):
        return llm_answer(question, found, history=history)

# Clarification path. Use same structure as the normal PDF retrieval path
def llm_clarify(question, history=None):
    logger.info("llm_clarify called", extra={"question": question})
    try:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            logger.warning("No API key found, falling back to local_clarify")
            return local_clarify()
        
        context = history_block(question, history)

        system_rules = dedent(f"""
            You a research assistant.

            The users question may be vague or underspecified. Your task is to:
            - Focus purely on understanding the question and its context.
            - Be concise and neutral; do not speculate or propose an answer yet.
            - Do not mention any PDFs or internal tools.
            - Keep the response short and enough (2-3 sentences) for user to understand what they have to clarify next.
            - Answer in simple paragraphs.
            - Keep the reponse neutral and general (not digging too deep to assumed context).
            - Try not to give too many examples as it could make the response feel too specific.
            
            If you truely need external or facts to clarify then reply with exactly this one line: SEARCH: <your short search query>
                        
            Question: {question}

            Conext:
            {context if context else "none"}

            Answer:
        """)

        llm = ChatOpenAI(
            model="gpt-5",
            temperature=0,
            api_key=api_key,
        )
        text, _ = self_search(
            llm,
            system_rules=system_rules,
            question=question,
            context=context or "none",
            evidence="" 
        )

        logger.info("llm_clarify success")
        return {"answer": text, "citations": [], "mode": "llm"}

    except Exception as e:
        logger.error("llm_clarify failed", extra={"error": str(e)})
        return local_clarify()

def compose_clarify(question, history=None):
    return llm_clarify(question, history=history)

# Out of Scope path. Use same structure as other route except it has web search capability
# Use same score and ranknig system as in search.py (TF-IDF + cosine)
def llm_web_answer(question, history=None):
    logger.info("llm_web_answer called", extra={"question": question})
    try:
        # Fetch a few candidate results
        search = DuckDuckGoSearchAPIWrapper(
            region="us-en", time="m", safesearch="moderate"
        )
        results = search.results(question, max_results=5) 

        # Form text chunk for each candidate. Keep link for citation.
        candidates = []
        for r in results:
            title = (r.get("title") or "").strip()
            link = (r.get("link") or "").strip()
            snippet = (r.get("snippet") or r.get("content") or "").strip()
            text = " ".join(t for t in [title, snippet] if t)
            if text:
                candidates.append({"text": text, "title": title, "link": link})

        # Local fallback if there is no search result
        if not candidates:
            logger.warning("No web candidates found, using local_web fallback")
            return local_web()

        # Rank candidates with the same idea as search.py
        texts = [c["text"] for c in candidates]
        vector = TfidfVectorizer(stop_words="english", max_df=0.85, ngram_range=(1, 2))
        matrix = vector.fit_transform([question] + texts)
        query_vector = matrix[0:1]
        doc_matrix = matrix[1:]
        scores = cosine_similarity(query_vector, doc_matrix)[0]

        order = list(range(len(candidates)))
        order.sort(key=lambda i: scores[i], reverse=True)
        top_k_idx = order[:5]

        # Build a clean snippet list for the API + a readable block for the LLM
        web_items = []
        lines = []
        for rank, idx in enumerate(top_k_idx, 1):
            candidate = candidates[idx]
            web_items.append({
                "title": candidate["title"],
                "link": candidate["link"],
                "snippet": candidate["text"],
                "score": float(scores[idx]),
            })
            line = f"[{rank}] {candidate['title']} — {candidate['text']}"
            if candidate["link"]:
                line += f"\n({candidate['link']})"
            lines.append(line)

        evidence = "\n\n".join(lines)

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            return {"answer": "This looks outside the PDFs, please re-check on the web.",
                    "citations": [], "mode": "local", "evidence": web_items}
        
        context = history_block(question, history)

        system_rules = dedent(f"""
        You are a research assistant.

        The users may ask questions that are not inside the provided PDFs list or they may explicitly ask you to search
        websites to find answers. Your task is to:
        - Use the search results (evidence) to answer the questions.
        - Give a clear, concise answer in plain English paragraphs (2–5 sentences).
        - Always say where you got the information from for transparency.
        - If the provided evidence are insufficient for you to form concise answer,                   
        you may request a web search by replying: SEARCH: <your short search query>.
        Then use the new snippets you got from search to write a clear and concise answer in plain English (2-5 sentences).
        """)

        llm = ChatOpenAI(
            model="gpt-5",
            temperature=0,
            api_key=api_key,
        )

        text, more_web = self_search(
            llm,
            system_rules=system_rules,
            question=question,
            context=context or "none",
            evidence=evidence
        )

        # If the second search is better then use it
        final_web = more_web or web_items

        logger.info("llm_web_answer success", extra={"results": len(final_web)})
        return {"answer": text, "citations": [], "mode": "llm", "evidence": final_web}

    except Exception as e:
        logger.error("llm_web_answer failed", extra={"error": str(e)})
        return local_web()

def compose_web(question, history=None):
    return llm_web_answer(question, history=history)



