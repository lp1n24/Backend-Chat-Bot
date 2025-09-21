import re

max_candidates = 8 # Total candidate queries we want to test
top_past_turns = 3 # How many past turns to check for context
min_token_overlap = 2 # Only consider past turns that share more than certain number of meaningful tokens
k_results = 5 # Retrieve top-k chunks for each candidate

low_max = 0.08 # Set threshold to determine the out-of-context query
low_max_and_avg = (0.15, 0.07) # Another test. Both top1 < 0.15 and avg(top3) < 0.07 then treat it as out-of-context query
low_confidence = 0.1 # If goes below then retrieval is weak unless not flat
very_low = 0.08 # If goes below and have flat score then always ask for clarification

flat12_percent = 0.12 # If the difference between top-1 and 2 is flat then its ambiguous
flat1avg_percent = 0.25 # If the avg of top-3 is flat then its ambiguous
mid_confidence = 0.18 # If scores are flat but max < mid_confidence then its amniguous 

oos_ratio = 0.6 # >60% of tokens are not in the TF-IDF vocab
oos_max = 0.25 # Max similarity to use in Out-Of-Scope condition

min_idf = 1.3 # Ignore tokens that appear frequently in most PDFs

# Build a mapping (term: idf score) from trained vectorizer to use in deciding which tokens are meaningful
def build_idf_map(vectorizer):
    terms = vectorizer.get_feature_names_out()
    idfs = vectorizer.idf_
    idf_map = {}
    for i in range (len(terms)):
        idf_map[terms[i]] = float(idfs[i])
    return idf_map

# Turn a raw text string into a set of meaningful tokens to use in comparing overlap tokens between question and past turns
def token_set(text, vocab, idf_map, min_idf=min_idf):
    words = re.split(r"\W+", (text or "").lower())
    tokens = set()
    for w in words:
        if not w or (w not in vocab):
            continue
        if idf_map.get(w, 0.0) >= min_idf:
            tokens.add(w)
    return tokens

# Find fraction of tokens that is not in the TF-IDF vocabulary
def oov_ratio(text, vocab):
    words = [w for w in re.split(r"\W+", (text or "").lower()) if w]
    if not words:
        return 1.0
    in_vocab = sum(1 for w in words if w in vocab)
    return 1.0 - (in_vocab / len(words))

# Get only numeric scores from information retrieved by our retriever
def scores_only(found):
    scores = []
    for f in found:
        scores.append(float(f["score"]))
    scores.sort(reverse=True)
    return scores

def tokens(s):
    return {w for w in re.split(r"\W+", (s or "").lower()) if len(w) >= 3}

# To see if the query contains something like author name that matches the PDF filename
def pdf_hint(query, source_name):
    return len(tokens(query) & tokens(source_name)) > 0

# Summarize retrieval scores to help in decision making (out-of-context, ask for clarification, or normal retrieval)
def summary(scores):
    if not scores:
        return 0.0, 0.0, 0.0, 0.0
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    top3 = scores[2] if len(scores) > 2 else 0.0
    gap12 = top1 - top2
    gap13 = top1 - top3
    avg3 = (top1 + top2 + top3) / max(1, min(3, len(scores)))
    return top1, gap12, gap13, avg3

# Select top-k most meaningful past turns (share most overlap tokens with new query) from session history
def top_k_past_turns (history, query, vocab, idf_map, k=top_past_turns):
    if not history:
        return []
    query_tokens = token_set(query, vocab, idf_map)
    scored = []
    for role, text in history:
        if not text or not text.strip():
            continue
        turn_tokens = token_set(text.strip(), vocab, idf_map)
        overlap = len(query_tokens & turn_tokens)
        if overlap >= min_token_overlap:
            scored.append((overlap, text.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    top_turns = []
    for overlap, turn_text in scored[:k]:
        top_turns.append(turn_text)
    return top_turns

# Count how many tokens in the text are specific to this query
def specific_token_count(text, vectorizer, min_idf=1.5):
    if not text:
        return 0
    vocab = vectorizer.vocabulary_
    terms = vectorizer.get_feature_names_out()
    idfs = vectorizer.idf_
    idf_map = {terms[i]: float(idfs[i]) for i in range(len(terms))}
    words = re.split(r"\W+", text.lower())
    count = 0
    for w in words:
        if not w or w not in vocab:
            continue
        if idf_map.get(w, 0.0) >= min_idf:
            count += 1
    return count

# To check how many distinct PDF sources are represented in the retrieved list
def source_diversity(found, top_n=5):
    try:
        return len({f.get("source", "") for f in (found or [])[:top_n] if f.get("source")})
    except Exception:
        return 0

# The main routing function (entry point used by the api script).
def route(question, retriever, history=None, k=k_results):
    q = (question or "").strip()
    q_specific = specific_token_count(q, retriever.vectorizer, min_idf=1.5)

    query_specific = 3
    vocab_keys = retriever.vectorizer.vocabulary_.keys()
    idf_map = build_idf_map(retriever.vectorizer)
        
    # Build candidate queries
    candidates = [("raw", q)]

    last_user = ""
    last_agent = ""
    if history:
        for role, text in reversed(history):
            if not last_user and role == "user" and text and text.strip():
                last_user = text.strip()
            if not last_agent and role == "agent" and text and text.strip():
                last_agent = text.strip()
            if last_user and last_agent:
                break
            
    if last_user:
        candidates.append(("user_c", last_user + " " + q))
    if last_agent:
        candidates.append(("agent_c", last_agent + " " + q))
    best_past = top_k_past_turns(history or [], q, vocab_keys, idf_map, k=top_past_turns)
    for past_text in best_past:
        candidates.append(("best_past_c", past_text + " " + q))
    
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    
    # Score each candidate using retriever
    scored = []
    for label, candidates_q in candidates:
        found = retriever.search(candidates_q, k=k)
        scores = scores_only(found)
        max_score, gap12, gap13, avg3 = summary(scores)
        scored.append({
            "label": label,
            "query": candidates_q,
            "found": found,
            "scores": scores,
            "max": max_score,
            "gap12": gap12,
            "gap13": gap13,
            "avg3": avg3
        })

    scored.sort(key=lambda item: (item["max"], item["avg3"]), reverse=True)
    best = scored[0] if len(scored) > 0 else None

    # Choose actions based on the score we got
    # If no retrieval results OR system picked raw query over past context AND
    # the differences between top-1 and top-2 is too close OR
    # the differences between top-1 and average top-3 is too close OR
    # the question has low specific tokens count then ask for claification
    # If the similarity is too weak and have too many words outside the TF-IDF vocabulary then its out-of-scope
    # Otherwise its normal retrieval from PDFs

    if (not best) or (not best["scores"]) or (sum(best["scores"]) == 0):
        return {"action": "clarify", "reason": "no retrieval results", "used_context": "none"}
    
    max_score = best["max"]
    avg3 = best["avg3"]

    oos_query = oov_ratio(q, retriever.vectorizer.vocabulary_.keys())

    # If the query mentions any of the top sources then its not out-of-scope
    hinted = any(pdf_hint(q, f.get("source", "")) for f in (best["found"] or [])[:3])

    # Out-of-scope: only trigger if not hinted
    if (not hinted) and (
        (max_score < low_max)
        or (max_score < low_max_and_avg[0] and avg3 < low_max_and_avg[1])
        or (oos_query >= oos_ratio and max_score < oos_max)
    ):
        return {
            "action": "web_search",
            "reason": "low similarity to the provided PDFs",
            "used_context": best["label"],
            "scores": best["scores"],
            "query": best["query"]
        }
    
    try:
        top_found = (best["found"] or [])[0]
    except Exception:
        top_found = None

    # If the query is specific enough and point to specific source then use that source
    if q_specific > query_specific:
        if top_found and pdf_hint(q, top_found.get("source", "")):
            return {
                "action": "retrieve_pdfs",
                "reason": "query mentions this PDF; prioritizing that source",
                "used_context": best["label"],
                "scores": best["scores"],
                "query": best["query"],
                "found": best["found"],
            }

    
    # Ambiguous: system picked raw query AND either the scores are flat but max score is not high
    # OR the query lacks specific tokens
    if best["label"] == "raw":
        top1 = max_score
        top2 = best["scores"][1] if len(best["scores"]) > 1 else 0.0
        percent_based = max(top1, 1e-9)
        flat12 = ((top1 - top2) / percent_based) < flat12_percent
        flat1avg = ((top1 - avg3) / percent_based) < flat1avg_percent
        vague = q_specific <= query_specific

        diverse_sources = source_diversity(best["found"], top_n=5) >= 2

        if top1 < very_low and (flat12 or flat1avg or diverse_sources):
            return {
                "action": "clarify",
                "message": "Could you be more specific?",
                "reason": "retrieval confidence is very low and top results are flat",
                "used_context": best["label"],
                "scores": best["scores"],
                "query": best["query"],
            }

        if top1 < low_confidence and (flat12 or flat1avg):
            return {
                "action": "clarify",
                "message": "Could you be more specific?",
                "reason": "retrieval confidence is low and top results are close",
                "used_context": best["label"],
                "scores": best["scores"],
                "query": best["query"],
            }

        if ((flat12 or flat1avg) and (top1 < mid_confidence)) or vague or (diverse_sources and (flat12 or flat1avg)):
            return {
                "action": "clarify",
                "message": "Could you be more specific?",
                "reason": "scores across top results are very close or query is vague",
                "used_context": best["label"],
                "scores": best["scores"],
                "query": best["query"]
            }
    
    # Otherwise: confident retrieval from the provided PDFs
    return {
            "action": "retrieve_pdfs",
            "reason": "confident match in PDFs",
            "used_context": best["label"],
            "scores": best["scores"],
            "query": best["query"],
            "found": best["found"]
    }