from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import AWS_REGION, LLM_MODEL, TEMPERATURE, MAX_TOKENS, GUARDRAIL_ID, GUARDRAIL_VERSION

# ─── Load LLM ────────────────────────────────────────────
def load_llm():
    llm_kwargs = {
        "model_id":    LLM_MODEL,
        "region_name": AWS_REGION,
        "streaming":   True,
        "model_kwargs": {
            "temperature":   TEMPERATURE,
            "max_tokens":    MAX_TOKENS,
            "stop_sequences": [
                "\nHuman:",
                "\nQuestion:",
                "[END]",
            ]
        }
    }

    if GUARDRAIL_ID:
        llm_kwargs["guardrails"] = {
            "guardrailIdentifier": GUARDRAIL_ID,
            "guardrailVersion":    GUARDRAIL_VERSION,
            "trace":               "enabled"
        }

    return ChatBedrock(**llm_kwargs)  # ← this was missing
    

# ─── Format chunks with source labels ────────────────────
def format_context(chunks):
    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks):
        page = chunk["meta"].get("page", "unknown")
        text = chunk["doc"]
        label = f"[Source {i+1}]"

        context_parts.append(f"{label} (Page {page}):\n{text}")
        sources.append({
            "label": label,
            "page": page,
            "text": text[:300]
        })

    return "\n\n".join(context_parts), sources
#-------Streaming-------------------------------------------
def stream_answer_with_citations(question, chunks, llm, chat_history=None, confidence: str = "high"):
    context, sources     = format_context(chunks)
    confidence_context   = build_confidence_context(confidence)
    system, user         = build_prompt(question, context, confidence_context)

    # Build message list with history
    messages = [SystemMessage(content=system)]

    if chat_history:
        for turn in chat_history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=user))

    # Stream tokens as they arrive
    full_response = ""
    for chunk in llm.stream(messages):
        token = chunk.content
        full_response += token
        yield token, None  # yield token, sources not ready yet

    yield None, sources  # final yield — signals completion with sources
# ─── Confidence context injected into user message ───────
_CONFIDENCE_LINES = {
    "high":     "[CONFIDENCE: high] Strong match found in the book passages — answer normally.",
    "moderate": "[CONFIDENCE: moderate] Partial match found — answer but flag any uncertainty.",
    "low":      "[CONFIDENCE: low] Weak match — do not guess; tell the user the passages don't cover this and suggest a correction if the query may be a misspelling.",
    "none":     "[CONFIDENCE: none] No relevant passages found — do not guess; tell the user this topic couldn't be found and suggest a correction if applicable.",
}

def build_confidence_context(confidence: str) -> str:
    return _CONFIDENCE_LINES.get(confidence, _CONFIDENCE_LINES["none"])

# ─── Build prompt with citations instruction ──────────────
def build_prompt(question, context, confidence_context: str = ""):
    system ="""You are a knowledgeable and passionate companion for readers of \
    "A Court of Mist and Fury" (ACOMAF) by Sarah J. Maas. You speak with warmth and \
    genuine love for the story — like a fellow reader who has memorised every page.

    Your rules:
    1. Begin every response with a brief, natural greeting or acknowledgment of the \
    question — something that feels like a conversation, not a database query. \
    For example: "Ah, that is one of the most haunting threads in the book..." \
    or "Such a good question — this moment is easy to miss..." \
    Keep it to one sentence, then move into the answer.

    2. ONLY answer questions about ACOMAF and its characters, plot, lore, and world.

    3. If the question is not about ACOMAF, respond warmly but firmly:
    "That question takes us outside the world of this book — I am only able to \
    speak to the story of A Court of Mist and Fury."

    4. Answer using ONLY the provided source passages. Never invent details.

    5. Cite your sources inline using [Source N] notation naturally within the prose \
    — not as footnotes at the end of sentences, but woven in: \
    "Rhysand reveals [Source 2] that the bond had been there since..."

    6. If the answer is not in the provided sources, say:
    "The passages I have do not shed light on this — it may live in a part of the \
    book not captured here."

    7. Never use outside knowledge beyond what is in the sources.

    8. A [CONFIDENCE] signal will appear at the top of the user message. Use it to \
    calibrate your response:
    - high: answer normally using the sources.
    - moderate: answer using the sources but acknowledge any uncertainty naturally \
    (e.g. "the passages hint at this, though they don't spell it out explicitly").
    - low or none: do not guess or extrapolate. Tell the user the passages do not \
    contain enough to answer. If the query looks like a possible misspelling of an \
    ACOMAF character or place name, gently suggest what they may have meant. \
    Never invent names or facts.

    9. Write in flowing, literary prose — not bullet points. \
    Match the tone of the book: atmospheric, emotionally intelligent, \
    specific about detail. A reader deserves an answer that feels as \
    alive as the story itself."""

    confidence_prefix = f"{confidence_context}\n\n" if confidence_context else ""
    user = f"""{confidence_prefix}Here are the relevant passages from the book:

{context}

Question: {question}

Answer with citations:"""

    return system, user

# ─── Generate answer with citations ──────────────────────
def answer_with_citations(question, chunks, llm, chat_history=None, confidence: str = "high"):
    context, sources     = format_context(chunks)
    confidence_context   = build_confidence_context(confidence)
    system, user         = build_prompt(question, context, confidence_context)

    messages = [SystemMessage(content=system)]

    if chat_history:
        for turn in chat_history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=user))

    try:
        response = llm.invoke(messages)

        # Guardrail returns blocked message as content instead of raising
        # Check for it and return empty sources so UI doesn't show citations
        if "I can only answer questions about A Court of Mist and Fury" in response.content:
            return response.content, []

        return response.content, sources

    except Exception as e:
        error = str(e)
        if "GuardrailIntervention" in error or "blocked" in error.lower():
            return (
                "I can only answer questions about A Court of Mist and Fury. "
                "Try asking about the characters, plot, or lore of the book.",
                []
            )
        raise

# ─── Test function ────────────────────────────────────────
if __name__ == "__main__":
    from retriever import load_vectorstore, build_bm25_index, retrieve

    print("Loading components...")
    vs              = load_vectorstore()
    bm25, docs, metas = build_bm25_index(vs)
    llm             = load_llm()
    print("Ready.\n")

    question = "Who is Feyre's mate?"
    print(f"Question: {question}\n")

    chunks, confidence = retrieve(question, vs, bm25, docs, metas)
    answer, sources    = answer_with_citations(question, chunks, llm, confidence=confidence)

    print("ANSWER:")
    print(answer)
    print("\nSOURCES:")
    for s in sources:
        print(f"  {s['label']} — Page {s['page']}")
        print(f"  {s['text'][:150]}...")
        print()