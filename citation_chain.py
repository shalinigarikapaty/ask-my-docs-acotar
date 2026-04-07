from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()
# ─── Load Configuration ───────────────────────────────────────
LLM_MODEL   = os.getenv("LLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION      = os.getenv("AWS_REGION", "us-east-1")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# ─── Load LLM ────────────────────────────────────────────
def load_llm():
    return ChatBedrock(
        model_id=LLM_MODEL,
        region_name=REGION,
        streaming=True,
         model_kwargs={
            "temperature": TEMPERATURE,
            "max_tokens":  int(os.getenv("MAX_TOKENS", "1000"))
        }
    )

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
def stream_answer_with_citations(question, chunks, llm, chat_history=None):
    context, sources = format_context(chunks)
    system, user     = build_prompt(question, context)

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
# ─── Build prompt with citations instruction ──────────────
def build_prompt(question, context):
    system = """You are an expert assistant for the book 
"A Court of Mist and Fury" (ACOMAF) by Sarah J. Maas.

Your rules:
1. ONLY answer questions about ACOMAF and its characters, 
   plot, lore, and world
2. If the question is not about ACOMAF, respond with:
   "I can only answer questions about A Court of Mist and Fury."
3. Answer using ONLY the provided source passages
4. Cite your sources inline using [Source N] notation  
5. If the answer is not in the sources say:
   "I cannot find this in the provided text"
6. Never use outside knowledge beyond what is in the sources
7. Be specific and quote directly when relevant"""

    user = f"""Here are the relevant passages from the book:

{context}

Question: {question}

Answer with citations:"""

    return system, user

# ─── Generate answer with citations ──────────────────────
def answer_with_citations(question, chunks, llm,chat_history=None):
    context, sources = format_context(chunks)
    system, user = build_prompt(question, context)
# Build message list with history
    messages = [SystemMessage(content=system)]
  # Add previous turns if they exist
    if chat_history:
        for turn in chat_history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
                
    # Add current question
    messages.append(HumanMessage(content=user))

    response = llm.invoke(messages)
    return response.content, sources

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

    chunks          = retrieve(question, vs, bm25, docs, metas)
    answer, sources = answer_with_citations(question, chunks, llm)

    print("ANSWER:")
    print(answer)
    print("\nSOURCES:")
    for s in sources:
        print(f"  {s['label']} — Page {s['page']}")
        print(f"  {s['text'][:150]}...")
        print()