import streamlit as st
import boto3
import os
import time
from retriever import load_vectorstore, build_bm25_index, retrieve
from citation_chain import load_llm, stream_answer_with_citations
from ui import inject_theme, render_title, render_sources, render_blocked, render_sidebar

# ─── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Ask My Docs — ACOTAR",
    page_icon="🌹",
    layout="wide"
)

# ─── Inject Night Court theme ─────────────────────────────
inject_theme()

# ─── Guardrail pre-check ──────────────────────────────────
def is_blocked_by_guardrail(question: str) -> tuple:
    BLOCKED_MESSAGE = (
        "I can only answer questions about A Court of Mist and Fury. "
        "Try asking about the characters, plot, or lore of the book."
    )

    # Layer 1 — keyword check, no Bedrock call
    how_to_signals = [
        "step by step", "how to make", "how do i make",
        "how do you make", "recipe", "instructions for",
        "teach me how", "give me steps", "walk me through"
    ]
    if any(signal in question.lower() for signal in how_to_signals):
        return True, BLOCKED_MESSAGE

    # Layer 2 — Bedrock guardrail
    guardrail_id      = os.getenv("GUARDRAIL_ID")
    guardrail_version = os.getenv("GUARDRAIL_VERSION", "1")

    if not guardrail_id:
        return False, ""

    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        response = client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source="INPUT",
            content=[{"text": {"text": question}}]
        )
        if response["action"] == "GUARDRAIL_INTERVENED":
            return True, BLOCKED_MESSAGE
    except Exception:
        return False, ""  # fail open

    return False, ""


# ─── Load components once ────────────────────────────────
@st.cache_resource
def load_components():
    vs                = load_vectorstore()
    bm25, docs, metas = build_bm25_index(vs)
    llm               = load_llm()
    return vs, bm25, docs, metas, llm

vs, bm25, docs, metas, llm = load_components()

# ─── Session state ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Sidebar ─────────────────────────────────────────────
should_clear = render_sidebar(st.session_state.messages)
if should_clear:
    st.session_state.messages = []
    st.rerun()

# ─── Title ───────────────────────────────────────────────
render_title()

# ─── Chat history ─────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

# ─── Chat input ───────────────────────────────────────────
if prompt := st.chat_input("Ask anything about the book..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Guardrail check — before retrieval ────────────────
    blocked, blocked_message = is_blocked_by_guardrail(prompt)

    if blocked:
        with st.chat_message("assistant"):
            render_blocked(blocked_message)
        st.session_state.messages.append({
            "role": "assistant",
            "content": blocked_message
        })

    else:
        # ── Full pipeline ─────────────────────────────────
        with st.chat_message("assistant"):
            try:
                chat_history = st.session_state.messages[-6:] if st.session_state.messages else None

                with st.status("Searching the Night Court...", expanded=True) as status:
                    st.write("✦ Searching with keywords...")
                    time.sleep(0.4)
                    st.write("✦ Searching by meaning...")
                    time.sleep(0.4)
                    st.write("✦ Combining results...")
                    chunks = retrieve(prompt, vs, bm25, docs, metas)
                    st.write("✦ Reranking passages...")
                    time.sleep(0.3)
                    st.write("✦ Composing answer...")
                    status.update(label="Passages found", state="complete")

                answer_placeholder = st.empty()
                full_answer        = ""
                final_sources      = []

                for token, sources in stream_answer_with_citations(
                    prompt, chunks, llm, chat_history=chat_history
                ):
                    if token is not None:
                        full_answer += token
                        answer_placeholder.markdown(full_answer + "▌")
                    else:
                        final_sources = sources

                answer_placeholder.markdown(full_answer)

                # Suppress sources if LLM guardrail caught on output
                is_blocked_output = (
                    "I can only answer questions about A Court of Mist and Fury"
                    in full_answer
                )

                if not is_blocked_output:
                    render_sources(final_sources)

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": full_answer,
                    "sources": [] if is_blocked_output else final_sources
                })

            except Exception as e:
                error_msg = f"Something went wrong: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })