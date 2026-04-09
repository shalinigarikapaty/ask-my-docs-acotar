import streamlit as st
import boto3
import os
import time
from retriever import load_vectorstore, build_bm25_index, retrieve
from citation_chain import load_llm, stream_answer_with_citations
from ui import (
    inject_theme, render_title, render_watermark,
    render_source_cards, render_blocked, render_sidebar,
    replace_citations_with_pills
)

# ─── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Ask My Docs — ACOTAR",
    page_icon="🪶",
    layout="wide"
)

inject_theme()

# ─── Avatars ──────────────────────────────────────────────
USER_AVATAR      = "🪶"   # the reader with a quill
ASSISTANT_AVATAR = "🗝️"  # the keeper of keys

# ─── Guardrail pre-check ──────────────────────────────────
def is_blocked_by_guardrail(question: str) -> tuple:
    BLOCKED_MESSAGE = (
        "I can only answer questions about A Court of Mist and Fury. "
        "Try asking about the characters, plot, or lore of the book."
    )
    GREETING_MESSAGE = (
        "Welcome. I am here to guide you through the world of "
        "A Court of Mist and Fury — ask me anything about the "
        "characters, plot, lore, or magic of the book."
    )

    q = question.lower().strip()

    # Layer 0 — greeting / chitchat
    greeting_phrases = [
        "hey", "hi", "hello", "hiya", "howdy", "sup",
        "good morning", "good evening", "good afternoon", "good night",
        "how are you", "what's up", "whats up",
        "who are you", "what are you", "what can you do",
        "what's your name", "whats your name", "what is your name",
        "tell me about yourself", "introduce yourself"
    ]
    short_starters = ["hey", "hi", "hello", "hiya", "howdy"]
    is_greeting = (
        any(q == phrase for phrase in greeting_phrases) or
        any(q.startswith(phrase) for phrase in greeting_phrases) or
        (any(q.startswith(s) for s in short_starters) and len(q.split()) < 6)
    )
    if is_greeting:
        return True, GREETING_MESSAGE, True

    # Layer 1 — how-to keyword check
    how_to_signals = [
        "step by step", "how to make", "how do i make",
        "how do you make", "recipe", "instructions for",
        "teach me how", "give me steps", "walk me through"
    ]
    if any(signal in q for signal in how_to_signals):
        return True, BLOCKED_MESSAGE, False

    # Layer 2 — Bedrock guardrail
    guardrail_id      = os.getenv("GUARDRAIL_ID")
    guardrail_version = os.getenv("GUARDRAIL_VERSION", "1")

    if not guardrail_id:
        return False, "", False

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
            return True, BLOCKED_MESSAGE, False
    except Exception:
        return False, "", False

    return False, "", False


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

# ─── Watermark — only when chat is empty ─────────────────
if not st.session_state.messages:
    render_watermark()

# ─── Chat history ─────────────────────────────────────────
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        # Render with pills if answer contains citation markers
        content = message["content"]
        sources = message.get("sources", [])
        if sources and "[Source" in content:
            st.markdown(
                replace_citations_with_pills(content, sources),
                unsafe_allow_html=True
            )
        else:
            st.markdown(content)
        if sources:
            render_source_cards(sources)

# ─── Chat input ───────────────────────────────────────────
if prompt := st.chat_input("Ask anything about the book..."):

    # Hide watermark once conversation starts
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    blocked, blocked_message, is_greeting = is_blocked_by_guardrail(prompt)

    if blocked:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            render_blocked(blocked_message, is_greeting=is_greeting)
        st.session_state.messages.append({
            "role": "assistant",
            "content": blocked_message
        })

    else:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            try:
                chat_history = st.session_state.messages[-6:] if st.session_state.messages else None

                # ── Pipeline status — no st.status(), no white box ──
                status_placeholder = st.empty()
                steps = [
                    "Searching with keywords...",
                    "Searching by meaning...",
                    "Combining results...",
                    "Reranking passages...",
                    "Composing answer...",
                ]
                for step in steps:
                    status_placeholder.markdown(
                        f'<div class="pipeline-status">{step}</div>',
                        unsafe_allow_html=True
                    )
                    if step != steps[-2]:  # don't sleep before retrieve
                        time.sleep(0.35)
                    if step == steps[2]:
                        chunks, confidence = retrieve(prompt, vs, bm25, docs, metas)

                status_placeholder.empty()

                # ── Stream answer ──────────────────────────────────
                answer_placeholder = st.empty()
                full_answer        = ""
                final_sources      = []

                for token, sources in stream_answer_with_citations(
                    prompt, chunks, llm, chat_history=chat_history, confidence=confidence
                ):
                    if token is not None:
                        full_answer += token
                        answer_placeholder.markdown(full_answer + "▌")
                    else:
                        final_sources = sources

                # ── Final render with citation pills ───────────────
                is_blocked_output = (
                    "I can only answer questions about A Court of Mist and Fury"
                    in full_answer
                )

                if not is_blocked_output and final_sources:
                    answer_placeholder.markdown(
                        replace_citations_with_pills(full_answer, final_sources),
                        unsafe_allow_html=True
                    )
                    render_source_cards(final_sources)
                else:
                    answer_placeholder.markdown(full_answer)

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

with st.sidebar:
    st.divider()
    st.markdown(
        '<p style="font-family: Cinzel, serif; font-size: 0.55rem; '
        'letter-spacing: 0.12em; text-transform: uppercase; '
        'color: #a07020; text-align: center;">Ask My Docs · ACOTAR · 2026</p>',
        unsafe_allow_html=True
    )