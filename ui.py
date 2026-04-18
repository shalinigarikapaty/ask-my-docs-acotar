"""
ui.py — Dawn Court light theme, Ask My Docs ACOTAR
Readability fix: darker ink, stronger contrast on cream background.
"""

import re
import streamlit as st
from config import AUTHOR_NAME


# ─── Citation pill post-processor ────────────────────────
def replace_citations_with_pills(answer: str, sources: list) -> str:
    def replace_match(m):
        n = int(m.group(1)) - 1
        if 0 <= n < len(sources):
            page = sources[n]["page"]
            return f'<span class="citation-pill">p.{page}</span>'
        return m.group(0)
    return re.sub(r'\[Source (\d+)\]', replace_match, answer)


# ─── Theme injection ──────────────────────────────────────
def inject_theme():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Lora:ital,wght@0,400;0,500;1,400&family=Cinzel:wght@400;600&display=swap" rel="stylesheet">

    <style>
    :root {
        --cream:         #fdf6ec;
        --parchment:     #e8d5b0;
        --parchment-mid: #d4bc96;
        --ink:           #1a0a00;
        --ink-dim:       #5a3a20;
        --gold:          #8a5500;
        --gold-dim:      #6b3f00;
        --gold-light:    #f0c870;
        --rose:          #8b2030;
        --border:        rgba(120,80,20,0.25);
        --border-mid:    rgba(120,80,20,0.5);
        --shadow:        rgba(26,10,0,0.08);
    }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--cream) !important;
        color: var(--ink) !important;
    }

    [data-testid="stAppViewContainer"] {
        background-image:
            radial-gradient(ellipse at 20% 0%, rgba(138,85,0,0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 100%, rgba(139,32,48,0.03) 0%, transparent 50%);
    }

    [data-testid="stHeader"] {
        background-color: var(--cream) !important;
        border-bottom: 1px solid var(--border) !important;
    }

    /* ── Body text — strong contrast, leaf nodes only ─── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    .stMarkdown p {
        font-family: 'Lora', Georgia, serif !important;
        font-size: 1.02rem !important;
        line-height: 1.85 !important;
        color: #1a0a00 !important;
    }

    /* ── Sidebar ───────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--parchment) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--ink) !important;
        font-family: 'Lora', Georgia, serif !important;
    }

    /* ── Title ─────────────────────────────────────────── */
    .dawn-title {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        animation: fadeInDown 0.7s ease both;
    }

    .dawn-title-text {
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 3.2rem;
        font-weight: 700;
        font-style: italic;
        color: #6b3f00;
        text-shadow: 0 1px 2px rgba(0,0,0,0.12);
        margin: 0;
        letter-spacing: 0.02em;
        line-height: 1.1;
    }

    .dawn-subtitle {
        font-family: 'Cinzel', serif;
        font-size: 0.65rem;
        letter-spacing: 0.28em;
        color: var(--ink-dim);
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    .dawn-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1rem 0 1.5rem;
    }

    /* ── Watermark ─────────────────────────────────────── */
    .chat-watermark {
        text-align: center;
        padding: 3rem 2rem;
        animation: fadeInUp 1s ease both;
    }

    .chat-watermark-text {
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 1.3rem;
        font-style: italic;
        color: var(--parchment-mid);
        letter-spacing: 0.03em;
        line-height: 1.6;
    }

    .chat-watermark-sig {
        font-family: 'Cinzel', serif;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        color: var(--parchment-mid);
        margin-top: 0.5rem;
        text-transform: uppercase;
    }

    /* ── Chat messages ─────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.4rem 0 !important;
        animation: fadeInUp 0.35s ease both;
    }

    /* ── Citation pills ────────────────────────────────── */
    .citation-pill {
        display: inline-block;
        background: var(--gold-light);
        color: var(--ink);
        font-family: 'Cinzel', serif;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.1rem 0.45rem;
        border-radius: 20px;
        border: 1px solid var(--gold);
        vertical-align: middle;
        margin: 0 0.15rem;
        cursor: default;
        white-space: nowrap;
    }

    /* ── Source cards ──────────────────────────────────── */
    .source-cards-wrap {
        margin-top: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
        animation: fadeInUp 0.5s ease both;
    }

    .source-card {
        background: var(--parchment);
        border: 1px solid var(--border);
        border-left: 3px solid var(--gold);
        border-radius: 6px;
        padding: 0.65rem 1rem;
        box-shadow: 0 1px 4px var(--shadow);
    }

    .source-card-label {
        font-family: 'Cinzel', serif;
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        color: var(--gold-dim);
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .source-card-text {
        font-family: 'Lora', Georgia, serif;
        font-size: 0.9rem;
        font-style: italic;
        color: var(--ink-dim);
        line-height: 1.65;
    }

    /* ── Pipeline status ───────────────────────────────── */
    .pipeline-status {
        font-family: 'Lora', Georgia, serif;
        font-size: 0.92rem;
        font-style: italic;
        color: var(--ink-dim);
        padding: 0.5rem 0;
        animation: pulse 1.5s ease infinite;
    }

    /* ── Chat input ────────────────────────────────────── */
    [data-testid="stChatInputContainer"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 999 !important;
        background: linear-gradient(to top, var(--cream) 75%, transparent) !important;
        padding: 1rem 2rem !important;
    }

    [data-testid="stChatInputContainer"] textarea {
        background-color: #fff !important;
        color: var(--ink) !important;
        font-family: 'Lora', Georgia, serif !important;
        font-size: 1rem !important;
        border: 1px solid var(--border-mid) !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 4px var(--shadow) !important;
        outline: none !important;
        caret-color: var(--gold) !important;
    }

    [data-testid="stChatInputContainer"] textarea:focus,
    [data-testid="stChatInputContainer"] textarea:focus-visible,
    [data-testid="stChatInputContainer"] > div:focus-within,
    [data-testid="stChatInputContainer"] > div > div:focus-within {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 2px rgba(138,85,0,0.18) !important;
        outline: none !important;
    }

    [data-testid="stChatInputContainer"] textarea::placeholder {
        color: var(--ink-dim) !important;
        font-style: italic !important;
    }

    /* ── Metrics ───────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--cream) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        box-shadow: 0 1px 3px var(--shadow) !important;
    }

    [data-testid="stMetricLabel"] p {
        font-family: 'Cinzel', serif !important;
        font-size: 0.6rem !important;
        letter-spacing: 0.12em !important;
        color: var(--ink-dim) !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--gold-dim) !important;
    }

    /* ── Buttons ───────────────────────────────────────── */
    [data-testid="stButton"] > button {
        background: transparent !important;
        border: 1px solid var(--border-mid) !important;
        color: var(--ink-dim) !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.1em !important;
        border-radius: 4px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stButton"] > button:hover {
        border-color: var(--gold) !important;
        color: var(--gold-dim) !important;
        background: rgba(138,85,0,0.06) !important;
    }

    /* ── HR ────────────────────────────────────────────── */
    hr { border-color: var(--border) !important; }

    /* ── Caption ───────────────────────────────────────── */
    [data-testid="stCaptionContainer"] p {
        color: var(--ink-dim) !important;
        font-style: italic !important;
        font-family: 'Lora', Georgia, serif !important;
        font-size: 0.88rem !important;
    }

    /* ── Blocked / greeting ────────────────────────────── */
    .blocked-msg {
        font-family: 'Lora', Georgia, serif;
        font-style: italic;
        color: var(--rose);
        font-size: 1rem;
        line-height: 1.75;
    }

    .greeting-msg {
        font-family: 'Lora', Georgia, serif;
        font-style: italic;
        color: var(--gold-dim);
        font-size: 1rem;
        line-height: 1.75;
    }

    /* ── Sidebar credit ────────────────────────────────── */
    .sidebar-credit {
        font-family: 'Lora', Georgia, serif;
        font-size: 0.78rem;
        font-style: italic;
        color: var(--ink-dim);
        text-align: center;
        line-height: 1.7;
    }

    /* ── Padding for fixed input ───────────────────────── */
    .main .block-container {
        padding-bottom: 140px !important;
    }

    /* ── Animations ────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50%       { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Title ───────────────────────────────────────────────
def render_title():
    st.markdown("""
    <div class="dawn-title">
        <div class="dawn-title-text">Ask My Docs</div>
        <div class="dawn-subtitle">A Court of Mist and Fury &nbsp;&middot;&nbsp; Sarah J. Maas</div>
    </div>
    <hr class="dawn-divider">
    """, unsafe_allow_html=True)


# ─── Watermark ────────────────────────────────────────────
def render_watermark():
    st.markdown("""
    <div class="chat-watermark">
        <div class="chat-watermark-text">
            For the readers who never forget a detail.
        </div>
        <div class="chat-watermark-sig">— S.G.</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Source cards ─────────────────────────────────────────
def render_source_cards(sources: list):
    if not sources:
        return
    cards_html = '<div class="source-cards-wrap">'
    for s in sources:
        cards_html += f"""
        <div class="source-card">
            <div class="source-card-label">{s['label']} &nbsp;&middot;&nbsp; Page {s['page']}</div>
            <div class="source-card-text">{s['text'][:280]}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


# ─── Blocked / greeting ───────────────────────────────────
def render_blocked(message: str, is_greeting: bool = False):
    css_class = "greeting-msg" if is_greeting else "blocked-msg"
    st.markdown(
        f'<div class="{css_class}">{message}</div>',
        unsafe_allow_html=True
    )


# ─── Sidebar ─────────────────────────────────────────────
def render_sidebar(messages: list):
    with st.sidebar:
        st.markdown(
            '<p style="font-family: Cinzel, serif; font-size: 0.72rem; '
            'letter-spacing: 0.2em; text-transform: uppercase; '
            'color: #6b3f00; margin-bottom: 1rem;">The Archive</p>',
            unsafe_allow_html=True
        )

        total = len([m for m in messages if m["role"] == "user"])
        st.metric("Questions Asked", total)
        st.metric("Chunks per Answer", 5)
        st.metric("Retrieval", "Hybrid + Rerank")

        st.divider()

        st.caption(
            "Every answer is drawn from the pages of the book itself. "
            "Citations reveal exactly where the truth lives."
        )

        st.divider()

        if AUTHOR_NAME:
            st.markdown(
                f'<div class="sidebar-credit">'
                f'A reader\'s guide, kept by<br>'
                f'<strong>{AUTHOR_NAME}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.divider()

        if st.button("Clear Conversation"):
            return True

    return False