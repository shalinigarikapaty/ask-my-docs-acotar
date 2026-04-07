import streamlit as st
from retriever import load_vectorstore, build_bm25_index, retrieve
from citation_chain import load_llm, answer_with_citations

# ─── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Ask My Docs — ACOTAR",
    page_icon="🌹",
    layout="wide"
)

# ─── Load components once ────────────────────────────────
@st.cache_resource
def load_components():
    print("Loading retrieval components...")
    vs                = load_vectorstore()
    bm25, docs, metas = build_bm25_index(vs)
    llm               = load_llm()
    print("All components loaded.")
    return vs, bm25, docs, metas, llm

vs, bm25, docs, metas, llm = load_components()

# ─── UI Layout ───────────────────────────────────────────
st.title("Ask My Docs")
st.caption("A Court of Mist and Fury — Sarah J. Maas")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for s in message["sources"]:
                        st.markdown(f"**{s['label']} — Page {s['page']}**")
                        st.caption(s['text'][:300])
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask anything about the book..."):

        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching the book..."):
                try:
                    chat_history    = st.session_state.messages[-6:] if st.session_state.messages else None
                    chunks          = retrieve(prompt, vs, bm25, docs, metas)
                    answer, sources = answer_with_citations(
                        prompt,
                        chunks,
                        llm,
                        chat_history=chat_history
                    )

                    st.markdown(answer)
                    with st.expander("View Sources"):
                        for s in sources:
                            st.markdown(f"**{s['label']} — Page {s['page']}**")
                            st.caption(s['text'][:300])
                            st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Something went wrong: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

with col2:
    st.subheader("Session Stats")
    total_questions = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Questions asked", total_questions)
    st.metric("Chunks per answer", 5)
    st.metric("Retrieval method", "Hybrid + Rerank")

    st.divider()
    st.subheader("About")
    st.caption("This chatbot answers questions using only the text of the book. Every answer includes page citations so you can verify the source.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()