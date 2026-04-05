import streamlit as st
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory


st.set_page_config(page_title="ACOTAR Chatbot", page_icon="🌹")
st.title("Chat with A Court of Thorns and Roses")

@st.cache_resource
def load_chain():
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatBedrock(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return chain

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about ACOTAR..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the book..."):
            response = chain({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})