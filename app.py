"""
app.py
Streamlit front-end for the RAG document Q&A app.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_chain import answer_question

load_dotenv()

st.set_page_config(page_title="Chat with Your Docs", page_icon="📄", layout="wide")

st.title("📄 Chat with Your Docs")
st.caption(
    "Ask questions about the documents in the `docs/` folder. "
    "Answers are grounded in the documents, with sources cited."
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Copy `.env.example` to `.env` and add your key.")
    st.stop()

if not os.path.isdir("chroma_db"):
    st.warning(
        "No vector store found yet. Add PDFs to the `docs/` folder and run "
        "`python ingest.py` in your terminal before chatting."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}**, page {src['page']}")
                    st.caption(src["snippet"])

# New question
question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                answer, retrieved_docs = answer_question(question)
            except Exception as e:
                answer = f"Something went wrong: {e}"
                retrieved_docs = []

        st.markdown(answer)

        sources = []
        if retrieved_docs:
            with st.expander("Sources"):
                for doc in retrieved_docs:
                    source_name = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    snippet = doc.page_content[:250] + "..."
                    st.markdown(f"**{source_name}**, page {page}")
                    st.caption(snippet)
                    sources.append({"source": source_name, "page": page, "snippet": snippet})

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

with st.sidebar:
    st.header("About")
    st.markdown(
        "This app answers questions using **only** the content of the PDFs "
        "in the `docs/` folder, via a Retrieval-Augmented Generation (RAG) "
        "pipeline: chunking → embeddings → vector search → grounded LLM answer."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()