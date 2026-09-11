"""
app.py
Streamlit front-end for the RAG document Q&A app.
Visitors upload their own PDF(s) directly in the browser — no pre-loaded
documents needed, and nothing is saved permanently on the server.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Chat with Your Docs", page_icon="📄", layout="wide")

st.title("📄 Chat with Your Docs")
st.caption(
    "Upload a PDF below and ask questions about it. Answers are grounded "
    "only in what's in your document, with sources cited."
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it in your .env file (or as a deployment secret).")
    st.stop()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context from the user's uploaded document(s). Follow these rules strictly:

1. If the answer is not contained in the context, say "I don't have enough
   information in the document(s) to answer that." Do not make anything up.
2. Always cite which source and page each fact comes from, like [filename.pdf, p.3].
3. Be concise and direct. Don't repeat the question back.
4. If the context is contradictory, point that out rather than picking one side.

Context:
{context}
"""


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_index_from_uploads(uploaded_files):
    """Save uploaded PDFs to a temp folder, chunk them, and build an in-memory vector store."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

    embeddings = get_embeddings()
    if not all_chunks:
        return None
    vector_store = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
    return vector_store


def format_context(retrieved_docs):
    blocks = []
    for doc in retrieved_docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{source}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer_question(vector_store, question: str):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)
    context = format_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content, retrieved_docs


# --- Session state setup ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_filenames" not in st.session_state:
    st.session_state.indexed_filenames = []

# --- File upload ---
uploaded_files = st.file_uploader(
    "Upload one or more PDFs", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    current_names = sorted(f.name for f in uploaded_files)
    if current_names != st.session_state.indexed_filenames:
        with st.spinner("Reading and indexing your document(s)..."):
            new_store = build_index_from_uploads(uploaded_files)
            if new_store is None:
                st.error(
                    "Couldn't extract any text from that PDF. It might be a "
                    "scanned document (photos of pages) rather than one with "
                    "real text — try a different PDF where you can normally "
                    "select/copy the text."
                )
                st.stop()
            st.session_state.vector_store = new_store
            st.session_state.indexed_filenames = current_names
            st.session_state.messages = []
        st.success(f"Indexed {len(uploaded_files)} document(s). Ask away!")

if st.session_state.vector_store is None:
    st.info("Upload a PDF above to get started.")
    st.stop()

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}**, page {src['page']}")
                    st.caption(src["snippet"])

# --- New question ---
question = st.chat_input("Ask something about your document(s)...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your document(s)..."):
            try:
                answer, retrieved_docs = answer_question(st.session_state.vector_store, question)
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
        "This app answers questions using **only** the content of the PDF(s) "
        "you upload, via a Retrieval-Augmented Generation (RAG) pipeline: "
        "chunking → embeddings → vector search → grounded LLM answer. "
        "Nothing you upload is stored permanently."
    )
    if st.button("Clear chat & documents"):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.session_state.indexed_filenames = []
        st.rerun()