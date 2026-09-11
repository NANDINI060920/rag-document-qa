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

st.set_page_config(page_title="The Reading Room", page_icon="🕮", layout="wide")

# --- "Reading Room" theme: botanical archive aesthetic ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"] {
    font-family: 'Public Sans', sans-serif;
}

.reading-room-header {
    background: #16302B;
    margin: -6rem -4rem 2rem -4rem;
    padding: 3rem 4rem 2rem 4rem;
    border-bottom: 3px solid #E37A5C;
}
.reading-room-header h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.6rem;
    color: #F6EFE1;
    margin: 0;
    letter-spacing: -0.01em;
}
.reading-room-header p {
    font-family: 'Public Sans', sans-serif;
    color: #A9C4BC;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    max-width: 46ch;
}

[data-testid="stFileUploaderDropzone"] {
    background: #E8DDC7 !important;
    border: 2px dashed #4A7A6E !important;
    border-radius: 4px !important;
}

[data-testid="stChatMessage"] {
    border-radius: 6px;
    padding: 0.25rem 0.5rem;
}

[data-testid="stChatMessageAvatarAssistant"] {
    background: #4A7A6E !important;
}
[data-testid="stChatMessageAvatarUser"] {
    background: #16302B !important;
}

[data-testid="stExpander"] {
    border: 1px solid #4A7A6E !important;
    border-radius: 4px;
    background: #E8DDC7;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #16302B;
}

.stButton button {
    font-family: 'Public Sans', sans-serif;
    border-radius: 4px;
    border: 1px solid #4A7A6E;
}

[data-testid="stSidebar"] {
    background: #E8DDC7;
    border-right: 2px solid #4A7A6E;
}
[data-testid="stSidebar"] h2 {
    font-family: 'Fraunces', serif;
    color: #16302B;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="reading-room-header">
<h1>The Reading Room</h1>
<p>Upload a document. Ask it anything. Every answer is drawn only from
the page in front of it, with the exact source cited — nothing invented,
nothing assumed.</p>
</div>
""", unsafe_allow_html=True)

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
5. Use the conversation history to understand follow-up questions (like "what
   about the second one?" or "can you say more about that?"), but still only
   answer using facts found in the context below — never invent details even
   to fill in a follow-up.

Context:
{context}
"""

MAX_HISTORY_TURNS = 4  # how many previous exchanges to remember


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


def answer_question(vector_store, question: str, history=None):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # For retrieval, combine the question with recent history so follow-ups
    # like "what about the second one?" can still find the right chunks.
    if history:
        recent_text = " ".join(h["content"] for h in history[-MAX_HISTORY_TURNS:])
        retrieval_query = f"{recent_text} {question}"
    else:
        retrieval_query = question

    retrieved_docs = retriever.invoke(retrieval_query)
    context = format_context(retrieved_docs)

    messages = [("system", SYSTEM_PROMPT)]
    if history:
        for h in history[-MAX_HISTORY_TURNS:]:
            role = "human" if h["role"] == "user" else "ai"
            messages.append((role, h["content"]))
    messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(messages)
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
                    st.markdown(f"`{src['source']}, page {src['page']}`")
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
                history = st.session_state.messages[:-1]  # exclude the question just added
                answer, retrieved_docs = answer_question(
                    st.session_state.vector_store, question, history=history
                )
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
                    st.markdown(f"`{source_name}, page {page}`")
                    st.caption(snippet)
                    sources.append({"source": source_name, "page": page, "snippet": snippet})

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

with st.sidebar:
    st.header("The Catalog")
    st.markdown(
        "*How this works:*\n\n"
        "Your document is broken into passages, each one indexed by meaning "
        "rather than exact words. When you ask a question, the passages most "
        "likely to hold the answer are pulled and handed to the model — "
        "which is told to work only from what it's given.\n\n"
        "Nothing you upload is kept after your session ends."
    )
    if st.button("Clear chat & documents"):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.session_state.indexed_filenames = []
        st.rerun()