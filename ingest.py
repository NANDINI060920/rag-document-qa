"""
ingest.py
Loads all PDFs from the docs/ folder, splits them into chunks,
embeds them, and stores them in a local ChromaDB vector store.

Run this once whenever you add/change documents:
    python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_documents():
    """Load every PDF in the docs/ folder, tagging each chunk with its source filename and page."""
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {DOCS_DIR}/")
    return documents


def split_documents(documents):
    """Split documents into overlapping chunks so context isn't cut off mid-thought."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks):
    """Embed all chunks and persist them to a local Chroma DB."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store built and saved to {CHROMA_DIR}/")
    return vector_store


if __name__ == "__main__":

    if not os.path.isdir(DOCS_DIR) or not os.listdir(DOCS_DIR):
        raise SystemExit(
            f"No documents found. Add PDFs to the {DOCS_DIR}/ folder first."
        )

    docs = load_documents()
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print("\nDone! Now run: streamlit run app.py")