"""
rag_chain.py
Core retrieval-augmented generation logic: given a question, retrieve
relevant chunks from ChromaDB and ask the LLM to answer using only that
context, citing sources.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

CHROMA_DIR = "chroma_db"
TOP_K = 4  # how many chunks to retrieve per question

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context from the user's documents. Follow these rules strictly:

1. If the answer is not contained in the context, say "I don't have enough
   information in the documents to answer that." Do not make anything up.
2. Always cite which source and page each fact comes from, like [source.pdf, p.3].
3. Be concise and direct. Don't repeat the question back.
4. If the context is contradictory, point that out rather than picking one side.

Context:
{context}
"""


def get_vector_store():
    if not os.path.isdir(CHROMA_DIR):
        raise FileNotFoundError(
            "No vector store found. Run `python ingest.py` first to index your documents."
        )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def format_context(retrieved_docs):
    """Turn retrieved chunks into a labeled context block the LLM can cite from."""
    blocks = []
    for doc in retrieved_docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{source}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, chat_history=None):
    """
    Run the full RAG pipeline for one question.
    Returns (answer_text, retrieved_docs) so the UI can show sources.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)

    context = format_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content, retrieved_docs