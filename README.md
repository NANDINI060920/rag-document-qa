# Chat with Your Docs — RAG Q&A App

A Retrieval-Augmented Generation (RAG) app that lets you ask natural-language
questions about your own PDF documents and get answers grounded in their
content, with source citations — instead of relying on an LLM's raw
(and potentially hallucinated) knowledge.

## How it works

1. **Ingestion** (`ingest.py`): PDFs in `docs/` are loaded, split into
   overlapping chunks, embedded with OpenAI's `text-embedding-3-small`,
   and stored in a local ChromaDB vector store.
2. **Retrieval** (`rag_chain.py`): When you ask a question, it's embedded
   and used to find the most semantically similar chunks in the vector store.
3. **Generation**: The retrieved chunks are passed to `gpt-4o-mini` as
   context, with a system prompt instructing it to answer *only* from that
   context and cite sources — reducing hallucination.
4. **UI** (`app.py`): A Streamlit chat interface shows the conversation and
   lets you expand each answer to see exactly which document/page it came from.

## Setup

```bash
# 1. Clone and enter the project
cd rag-app

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# then edit .env and paste your OpenAI key

# 5. Add documents
# Drop one or more PDFs into the docs/ folder

# 6. Build the vector store
python ingest.py

# 7. Launch the app
streamlit run app.py
```

## Project structure

```
rag-app/
├── docs/              # Put your PDFs here
├── ingest.py          # Builds the vector store from docs/
├── rag_chain.py        # Retrieval + generation logic
├── app.py             # Streamlit chat UI
├── requirements.txt
├── .env.example
└── README.md
```

## Deploying

Push this repo to GitHub, then deploy for free on either:
- **Streamlit Community Cloud** (streamlit.io/cloud) — point it at `app.py`,
  add `OPENAI_API_KEY` as a secret.
- **Hugging Face Spaces** — create a Streamlit Space, upload these files,
  add the key as a Space secret.

Note: `chroma_db/` is generated locally. For a public deploy, either commit
a small pre-built `chroma_db/` for a demo document set, or add an in-app
PDF uploader that runs ingestion on the fly (a great v2 feature to add).

## Ideas to extend this (great talking points for interviews)

- **Source highlighting**: highlight the exact sentence used, not just the chunk.
- **Hybrid search**: combine keyword (BM25) + semantic search for better recall.
- **Multi-document comparison**: "How does document A differ from document B on X?"
- **Streaming responses**: stream tokens instead of waiting for the full answer.
- **Re-ranking**: use a cross-encoder to re-rank retrieved chunks before generation.
- **Evaluation**: build a small test set of Q&A pairs and measure answer accuracy.

## Tech stack

Python · LangChain · ChromaDB · OpenAI (embeddings + gpt-4o-mini) · Streamlit
