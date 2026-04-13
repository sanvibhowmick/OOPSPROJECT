# ⚡ NeuralHop — Multi-Hop RAG Engine

> Ask questions that span multiple documents. NeuralHop decomposes your query into reasoning hops, retrieves evidence from each, and synthesises a final answer — all running locally.

---

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Project Structure](#project-structure)


---

## How It Works

```
Your Query
    │
    ▼
┌─────────────────────┐
│  Query Decomposer   │  → breaks query into N atomic sub-queries
└─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼  (for each sub-query)
┌────────┐  ┌────────┐
│Retrieve│  │  LLM   │  → fetch top-k chunks → answer sub-query
└────────┘  └────────┘
         │
         ▼
┌─────────────────────┐
│  Answer Aggregator  │  → synthesise all hop answers into final response
└─────────────────────┘
```

1. **Ingest** — Upload `.txt` or `.pdf` files. They are chunked and embedded locally via Ollama.
2. **Index** — Chunks are stored in a Qdrant vector database for similarity search.
3. **Query** — Your question is split into sub-queries. Each hop retrieves relevant chunks and gets an LLM answer.
4. **Synthesise** — All hop answers are aggregated into one final response.

---

## Prerequisites

Make sure you have all of the following installed before starting.

### 1. Python 3.10+

```bash
python --version   # should be 3.10 or higher
```

Download from [python.org](https://www.python.org/downloads/) if needed.

---

### 2. Ollama (Local LLM Runner)

Ollama lets you run LLMs and embedding models entirely on your machine — no API key needed.

👉 **[Download Ollama from ollama.com](https://ollama.com/download)**

After installing, pull the models NeuralHop uses:

```bash
# LLM for reasoning and answering
ollama pull llama3

# Embedding model for vector search
ollama pull nomic-embed-text
```

Confirm Ollama is running:

```bash
ollama list   # should list your downloaded models
```

> You can swap these for any other Ollama-compatible models — just update `src/config.py`.

---

### 3. Qdrant (Vector Database)

Qdrant stores your document embeddings.:


#### Qdrant Cloud 

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster and copy your **API key** and **cluster URL**

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/sanvibhowmick/OOPSPROJECT.git
cd OOPSPROJECT

# 2. Create and activate a virtual environment
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

All settings live in **`src/config.py`**. Open it and adjust to your setup:

```python
# ── Ollama models ──────────────────────────────────────────
OLLAMA_MODEL        = "llama3"           # LLM used for reasoning
OLLAMA_EMBED_MODEL  = "nomic-embed-text" # Embedding model

# ── Chunking ───────────────────────────────────────────────
CHUNK_SIZE    = 512   # characters per chunk
CHUNK_OVERLAP = 64    # overlap between consecutive chunks


```

### Environment Variables (for Qdrant Cloud)

Create a `.env` file in the project root:

```env
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your-secret-api-key-here
```




---

## Running the App

```bash
streamlit run app.py
```

The app will open at **[http://localhost:8501](http://localhost:8501)**.

**Workflow inside the app:**
1. Upload one or more `.txt` or `.pdf` files in the left panel
2. Click **⚡ Build Index** to chunk and embed your documents
3. Type your question in the right panel
4. Click **⚡ Analyze Query** and watch the hops unfold
5. Use **🗑 Delete All Documents** to clear the index and start fresh

---

## Project Structure

```
neuralhop/
│
├── app.py                  # Streamlit UI — main entry point
│
├── src/
│   ├── config.py           # Model names, chunk settings
│   ├── document_store.py   # Chunking, embedding, Qdrant index management
│   ├── llm.py              # Query decomposition, sub-query answering, aggregation
│   └── pipeline.py         # Data classes: SubQueryResult, PipelineResult
│
├── requirements.txt        # Python dependencies
├── .env                    # ← YOUR SECRETS (never commit this)
├── .gitignore              # Excludes .env and other sensitive files
└── README.md
```

---





## Tech Stack

| Component | Tool |
|-----------|------|
| UI | [Streamlit](https://streamlit.io) |
| LLM + Embeddings | [Ollama](https://ollama.com) |
| Vector Store | [Qdrant](https://qdrant.tech) |
| PDF Parsing | PyMuPDF|

---
