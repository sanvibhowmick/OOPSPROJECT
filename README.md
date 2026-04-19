# ⚡ NeuralHop — Multi-Hop RAG Engine

NeuralHop is a **multi-hop retrieval-augmented generation (RAG)** system with a Streamlit UI. Instead of treating your question as a single lookup, it breaks complex queries into atomic sub-questions, retrieves evidence for each one independently, answers them individually, and then synthesises everything into a single coherent response. This lets it reason across multiple documents and connect information that no single chunk would contain on its own.

---

## 🌿 Branches

This repository has two versions:

| Branch | LLM & Embeddings | Requires |
|---|---|---|
| **`api`** ← **you are here** | HuggingFace Inference API (cloud) | HF API token + Qdrant |
| `main` | Ollama (local models, runs fully offline) | Ollama installed locally + Qdrant |

> **`api` branch** uses `Qwen2.5-7B-Instruct` and `BAAI/bge-base-en-v1.5` served via the HuggingFace Inference API — no local GPU required. If you want to run everything offline without any API keys, switch to the `main` branch.

---

## How it works

```
Your query
    │
    ▼
┌─────────────────────────┐
│  Query Decomposition    │  LLM splits the question into 2–5 sub-questions
└────────────┬────────────┘
             │  (for each sub-question)
             ▼
┌─────────────────────────┐
│  Semantic Retrieval     │  Qdrant vector search finds the most relevant chunks
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Sub-Query Answering    │  LLM answers each sub-question from its retrieved chunks
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Answer Aggregation     │  LLM synthesises all sub-answers into a final response
└─────────────────────────┘
```

**Models used**
- LLM: `Qwen/Qwen2.5-7B-Instruct` (via HuggingFace Inference API)
- Embeddings: `BAAI/bge-base-en-v1.5` (via HuggingFace Inference API)
- Vector store: Qdrant Cloud (or local Qdrant)

**Chunking strategy:** Semantic chunking with a percentile breakpoint threshold — chunks are split at natural semantic boundaries rather than fixed character counts.

---

## Project structure

```
.
├── app.py                  # Streamlit UI entry point
├── requirements.txt
├── .env                    # secrets 
└── src/
    ├── config.py           # model names, chunk settings, TOP_K
    ├── document_store.py   # ingestion, embedding, Qdrant indexing & retrieval
    ├── llm.py              # decompose / answer / aggregate LangChain chains
    ├── pipeline.py         # CLI-friendly pipeline that wires everything together
    └── utils.py            # shared Rich console
```

---

## Prerequisites

- Python **3.10+**
- A [HuggingFace](https://huggingface.co/settings/tokens) account with an API token that has **Inference API** access
- A [Qdrant](https://cloud.qdrant.io/) Cloud cluster **or** Docker for a local Qdrant instance

---

## Setup

### 1 — Clone the repo and switch to the `api` branch

```bash
git clone https://github.com/your-org/neuralhop.git
cd neuralhop
git checkout api
```

### 2 — Create and activate a virtual environment

```bash
# Create the venv
python -m venv .venv

# Activate — macOS / Linux
source .venv/bin/activate

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **PDF support:** `pymupdf` is listed in `requirements.txt` and handles most PDFs well. If it fails to install on your platform, use `pdfplumber` instead — both work transparently.

### 4 — Configure environment variables

Create a `.env` file in the project root (never commit this):

```env
# HuggingFace Inference API token
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Qdrant Cloud
QDRANT_URL=https://<your-cluster-id>.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

> **Where to get these:**
> - HuggingFace token → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a token with **Inference API** access
> - Qdrant URL + API key → [cloud.qdrant.io](https://cloud.qdrant.io) — create a free cluster and copy the endpoint URL and API key from the dashboard

### 5 — Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Upload documents** — drag `.txt` or `.pdf` files into the upload panel on the left.
2. **Build the index** — click **⚡ Build Index**. Each file is semantically chunked, embedded, and uploaded to Qdrant. A chunk breakdown is shown when complete.
3. **Ask a question** — type a complex, multi-document question in the query panel and click **⚡ Analyze Query**.
4. **Explore results** — the UI shows the synthesised answer, a confidence gauge, the full reasoning chain with per-hop retrieved chunks, and a scrollable query history.

---

## Configuration

All tuneable knobs live in `src/config.py`:

| Setting | Default | Description |
|---|---|---|
| `HF_LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model for generation |
| `HF_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace model for embeddings |
| `TOP_K_CHUNKS` | `3` | Chunks retrieved per sub-query |
| `CHUNK_SIZE` | `300` | Reference size (SemanticChunker uses percentile, not fixed size) |
| `CHUNK_OVERLAP` | `50` | Reference overlap (not used by SemanticChunker directly) |

The embedding batch size (`EMBED_BATCH_SIZE = 32`) and retry behaviour (`EMBED_RETRY`, `EMBED_BACKOFF`) can be tuned at the top of `src/document_store.py`.

---
