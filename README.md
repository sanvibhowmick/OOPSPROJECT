# Multi-Hop RAG Pipeline

A fully local, free multi-hop Retrieval-Augmented Generation pipeline powered by
**LangChain** and **Ollama** — no API keys, no cloud services.

```
Complex Query ──► Decompose into Sub-Queries  (ChatOllama)
                       │
                       ▼
               Retrieve Chunks per Sub-Query   (OllamaEmbeddings + InMemoryVectorStore)
                       │
                       ▼
               Answer each Sub-Query           (ChatOllama + context)
                       │
                       ▼
               Aggregate Final Answer          (ChatOllama)
                       │
                       ▼
               Aggregated Score = mean cosine similarity across all retrieved chunks
```

---

## Project Structure

```
rag_pipeline/
├── src/
│   ├── __init__.py
│   ├── config.py            # Models, chunk size, top-k, Ollama URL
│   ├── document_store.py    # LangChain splitter + OllamaEmbeddings + InMemoryVectorStore
│   ├── llm.py               # ChatOllama chains for decompose / answer / aggregate
│   ├── pipeline.py          # Orchestrator + PipelineResult dataclass
│   └── utils.py             # Rich console + print_results()
├── data/
│   └── sample_doc.txt       # Drop your own .txt documents here
├── tests/
│   ├── __init__.py
│   ├── test_document_store.py
│   └── test_llm.py
├── main.py                  # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Prerequisites

1. **Ollama** — install from https://ollama.com and start the server:
   ```bash
   ollama serve
   ```

2. **Pull the required models:**
   ```bash
   ollama pull llama3.2          # chat / reasoning
   ollama pull nomic-embed-text  # dense embeddings
   ```

---

## Quick Start

```bash
# 1. Clone / enter the project
cd rag_pipeline

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python main.py
```

---

## Configuration

Edit `src/config.py` to change models or chunking parameters:

| Setting            | Default            | Description                        |
|--------------------|--------------------|------------------------------------|
| `OLLAMA_MODEL`     | `llama3.2`         | Chat / reasoning model             |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Dense embedding model            |
| `OLLAMA_BASE_URL`  | `http://localhost:11434` | Ollama server URL           |
| `TOP_K_CHUNKS`     | `3`                | Chunks retrieved per sub-query     |
| `CHUNK_SIZE`       | `300`              | Characters per chunk               |
| `CHUNK_OVERLAP`    | `50`               | Overlap between chunks             |

---

## Aggregated Score

The **aggregated cosine score** is the mean of every individual chunk cosine
similarity returned across all sub-queries:

```
agg_score = mean( all cosine scores from all sub-queries )
```

This is a purely retrieval-based signal — no LLM-generated confidence number.
Scores range from 0 to 1; higher means the retrieved context was semantically
closer to the queries on average.

| Score range | Interpretation          |
|-------------|-------------------------|
| ≥ 0.70      | Strong retrieval match  |
| 0.40–0.69   | Moderate match          |
| < 0.40      | Weak / off-topic match  |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests mock both `ChatOllama` and `OllamaEmbeddings` so they run without a
live Ollama server.

---

## Adding Your Own Documents

```python
# In main.py, replace SAMPLE_DOCS with file I/O:
with open("data/my_corpus.txt") as f:
    store.add_document("my_doc", f.read())
```

Or loop over a directory:

```python
import pathlib
for path in pathlib.Path("data").glob("*.txt"):
    store.add_document(path.stem, path.read_text())
```

---

## LangChain Components Used

| Component | Package | Purpose |
|-----------|---------|---------|
| `ChatOllama` | `langchain-ollama` | Local LLM chat |
| `OllamaEmbeddings` | `langchain-ollama` | Dense text embeddings |
| `RecursiveCharacterTextSplitter` | `langchain-text-splitters` | Smart chunking |
| `InMemoryVectorStore` | `langchain-core` | Cosine vector retrieval |
| `ChatPromptTemplate` | `langchain-core` | Structured prompts |
| `StrOutputParser` | `langchain-core` | LLM → plain string |