Multi-Hop RAG Pipeline

A local, privacy-first Retrieval-Augmented Generation (RAG) pipeline designed to answer complex, multi-part questions. Instead of trying to answer a complicated query in a single shot, this pipeline uses an LLM to decompose the question into atomic sub-queries, fetches relevant context for each, and synthesizes a final comprehensive answer.

Powered by LangChain, Ollama, and Qdrant, with a beautiful terminal UI using Rich.

✨ Features

Multi-Hop Reasoning: Breaks down complex queries into simple, independently answerable sub-queries.

100% Local & Private: Uses Ollama for both LLM generation and text embeddings. No API keys required, and your data never leaves your machine.

Qdrant Vector Database: Extremely fast and efficient vector search. Runs entirely in-memory (or locally on disk) with zero external database setup required.

Dynamic Data Ingestion: Simply drop your .txt files into the data/ directory and they are automatically chunked and indexed.

Interactive CLI: A beautiful, real-time terminal interface that shows you exactly what the LLM is thinking at every step.

🛠️ Prerequisites

Python 3.10+

Ollama installed and running on your machine.

Before running the application, ensure you have pulled the necessary Ollama models:

# Pull the default LLM (used for generation)
ollama pull llama3.2

# Pull the default embedding model (used for vector search)
ollama pull nomic-embed-text


🚀 Installation

Clone this repository (or download the source code).

(Optional but recommended) Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install the required Python packages:

pip install -r requirements.txt


📖 Usage

Add your data: Create a folder named data/ in the root of the project (the script will automatically create it if it doesn't exist). Place any .txt files you want to query inside this folder.

Start the pipeline:

python main.py


Ask questions: The interactive terminal will prompt you for a query. Try asking a complex question that requires connecting information from different parts of your documents!

Exit: Type exit, quit, or press Ctrl+C to stop.

🧠 How It Works (The Pipeline)

Ingestion: The DocumentStore reads your .txt files, splits them into overlapping chunks using RecursiveCharacterTextSplitter, embeds them using OllamaEmbeddings, and stores them in a local Qdrant vector database.

Decomposition (Hop 1): The user's complex query is sent to the LLM, which breaks it down into 2-5 simple sub-queries.

Retrieval & Answering (Hop 2): For each sub-query, the pipeline:

Performs a cosine similarity search in Qdrant to find the top K most relevant chunks.

Prompts the LLM to answer that specific sub-query using only the retrieved chunks.

Aggregation (Hop 3): The LLM is provided with the original complex query and the list of generated sub-answers, and is tasked with synthesizing a final, cohesive response.

⚙️ Configuration

You can tweak the behavior of the pipeline by editing src/config.py.

OLLAMA_MODEL: The LLM used for text generation (Default: llama3.2).

OLLAMA_EMBED_MODEL: The model used for generating vector embeddings (Default: nomic-embed-text).

TOP_K_CHUNKS: How many text chunks to retrieve per sub-query (Default: 3).

CHUNK_SIZE / CHUNK_OVERLAP: Text splitting parameters for your documents.

Persisting the Database

By default, the Qdrant database runs in :memory:, meaning the index is wiped when the script closes. If you have a large dataset and want to save the index to your hard drive to avoid re-embedding on every run, change the location parameter to path="./qdrant_db" in src/document_store.py.