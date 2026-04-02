from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    TOP_K_CHUNKS,
)
from src.utils import console


# ─── Thin data class (keeps the rest of the codebase model-agnostic) ──────────

@dataclass
class Chunk:
    """A single text chunk with source metadata."""
    doc_id: str
    text:   str
    index:  int


# ─── DocumentStore ────────────────────────────────────────────────────────────

class DocumentStore:
    """
    Ingests plain-text documents, splits them into overlapping chunks,
    embeds each chunk with Ollama, and serves top-k cosine retrieval.

    Typical usage::

        store = DocumentStore()
        store.add_document("doc_climate", climate_text)
        store.add_document("doc_ai", ai_text)
        store.build_index()                       # embeds all chunks
        chunks, scores = store.retrieve("query")  # cosine scores in [0, 1]
    """

    def __init__(
        self,
        chunk_size:  int = CHUNK_SIZE,
        overlap:     int = CHUNK_OVERLAP,
        embed_model: str = OLLAMA_EMBED_MODEL,
        base_url:    str = OLLAMA_BASE_URL,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            # Split on natural boundaries first, fall back to characters
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._embeddings_fn = OllamaEmbeddings(
            model=embed_model,
            base_url=base_url,
        )
        self._vectorstore: InMemoryVectorStore | None = None

        # Canonical chunk list – order is stable after build_index()
        self._chunks: list[Chunk] = []

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_document(self, doc_id: str, text: str) -> None:
        """
        Split *text* into overlapping chunks with LangChain's splitter and
        stage them for indexing.  Call build_index() when all docs are added.
        """
        raw_chunks = self._splitter.split_text(text.strip())
        for idx, chunk_text in enumerate(raw_chunks):
            self._chunks.append(Chunk(doc_id=doc_id, text=chunk_text.strip(), index=idx))

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(self) -> None:
        """
        Embed every staged chunk with OllamaEmbeddings and load them into
        an InMemoryVectorStore (cosine similarity, no external service needed).
        """
        if not self._chunks:
            raise ValueError("No documents staged — call add_document() first.")

        n_docs = len(set(c.doc_id for c in self._chunks))
        console.print(
            f"  [dim]Embedding [bold]{len(self._chunks)}[/bold] chunks "
            f"from [bold]{n_docs}[/bold] doc(s) "
            f"with [bold]{self._embeddings_fn.model}[/bold]…[/dim]"
        )

        # Convert our Chunk objects → LangChain Documents so the store can
        # hold metadata (doc_id, index) alongside each vector.
        lc_docs = [
            Document(
                page_content=c.text,
                metadata={"doc_id": c.doc_id, "index": c.index},
            )
            for c in self._chunks
        ]

        self._vectorstore = InMemoryVectorStore.from_documents(
            documents=lc_docs,
            embedding=self._embeddings_fn,
        )
        console.print(
            f"  [dim][green]✓[/green] Index ready — "
            f"{len(self._chunks)} chunks indexed.[/dim]"
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS,
    ) -> tuple[list[Chunk], list[float]]:
        """
        Embed *query* and return the top-k most similar chunks together with
        their cosine similarity scores (float in [0, 1]).
        """
        if self._vectorstore is None:
            raise RuntimeError("Call build_index() before retrieving.")

        # similarity_search_with_score on InMemoryVectorStore returns cosine scores
        hits: list[tuple[Document, float]] = (
            self._vectorstore.similarity_search_with_score(query, k=top_k)
        )

        chunks: list[Chunk] = []
        scores: list[float] = []
        for lc_doc, score in hits:
            meta = lc_doc.metadata
            chunks.append(Chunk(
                doc_id=meta["doc_id"],
                text=lc_doc.page_content,
                index=meta["index"],
            ))
            scores.append(float(score))

        return chunks, scores