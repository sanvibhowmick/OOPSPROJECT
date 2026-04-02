from __future__ import annotations
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
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

@dataclass
class Chunk:
    """A simple dataclass to represent a chunk of text instead of Pydantic."""
    doc_id: str
    text: str
    index: int


class DocumentStore:
    """
    Ingests plain-text documents, splits them into overlapping chunks,
    embeds each chunk with Ollama, and serves top-k cosine retrieval using Qdrant.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        embed_model: str = OLLAMA_EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._embeddings_fn = OllamaEmbeddings(
            model=embed_model,
            base_url=base_url,
        )
        self._vectorstore: QdrantVectorStore | None = None
        self._chunks: list[Chunk] = []

    def add_document(self, doc_id: str, text: str) -> None:
        """Splits a document and stages it for index building."""
        raw_chunks = self._splitter.split_text(text.strip())
        for idx, chunk_text in enumerate(raw_chunks):
            self._chunks.append(Chunk(doc_id=doc_id, text=chunk_text.strip(), index=idx))

    def build_index(self) -> None:
        """Builds the Qdrant index from staged chunks."""
        if not self._chunks:
            raise ValueError("No documents staged — call add_document() first.")

        n_docs = len(set(c.doc_id for c in self._chunks))
        console.print(
            f"  [dim]Embedding [bold]{len(self._chunks)}[/bold] chunks "
            f"from [bold]{n_docs}[/bold] file(s) into Qdrant DB…[/dim]"
        )

        lc_docs = [
            Document(
                page_content=c.text,
                metadata={"doc_id": c.doc_id, "index": c.index},
            )
            for c in self._chunks
        ]
        
        # Initializes Qdrant dynamically. 
        # `location=":memory:"` makes it ephemeral. Replace with `path="./qdrant_db"` to persist locally.
        self._vectorstore = QdrantVectorStore.from_documents(
            documents=lc_docs,
            embedding=self._embeddings_fn,
            location=":memory:", 
            collection_name="rag_documents"
        )
        
        console.print(f"  [dim][green]✓[/green] Qdrant Index ready.[/dim]")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS,
    ) -> tuple[list[Chunk], list[float]]:
        """Retrieves top-k chunks from Qdrant via cosine similarity."""
        if self._vectorstore is None:
            raise RuntimeError("Call build_index() before retrieving.")

        hits: list[tuple[Document, float]] = (
            self._vectorstore.similarity_search_with_score(query, k=top_k)
        )

        chunks: list[Chunk] = []
        scores: list[float] = []
        for lc_doc, score in hits:
            meta = lc_doc.metadata
            chunks.append(Chunk(
                doc_id=meta.get("doc_id", "Unknown"),
                text=lc_doc.page_content,
                index=meta.get("index", 0),
            ))
            scores.append(float(score))

        return chunks, scores