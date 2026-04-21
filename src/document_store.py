from __future__ import annotations
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from qdrant_client.models import Distance
from dotenv import load_dotenv
import os
import time
from typing import List

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HF_EMBED_MODEL,
    TOP_K_CHUNKS,
)
from src.utils import console


# ──────────────────────────────────────────────────────────────────
#  Batched embedding wrapper
#
#  Problem:  SemanticChunker sends ALL sentences in one embed call.
#            Large docs → payload too big / HF rate-limit.
#
#  Fix:     Split into fixed-size batches (EMBED_BATCH_SIZE).
# ──────────────────────────────────────────────────────────────────

EMBED_BATCH_SIZE = 32   # sentences per HTTP call — tune per HF tier
EMBED_RETRY      = 3    # retries per batch on transient failure
EMBED_BACKOFF    = 2.0  # base back-off seconds (multiplied by attempt#)


def _embed_batch_with_retry(
    base: HuggingFaceEndpointEmbeddings,
    batch: List[str],
    batch_num: int,
    n_batches: int,
) -> List[List[float]]:
    """Embed one batch, retrying on transient errors."""
    for attempt in range(1, EMBED_RETRY + 1):
        try:
            result = base.embed_documents(batch)
            console.print(
                f"    [dim]batch {batch_num}/{n_batches} "
                f"({len(batch)} sentences) ✓[/dim]"
            )
            return result
        except Exception as exc:
            if attempt == EMBED_RETRY:
                raise RuntimeError(
                    f"Embedding batch {batch_num} failed after "
                    f"{EMBED_RETRY} attempts: {exc}"
                ) from exc
            wait = EMBED_BACKOFF * attempt
            console.print(
                f"    [yellow]batch {batch_num} attempt {attempt} failed "
                f"({exc}), retrying in {wait:.0f}s…[/yellow]"
            )
            time.sleep(wait)


class BatchedEmbeddings:
    """
    Wraps HuggingFaceEndpointEmbeddings with fixed-size batching
    to avoid oversized payloads and HF rate-limits.

    All other attributes/methods forward to the base object so
    SemanticChunker and QdrantVectorStore need no changes.
    """

    def __init__(
        self,
        base: HuggingFaceEndpointEmbeddings,
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> None:
        self._base       = base
        self._batch_size = batch_size

    # ── transparent passthrough ───────────────────────────────────
    def __getattr__(self, name: str):
        return getattr(self._base, name)

    # ── single query (retrieval path — no batching needed) ────────
    def embed_query(self, text: str) -> List[float]:
        return self._base.embed_query(text)

    # ── sequential batched embed (SemanticChunker + Qdrant indexing) ──
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        total     = len(texts)
        n_batches = (total + self._batch_size - 1) // self._batch_size
        slices    = [
            texts[i : i + self._batch_size]
            for i in range(0, total, self._batch_size)
        ]

        console.print(
            f"  [dim]Embedding [bold]{total}[/bold] sentences → "
            f"{n_batches} batches of ≤{self._batch_size}[/dim]"
        )

        results: List[List[List[float]]] = []
        for idx, batch in enumerate(slices):
            results.append(
                _embed_batch_with_retry(self._base, batch, idx + 1, n_batches)
            )

        # Flatten: [[vec, vec, …], [vec, vec, …], …] → [vec, vec, …]
        return [vec for batch_result in results for vec in batch_result]




@dataclass
class Chunk:
    doc_id: str
    text: str
    index: int



class DocumentStore:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        embed_model: str = HF_EMBED_MODEL,
        embed_batch_size: int = EMBED_BATCH_SIZE,
    ) -> None:
        load_dotenv()
        self.qdrant_url     = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        hf_token            = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        _base_embeddings = HuggingFaceEndpointEmbeddings(
            model=embed_model,
            huggingfacehub_api_token=hf_token,
        )

        self._embeddings_fn = BatchedEmbeddings(_base_embeddings, batch_size=embed_batch_size)

        self._splitter = SemanticChunker(
            embeddings=self._embeddings_fn,
            breakpoint_threshold_type="percentile",
        )

        self._vectorstore: QdrantVectorStore | None = None
        self._chunks: list[Chunk] = []

    

    def add_document(self, doc_id: str, text: str) -> None:
        raw_chunks = self._splitter.split_text(text.strip())
        for idx, chunk_text in enumerate(raw_chunks):
            self._chunks.append(Chunk(doc_id=doc_id, text=chunk_text.strip(), index=idx))

    def build_index(self) -> None:
        if not self._chunks:
            raise ValueError("No documents staged — call add_document() first.")

        console.print(
            f"  [dim]Uploading [bold]{len(self._chunks)}[/bold] chunks to Qdrant Cloud…[/dim]"
        )

        lc_docs = [
            Document(
                page_content=c.text,
                metadata={"doc_id": c.doc_id, "index": c.index},
            )
            for c in self._chunks
        ]

        self._vectorstore = QdrantVectorStore.from_documents(
            documents=lc_docs,
            embedding=self._embeddings_fn,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name="rag_documents",
            force_recreate=True,
            distance=Distance.COSINE,
        )

        console.print(f"  [dim][green]✓[/green] Qdrant Cloud Index ready.[/dim]")

  
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS,
    ) -> tuple[list[Chunk], list[float]]:
        if self._vectorstore is None:
            self._vectorstore = QdrantVectorStore.from_existing_collection(
                embedding=self._embeddings_fn,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name="rag_documents",
            )

        hits = self._vectorstore.similarity_search_with_score(query, k=top_k)

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