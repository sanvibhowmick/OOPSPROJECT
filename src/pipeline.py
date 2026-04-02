"""
pipeline.py
───────────
Core data structures and the run_pipeline() orchestrator.

Aggregated score
  = mean cosine similarity across every retrieved chunk from every sub-query.
  Computed purely from embedding distances — no LLM-generated confidence number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.document_store import Chunk, DocumentStore
from src.llm import aggregate_answers, answer_sub_query, decompose_query
from src.utils import console


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SubQueryResult:
    sub_query:      str
    chunks:         list[Chunk]
    chunk_scores:   list[float]   # cosine similarity score per retrieved chunk
    llm_answer:     str
    avg_similarity: float         # mean cosine score for this sub-query


@dataclass
class PipelineResult:
    original_query: str
    sub_queries:    list[str]
    sub_results:    list[SubQueryResult]
    final_answer:   str
    agg_score:      float         # mean cosine across ALL chunks, ALL sub-queries
    elapsed_sec:    float


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_pipeline(query: str, store: DocumentStore) -> PipelineResult:
    """
    Execute the full multi-hop RAG pipeline:

      1. Decompose  – LangChain ChatOllama splits the query into sub-questions
      2. Retrieve   – OllamaEmbeddings + InMemoryVectorStore returns top-k chunks
      3. Answer     – ChatOllama answers each sub-question with retrieved context
      4. Aggregate  – ChatOllama synthesises sub-answers into a final answer
      5. Score      – agg_score = mean of ALL cosine similarities (steps 2-3)
    """
    t0 = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as prog:

        # ── 1. Decompose ──────────────────────────────────────────────────────
        task = prog.add_task("[cyan]Decomposing query with LLM…", total=None)
        sub_queries = decompose_query(query)
        prog.remove_task(task)

        # ── 2 & 3. Retrieve + Answer per sub-query ────────────────────────────
        sub_results: list[SubQueryResult] = []
        for i, sq in enumerate(sub_queries):
            task = prog.add_task(
                f"[yellow]Sub-query {i+1}/{len(sub_queries)}: retrieving & answering…",
                total=None,
            )
            chunks, scores  = store.retrieve(sq)
            llm_answer      = answer_sub_query(sq, chunks)
            avg_sim         = float(np.mean(scores)) if scores else 0.0
            sub_results.append(SubQueryResult(
                sub_query=sq,
                chunks=chunks,
                chunk_scores=scores,
                llm_answer=llm_answer,
                avg_similarity=avg_sim,
            ))
            prog.remove_task(task)

        # ── 4. Aggregate final answer ─────────────────────────────────────────
        task = prog.add_task("[green]Aggregating final answer…", total=None)
        final_answer = aggregate_answers(
            query,
            sub_queries,
            [r.llm_answer for r in sub_results],
        )
        prog.remove_task(task)

    # ── 5. Compute aggregated score ───────────────────────────────────────────
    # Flatten every cosine score from every sub-query into one list, then average.
    all_scores = [score for r in sub_results for score in r.chunk_scores]
    agg_score  = float(np.mean(all_scores)) if all_scores else 0.0

    return PipelineResult(
        original_query=query,
        sub_queries=sub_queries,
        sub_results=sub_results,
        final_answer=final_answer,
        agg_score=agg_score,
        elapsed_sec=time.perf_counter() - t0,
    )