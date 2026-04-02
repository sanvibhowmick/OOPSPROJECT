"""
llm.py
──────
All LLM interactions using LangChain's ChatOllama and LCEL (LangChain
Expression Language) chains.

LangChain components used
  ─ langchain_ollama.ChatOllama          local Ollama chat model
  ─ langchain_core.prompts               ChatPromptTemplate
  ─ langchain_core.output_parsers        StrOutputParser
  ─ langchain_core.runnables             pipe operator  (|)

Public API
  chat()               – raw prompt → string helper
  decompose_query()    – complex question → list[sub_query]
  answer_sub_query()   – sub_query + chunks → answer string
  aggregate_answers()  – sub Q&As → final answer string
"""

from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.document_store import Chunk


# ─── Shared LLM factory ───────────────────────────────────────────────────────

def _make_llm(model: str = OLLAMA_MODEL) -> ChatOllama:
    """Return a ChatOllama instance pointing at the local Ollama server."""
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL)


# ─── Generic chat helper ──────────────────────────────────────────────────────

def chat(system: str, user: str, model: str = OLLAMA_MODEL) -> str:
    """
    Send a (system, user) message pair and return the assistant reply as a
    plain string.  Useful for one-off calls outside of a chain.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("human",  "{user}"),
    ])
    chain = prompt | _make_llm(model) | StrOutputParser()
    return chain.invoke({"system": system, "user": user}).strip()


# ─── Pipeline steps ───────────────────────────────────────────────────────────

# ── Step 1: decompose ─────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """\
You are a query decomposition expert.
Break the complex, multi-hop question below into a numbered list of simple,
atomic sub-questions that together cover the full original question.

Rules:
  - Output ONLY the numbered list, nothing else.
  - Each line format: "1. <sub-question>"
  - Produce 2–5 sub-questions, no more.
  - Each sub-question must be self-contained and independently answerable.\
"""

_decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", _DECOMPOSE_SYSTEM),
    ("human",  "Complex question:\n{query}"),
])


def decompose_query(query: str, model: str = OLLAMA_MODEL) -> list[str]:
    """
    Use ChatOllama to decompose *query* into 2-5 atomic sub-questions.
    Returns a list of sub-query strings.
    """
    chain  = _decompose_prompt | _make_llm(model) | StrOutputParser()
    raw    = chain.invoke({"query": query})

    # Parse "1. …", "2) …" lines
    result: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line.strip())
        if m:
            result.append(m.group(1).strip())

    # Fallback: treat every non-empty line as a sub-query
    if not result:
        result = [l.strip() for l in raw.splitlines() if l.strip()]

    return result[:5]   # hard cap at 5


# ── Step 2: answer sub-query ──────────────────────────────────────────────────

_ANSWER_SYSTEM = """\
You are a precise question-answering assistant.
Answer the question using ONLY the provided context chunks.
Be concise (1–3 sentences).
If the context does not contain enough information, say "Insufficient context."\
"""

_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", _ANSWER_SYSTEM),
    ("human",  "Context:\n{context}\n\nQuestion: {question}"),
])


def answer_sub_query(
    sub_query: str,
    chunks:    list[Chunk],
    model:     str = OLLAMA_MODEL,
) -> str:
    """Answer *sub_query* using only the provided *chunks* as context."""
    context = "\n\n".join(
        f"[Chunk {i+1} · doc={c.doc_id}]\n{c.text}"
        for i, c in enumerate(chunks)
    )
    chain = _answer_prompt | _make_llm(model) | StrOutputParser()
    return chain.invoke({"context": context, "question": sub_query}).strip()


# ── Step 3: aggregate ─────────────────────────────────────────────────────────

_AGGREGATE_SYSTEM = """\
You are a synthesis expert.
You receive a complex question, several sub-questions, and their individual answers.
Write ONE coherent, concise final answer to the complex question.
Base your answer only on the provided sub-answers — do not add new facts.
Output the final answer as plain prose with no preamble, JSON, or bullet points.\
"""

_aggregate_prompt = ChatPromptTemplate.from_messages([
    ("system", _AGGREGATE_SYSTEM),
    ("human",  "Complex question:\n{original_query}\n\n{pairs}"),
])


def aggregate_answers(
    original_query: str,
    sub_queries:    list[str],
    sub_answers:    list[str],
    model:          str = OLLAMA_MODEL,
) -> str:
    """
    Synthesise sub-question/answer pairs into a single coherent final answer.
    Returns the answer string; the confidence score is derived separately from
    the aggregated cosine similarities computed in pipeline.py.
    """
    pairs = "\n".join(
        f"Sub-question {i+1}: {sq}\nAnswer {i+1}: {sa}"
        for i, (sq, sa) in enumerate(zip(sub_queries, sub_answers))
    )
    chain = _aggregate_prompt | _make_llm(model) | StrOutputParser()
    return chain.invoke({
        "original_query": original_query,
        "pairs":          pairs,
    }).strip()