from __future__ import annotations
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

from src.config import HF_LLM_MODEL
from src.document_store import Chunk

load_dotenv()
_HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # CHANGED: read once at module level


def _make_llm(model: str = HF_LLM_MODEL) -> ChatHuggingFace:
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        max_new_tokens=512,
        huggingfacehub_api_token=_HF_TOKEN,  # CHANGED: pass token explicitly
    )
    return ChatHuggingFace(llm=endpoint)


# ── Decompose Step ──

_DECOMPOSE_SYSTEM = """\
You are a query decomposition expert.
Break the complex, multi-hop question below into a numbered list of simple,
atomic sub-questions that together cover the full original question.

Rules:
  - Output ONLY the numbered list, nothing else.
  - Each line format: "1. <sub-question>"
  - Produce 2-5 sub-questions, no more.
  - Minimize redundancy: Ensure each sub-question targets a distinct piece of information to prevent overlapping queries and duplicate retrieval efforts
  - Each sub-question must be self-contained and independently answerable.\
"""

_decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", _DECOMPOSE_SYSTEM),
    ("human",  "Complex question:\n{query}"),
])

def decompose_query(query: str, model: str = HF_LLM_MODEL) -> list[str]:
    chain = _decompose_prompt | _make_llm(model) | StrOutputParser()
    raw = chain.invoke({"query": query})

    result: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line.strip())
        if m:
            result.append(m.group(1).strip())

    if not result:
        result = [l.strip() for l in raw.splitlines() if l.strip()]

    return result[:5]


# ── Answer Sub-Query Step ──

_ANSWER_SYSTEM = """\
You are a precise question-answering assistant.
Answer the question using ONLY the provided context chunks.
Be concise (1-3 sentences).
If the context does not contain enough information, say "Insufficient context."\
"""

_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", _ANSWER_SYSTEM),
    ("human",  "Context:\n{context}\n\nQuestion: {question}"),
])

def answer_sub_query(sub_query: str, chunks: list[Chunk], model: str = HF_LLM_MODEL) -> str:
    context = "\n\n".join(f"[Chunk {i+1} · doc={c.doc_id}]\n{c.text}" for i, c in enumerate(chunks))
    chain = _answer_prompt | _make_llm(model) | StrOutputParser()
    return chain.invoke({"context": context, "question": sub_query}).strip()


# ── Aggregate Step ──

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

def aggregate_answers(original_query: str, sub_queries: list[str], sub_answers: list[str], model: str = HF_LLM_MODEL) -> str:
    pairs = "\n".join(f"Sub-question {i+1}: {sq}\nAnswer {i+1}: {sa}" for i, (sq, sa) in enumerate(zip(sub_queries, sub_answers)))
    chain = _aggregate_prompt | _make_llm(model) | StrOutputParser()
    return chain.invoke({"original_query": original_query, "pairs": pairs}).strip()