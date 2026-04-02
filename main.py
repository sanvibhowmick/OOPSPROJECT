"""
main.py
───────
Entry point for the Multi-Hop RAG Pipeline.

Run:
    python main.py
"""

from rich.panel import Panel

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, OLLAMA_EMBED_MODEL, OLLAMA_MODEL
from src.document_store import DocumentStore
from src.pipeline import run_pipeline
from src.utils import console, print_results

# ─── Sample corpus ────────────────────────────────────────────────────────────
# In production, replace this with file I/O from the data/ directory, e.g.:
#   with open("data/sample_doc.txt") as f:
#       store.add_document("sample", f.read())

SAMPLE_DOCS: dict[str, str] = {
    "doc_climate": """
Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural,
but since the 1800s, human activities have been the main driver, primarily due to the burning of fossil fuels
like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped
around the Earth, trapping the sun's heat and raising temperatures. The main greenhouse gases are carbon dioxide
and methane. These come from using gasoline for driving cars or coal for heating buildings, for example.
Clearing land and forests can also release carbon dioxide. Agriculture, oil and gas operations are major sources
of methane emissions. Energy, industry, transport, buildings, agriculture and land use are among the main sectors
causing greenhouse gases. The effects of climate change include severe droughts, water scarcity, severe fires,
rising sea levels, flooding, melting polar ice, catastrophic storms and declining biodiversity.
    """,
    "doc_ai": """
Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer
systems. Specific applications of AI include expert systems, natural language processing (NLP), speech recognition
and machine vision. Machine learning is a subset of AI that gives computers the ability to learn without being
explicitly programmed. Deep learning is part of a broader family of machine learning methods based on artificial
neural networks with representation learning. Large language models (LLMs) are advanced AI systems trained on
vast text datasets that can generate human-like text, answer questions, and perform reasoning tasks.
AI is being applied across industries: in healthcare for disease diagnosis, in finance for fraud detection,
in transportation for autonomous vehicles, and in climate science for weather prediction and climate modelling.
Ethical concerns around AI include bias, privacy, job displacement and the potential for misuse.
    """,
    "doc_energy": """
Renewable energy comes from naturally replenishing sources such as sunlight, wind, rain, tides, waves, and
geothermal heat. Solar power uses photovoltaic panels or concentrated solar systems to convert sunlight to
electricity. Wind energy captures kinetic energy from wind using turbines. Hydropower generates electricity
from the flow of water. Geothermal energy taps heat from the Earth's interior. Renewable energy is crucial
for decarbonising the global economy. The cost of solar and wind energy has fallen dramatically by over 90%
in the past decade, making them now cheaper than fossil fuels in most of the world. In 2023, renewable energy
accounted for over 30% of global electricity generation. Battery storage and smart grids are key technologies
enabling higher penetration of intermittent renewables into power systems.
    """,
    "doc_health": """
The human immune system is the body's defense against infections and diseases. It consists of physical barriers
such as skin, cellular components including white blood cells, and molecular elements such as antibodies.
Vaccines work by introducing a weakened or inactivated form of a pathogen, or a piece of it (like a protein),
to stimulate the immune system to build memory without causing disease. mRNA vaccines, such as those developed
for COVID-19, instruct cells to produce a protein that triggers an immune response. Climate change is increasingly
affecting human health: rising temperatures expand the range of disease vectors like mosquitoes, leading to
the spread of malaria and dengue fever into new regions. Air pollution from fossil fuel combustion causes
respiratory diseases and approximately 7 million premature deaths annually worldwide.
    """,
}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> dict:
    console.print(Panel.fit(
        "[bold bright_blue]Multi-Hop RAG Pipeline[/bold bright_blue]\n"
        "[dim]LangChain · OllamaEmbeddings · ChatOllama · InMemoryVectorStore[/dim]",
        border_style="bright_blue",
    ))
    console.print(
        f"  [dim]Chat model  : [bold]{OLLAMA_MODEL}[/bold]\n"
        f"  Embed model : [bold]{OLLAMA_EMBED_MODEL}[/bold]\n"
        f"  Chunk size  : {CHUNK_SIZE} chars  |  overlap: {CHUNK_OVERLAP} chars[/dim]\n"
    )

    # ── Build document store ──────────────────────────────────────────────────
    console.print("[bold]Loading documents…[/bold]")
    store = DocumentStore(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    for doc_id, text in SAMPLE_DOCS.items():
        store.add_document(doc_id, text)
        console.print(f"  [green]✓[/green] {doc_id}")
    store.build_index()

    # ── Multi-hop query ───────────────────────────────────────────────────────
    complex_query = (
        "How does climate change driven by fossil fuel use relate to both human health outcomes "
        "and the acceleration of renewable energy adoption, and what role does artificial intelligence "
        "play in addressing these interconnected challenges?"
    )
    console.print(
        f"\n[bold]Running pipeline on query:[/bold]\n  [italic]{complex_query}[/italic]\n"
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    result = run_pipeline(complex_query, store)

    # ── Display ───────────────────────────────────────────────────────────────
    print_results(result)

    # ── Return structured dict for programmatic / test use ───────────────────
    return {
        "original_query": result.original_query,
        "sub_queries":    result.sub_queries,
        "retrieved_chunks_per_subquery": [
            {
                "sub_query":     r.sub_query,
                "chunks": [
                    {"doc_id": c.doc_id, "text": c.text, "cosine_score": s}
                    for c, s in zip(r.chunks, r.chunk_scores)
                ],
                "llm_answer":       r.llm_answer,
                "avg_cosine_score": r.avg_similarity,
            }
            for r in result.sub_results
        ],
        "final_answer":          result.final_answer,
        "aggregated_cosine_score": result.agg_score,
        "elapsed_seconds":       result.elapsed_sec,
    }


if __name__ == "__main__":
    main()