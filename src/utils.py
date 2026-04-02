from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()

def print_results(result) -> None:
    from src.pipeline import PipelineResult
    assert isinstance(result, PipelineResult)

    console.print()
    console.print(Rule("[bold white]  MULTI-HOP RAG PIPELINE RESULTS  ", style="bright_blue"))

    console.print(Panel(
        f"[bold white]{result.original_query}",
        title="[bright_blue]● Original Query",
        border_style="bright_blue",
        padding=(0, 2),
    ))

    sq_text = Text()
    for i, sq in enumerate(result.sub_queries, 1):
        sq_text.append(f"  {i}. ", style="bold cyan")
        sq_text.append(f"{sq}\n")
    console.print(Panel(sq_text, title="[cyan]① Sub-Queries", border_style="cyan", padding=(0, 1)))

    for i, r in enumerate(result.sub_results, 1):
        console.print()
        console.print(Rule(f"[yellow]Sub-Query {i}", style="yellow", align="left"))
        console.print(f"  [bold yellow]Q:[/bold yellow] {r.sub_query}")
        console.print(f"  [bold green]A:[/bold green] {r.llm_answer}")
        console.print(f"  [dim]Avg cosine similarity (this sub-query): {r.avg_similarity:.4f}[/dim]")

        tbl = Table(
            "Rank", "Doc ID", "Cosine Score", "Chunk Preview",
            show_header=True, header_style="bold dim", border_style="dim", show_lines=True, expand=True,
        )
        for rank, (chunk, score) in enumerate(zip(r.chunks, r.chunk_scores), 1):
            preview = chunk.text[:120].replace("\n", " ") + ("…" if len(chunk.text) > 120 else "")
            tbl.add_row(str(rank), chunk.doc_id, f"{score:.4f}", preview)
        console.print(tbl)

    console.print()
    colour = "green" if result.agg_score >= 0.7 else "yellow" if result.agg_score >= 0.4 else "red"
    
    console.print(Panel(
        f"{result.final_answer}\n\n"
        f"[{colour}]Aggregated Cosine Score: {result.agg_score:.4f}[/{colour}]   "
        f"[dim]· elapsed: {result.elapsed_sec:.1f}s[/dim]",
        title="[green]④ Final Aggregated Answer",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()