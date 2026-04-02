import os
import sys
from rich.panel import Panel

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, OLLAMA_EMBED_MODEL, OLLAMA_MODEL, DATA_DIR
from src.document_store import DocumentStore
from src.pipeline import run_pipeline
from src.utils import console, print_results

def main():
    console.print(Panel.fit(
        "[bold bright_blue]Multi-Hop RAG Pipeline[/bold bright_blue]\n"
        "[dim]LangChain · OllamaEmbeddings · ChatOllama · Qdrant DB[/dim]",
        border_style="bright_blue",
    ))
    console.print(
        f"  [dim]Chat model  : [bold]{OLLAMA_MODEL}[/bold]\n"
        f"  Embed model : [bold]{OLLAMA_EMBED_MODEL}[/bold]\n"
        f"  Chunk size  : {CHUNK_SIZE} chars  |  overlap: {CHUNK_OVERLAP} chars[/dim]\n"
    )

    # 1. Ensure Data Directory Exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        console.print(f"\n[bold red]Created '{DATA_DIR}' folder.[/bold red]")
        console.print(f"[yellow]Please add your .txt files to the '{DATA_DIR}' folder and run the script again.[/yellow]")
        sys.exit(1)

    # 2. Read all .txt files from the Directory
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not files:
        console.print(f"\n[bold red]No .txt files found in the '{DATA_DIR}' folder.[/bold red]")
        console.print(f"[yellow]Please add some text files into '{DATA_DIR}/' to query against.[/yellow]")
        sys.exit(1)

    # 3. Build document store
    console.print(f"\n[bold]Loading {len(files)} document(s) from '{DATA_DIR}/'…[/bold]")
    store = DocumentStore(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        store.add_document(doc_id=filename, text=content)
        console.print(f"  [green]✓[/green] Ingested: {filename}")
        
    store.build_index()

    # 4. Interactive Query Loop
    while True:
        console.print("\n[bold cyan]Enter your complex query (or type 'exit' to quit):[/bold cyan]")
        try:
            user_query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting.")
            break
        
        if user_query.lower() in ["exit", "quit", "q", ""]:
            console.print("Exiting.")
            break

        console.print(f"\n[bold]Processing query:[/bold] [italic]{user_query}[/italic]\n")

        # 5. Run Pipeline & Display Results
        result = run_pipeline(user_query, store)
        print_results(result)

if __name__ == "__main__":
    main()