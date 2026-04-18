HF_LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
HF_EMBED_MODEL: str = "BAAI/bge-base-en-v1.5"
TOP_K_CHUNKS: int = 3
CHUNK_SIZE: int = 300       # kept — used as breakpoint_threshold fallback
CHUNK_OVERLAP: int = 50     # kept for reference (SemanticChunker doesn't use it)
DATA_DIR: str = "data"