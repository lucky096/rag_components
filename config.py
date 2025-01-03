import os
from pathlib import Path


class Config:
    class Path:
        ROOT_DIR = Path(os.getenv("APP_HOME",  Path(__file__).parent.parent))
        DATABASE_DIR = ROOT_DIR / "docs-db"
        DOCUMENTS_DIR = ROOT_DIR / "tmp"
        IMAGES_DIR = ROOT_DIR / "images"

    class Database:
        DOCUMENTS_COLLECTION = "documents"

    class Model:
        EMBEDDINGS = "BAAI/bge-base-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LOCAL_LLM = "gemma2:9b"
        REMOTE_LLM = "llama-3.1-70b-versatile"
        TEMPERATURE = 0.0
        MAX_TOKENS = 8000
        USE_LOCAL = True
        
    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = False

    DEBUG = True
    CONVERSATION_MESSAGES_LIMIT = 6