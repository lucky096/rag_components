from typing import Optional

from sqlalchemy import true


class Config:
    class Path:
        ROOT_DIR = "/Users/lucky/Workspace/rag_components/data/"
        DATABASE_DIR = ROOT_DIR / "database"
        DOCUMENTS_DIR = ROOT_DIR / "documents"
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

    
