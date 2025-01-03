from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


class Ingestor:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(Config.Model.EMBEDDINGS)
        self.semantic_splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="interquartile")
        self.recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128, add_start_index=True)

    def ingest(self, doc_paths: List[Path]) -> VectorStore:
        documents = []
        for doc_path in doc_paths:
            loaded_docs = PyPDFium2Loader(doc_path).load()
            doc_text =  "\n".join([doc.page_content for doc in loaded_docs])
            documents.extend(self.recursive_splitter.split_documents(self.semantic_splitter.create_documents([doc_text])))
        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            path=Config.path.DATABASE_DIR,
            collection_name=Config.Database.DOCUMENTS_COLLECTION
        )