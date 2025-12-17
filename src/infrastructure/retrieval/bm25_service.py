import pickle
import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from rank_bm25 import BM25Okapi
from src.application.ports.vector_store_port import RetrievalStrategy
from src.domain.models import ProcessedChunk

class BM25Service(RetrievalStrategy):
    """
    Local BM25 Service for keyword retrieval (In-Memory).
    Implements RetrievalStrategy.
    """
    def __init__(self, storage_path: str = "bm25_index.pkl"):
        self.storage_path = storage_path
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[ProcessedChunk] = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_index = data.get("index")
                    self.chunks = data.get("chunks", [])
                    print(f"[BM25] Index loaded from {self.storage_path}. Documents: {len(self.chunks)}")
            except Exception as e:
                print(f"[BM25] Error loading index: {e}. Initialize empty.")
                self.bm25_index = None
                self.chunks = []
        else:
            print(f"[BM25] No existing index found at {self.storage_path}. Initialized empty.")

    def save(self):
        data = {
             "index": self.bm25_index,
             "chunks": self.chunks
        }
        with open(self.storage_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[BM25] Index saved to {self.storage_path}")

    def index_documents(self, new_chunks: List[ProcessedChunk], overwrite: bool = False):
        """
        Updates the global index with new chunks.
        Note: BM25Okapi requires the full corpus to be built, so we must rebuild it 
        when new documents are added unless we use a different implementation.
        """
        if overwrite:
            print("[BM25] Overwriting existing index.")
            self.chunks = []

        if not new_chunks and not self.chunks:
            return
        
        # Append new chunks
        self.chunks.extend(new_chunks)
        
        print(f"[BM25] Rebuilding index with {len(self.chunks)} total documents...")
        
        # Tokenize corpus
        # Simple tokenization: split by whitespace. For better results, use consistent tokenization.
        tokenized_corpus = [self._tokenize(chunk.content) for chunk in self.chunks]
        
        # Build Index
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.save()

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        if not self.bm25_index or not self.chunks:
            print("[BM25] Index is empty. Returning empty list.")
            return []

        tokenized_query = self._tokenize(query)
        
        # rank_bm25 returns the docs themselves
        results = self.bm25_index.get_top_n(tokenized_query, self.chunks, n=top_k)
        
        print(f"[BM25] Retrieved {len(results)} chunks for query: '{query}'")
        return results

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()
