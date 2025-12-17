# src/infrastructure/retrieval/bm25_service.py
import pickle
import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
from src.application.ports.vector_store_port import RetrievalStrategy
from src.domain.models import ProcessedChunk

class BM25RetrieverImpl(RetrievalStrategy):
    def __init__(self, storage_path: str = "bm25_index.pkl"):
        self.storage_path = storage_path
        self.bm25_index = None
        self.chunks: List[ProcessedChunk] = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "rb") as f:
                data = pickle.load(f)
                self.bm25_index = data["index"]
                self.chunks = data["chunks"]
                print(f"[BM25] Index loaded from {self.storage_path}. Documents: {len(self.chunks)}")
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
        if overwrite:
            self.chunks = []
        
        # Append new chunks
        self.chunks.extend(new_chunks)
        
        # Tokenize corpus
        # Simple tokenization: split by whitespace. For better results, use sizing/stemming.
        tokenized_corpus = [chunk.content.lower().split() for chunk in self.chunks]
        
        # Build Index
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.save()

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        if not self.bm25_index or not self.chunks:
            print("[BM25] Index is empty. Returning valid empty list.")
            return []

        tokenized_query = query.lower().split()
        
        # rank_bm25 returns the docs themselves
        # get_top_n returns the actual items from the corpus list
        results = self.bm25_index.get_top_n(tokenized_query, self.chunks, n=top_k)
        
        # We can also get scores if we wanted:
        # scores = self.bm25_index.get_scores(tokenized_query)
        # But get_top_n is convenient.
        
        print(f"[BM25] Retrieved {len(results)} chunks for query: '{query}'")
        return results
