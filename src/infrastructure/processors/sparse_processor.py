from src.application.services.ingestion_pipeline import ChunkProcessor
from src.infrastructure.embeddings.bm25 import BM25Adapter
from src.domain.models import ProcessedChunk
from typing import List

class SparseEmbeddingProcessor(ChunkProcessor):
    def __init__(self, adapter: BM25Adapter):
        self.adapter = adapter
        
    def process(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        if not chunks:
            return []
            
        print(f"[SparseEmbeddingProcessor] Generando vectores sparse para {len(chunks)} chunks...")
        # BM25Adapter.embed_chunks devuelve lista de tuples (indices, values)
        sparse_vectors = self.adapter.embed_chunks(chunks)
        
        for chunk, s_vec in zip(chunks, sparse_vectors):
            chunk.sparse_vector = s_vec
            
        return chunks
