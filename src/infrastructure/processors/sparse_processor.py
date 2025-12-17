from src.application.services.ingestion_pipeline import ChunkProcessor

from src.infrastructure.retrieval.bm25_service import BM25Service
from src.domain.models import ProcessedChunk
from typing import List

class SparseEmbeddingProcessor(ChunkProcessor):
    def __init__(self, service: BM25Service):
        self.service = service
        
    def process(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Processes chunks for sparse embedding.
        In this refactored version, we index the chunks into the BM25 service directly here,
        or we generate sparse vectors if the service supports it.
        
        Since BM25Service manages an internal index, we might want to Add them to that index here.
        """
        if not chunks:
            return []
            
        print(f"[SparseEmbeddingProcessor] Indexing {len(chunks)} chunks in BM25Service...")
        self.service.index_documents(chunks, overwrite=False)
        
        # If we needed to attach sparse_vectors to the chunks for Qdrant:
        # for chunk in chunks:
        #     chunk.sparse_vector = self.service.get_sparse_vector(chunk.content)
            
        return chunks
