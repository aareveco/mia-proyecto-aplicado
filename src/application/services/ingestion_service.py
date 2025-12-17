from typing import Optional
import os
from src.application.services.ingestion_pipeline import IngestionPipeline
from src.infrastructure.processors.processors import CleanerProcessor, MetadataExtractorProcessor
from src.infrastructure.processors.sparse_processor import SparseEmbeddingProcessor
from src.infrastructure.loaders.factory import DocumentLoaderFactory
from src.application.services.rag_service import VectorStoreService

class IngestionService:
    """
    Orchestrates the document ingestion capabilities of the application.
    Encapsulates the pipeline construction and execution.
    """
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store = vector_store_service

    def run_ingestion(self, file_path: str, overwrite: bool = False) -> None:
        """
        Runs the full ingestion pipeline for a given file.
        """
        loader = DocumentLoaderFactory.get_loader(file_path)
        chunks = loader.load_and_chunk(file_path)
        
        # --- PIPELINE STEP ---
        # Injecting SparseEmbeddingProcessor using the adapter from service
        # NOTE: We cast to SparseEncoder if needed, or assume the service port works.
        pipeline = IngestionPipeline([
            CleanerProcessor(),
            MetadataExtractorProcessor(),
            SparseEmbeddingProcessor(service=self.vector_store.sparse_retriever)
        ])
        
        print(f"[IngestionService] Running Pipeline (Cleaner + Extractor + Sparse) on {len(chunks)} chunks...")
        refined_chunks = pipeline.run(chunks)
        # ---------------------

        print("[IngestionService] Generating embeddings and indexing...")
        self.vector_store.index_chunks(refined_chunks, overwrite=overwrite)
        print("[IngestionService] Ingestion complete.")
