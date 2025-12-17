# src/infrastructure/helpers/ingestion_helpers.py
"""
Helper functions for document ingestion with processing pipeline.
Moved from app-level to keep Application layer clean.
"""
from typing import List
from src.domain.models import ProcessedChunk
from src.application.services.rag_service import VectorStoreService
from src.infrastructure.loaders.factory import DocumentLoaderFactory
from src.application.services.ingestion_pipeline import IngestionPipeline
from src.infrastructure.processors.processors import (
    CleanerProcessor,
    MetadataExtractorProcessor,
)
from src.application.ports.llm_port import LLMService


def run_indexing_service(
    file_path: str,
    vector_store: VectorStoreService,
    llm_service: LLMService | None = None,
    overwrite: bool = False,
) -> None:
    """
    Complete indexing service with pipeline processing.
    
    Args:
        file_path: Path to document
        vector_store: RAG service instance
        llm_service: Optional LLM for metadata extraction
        overwrite: Whether to recreate index
    """
    # 1. Load and chunk with LLM extraction
    if file_path.lower().endswith('.pdf'):
        from src.infrastructure.loaders.pdf_loader import PDFLoader
        loader = PDFLoader(llm_service=llm_service)
        print(f"[Index] Using PDFLoader with LLM extraction: {llm_service is not None}")
    else:
        loader = DocumentLoaderFactory.get_loader(file_path)
    
    chunks = loader.load_and_chunk(file_path)

    # 2. Processing Pipeline (Regex-based extraction adicional)
    pipeline = IngestionPipeline([CleanerProcessor(), MetadataExtractorProcessor()])

    print("[Index] Ejecutando Pipeline de Ingesta (Limpieza + Extracción)...")
    refined_chunks = pipeline.run(chunks)

    # 3. Index
    print("[Index] Generando embeddings e indexando en Qdrant...")
    vector_store.index_chunks(refined_chunks, overwrite=overwrite)
    print("[Index] Listo.")