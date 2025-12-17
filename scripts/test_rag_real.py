import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService

# Injectable Services
from src.infrastructure.llm.local_llm_service import LocalLLMService
from src.infrastructure.reranker.cross_encoder_reranker import CrossEncoderRerankerService
from src.infrastructure.retrieval.bm25_service import BM25RetrieverImpl
from src.infrastructure.adapters.pubchem_adapter import PubChemAdapter
from src.domain.models import ProcessedChunk

# Pipeline & Loading
from src.infrastructure.loaders.factory import DocumentLoaderFactory
from src.application.services.ingestion_pipeline import IngestionPipeline
from src.infrastructure.processors.processors import CleanerProcessor, MetadataExtractorProcessor

def main():
    print("Initializing RAG Components (REAL LLM)...")
    
    # 1. Embedder
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    
    # 2. Vector Store (In-Memory for this test, or use persisted if you prefer)
    db_impl = QdrantImpl(collection_name="test_rag_real", path=None)

    # 3. Infrastructure
    llm = LocalLLMService()
    reranker = CrossEncoderRerankerService()
    # Using memory storage for test BM25 (or temp path)
    bm25 = BM25RetrieverImpl(storage_path="data/test_bm25.pkl")
    pubchem = PubChemAdapter()
    
    # 3. Service (This initializes Localrep connecting to Ollama)
    service = VectorStoreService(
        embedder=embedder, 
        db_impl=db_impl,
        llm_service=llm,
        reranker_service=reranker,
        sparse_retriever=bm25,
        pubchem_service=pubchem
    )
    
    # 4. Load & Process Data (Real PDF)
    pdf_path = "data/1-s2.0-S259015752400539X-main.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found. Please ensure a PDF exists in data/.")
        return

    print(f"\n--- Loading PDF: {pdf_path} ---")
    loader = DocumentLoaderFactory.get_loader(pdf_path)
    raw_chunks = loader.load_and_chunk(pdf_path)
    print(f"Loaded {len(raw_chunks)} raw chunks.")

    # 5. Ingestion Pipeline
    print("--- Running Ingestion Pipeline ---")
    pipeline = IngestionPipeline([
        CleanerProcessor(),
        MetadataExtractorProcessor()
    ])
    refined_chunks = pipeline.run(raw_chunks)
    
    # Show example of metadata extraction
    for i, c in enumerate(refined_chunks[:3]):
        if c.metadata.get('mz') or c.metadata.get('rt'):
             print(f"Chunk {i} extracted metadata: {c.metadata}")

    # 6. Indexing
    print("--- Indexing Refined Chunks ---")
    service.index_chunks(refined_chunks)
    
    # Query (Adjusted to be relevant to a scientific paper usually found in data)
    # Let's ask something generic if we don't know the content, or try to search for something likely there.
    # We switch the query to test PubChem integration as requested.
    query = "Feature m/z 495.1285 rt 5.99 in cocoa powder"
    print(f"\n--- Running Pipeline for: '{query}' ---")
    
    # Retrieve
    retrieved = service.query(query, top_k=3)
    print(f"Retrieved {len(retrieved)} chunks.")
    
    # Generate
    print("Generating answer (calling Ollama)...")
    answer = service.generate(query, retrieved)
    
    print("\n=== FINAL ANSWER ===")
    print(answer)
    print("====================")

if __name__ == "__main__":
    main()
