from typing import Optional
from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService
from src.infrastructure.llm.local_llm_service import LocalLLMService
from src.infrastructure.reranker.cross_encoder_reranker import CrossEncoderRerankerService
from src.infrastructure.retrieval.bm25_service import BM25RetrieverImpl
from src.infrastructure.adapters.pubchem_adapter import PubChemAdapter

def create_rag_service(
    qdrant_path: str,
    qdrant_collection: str = "rag_chunks",
    bm25_path: str = "data/bm25_index.pkl",
    overwrite_qdrant: bool = False, # Note: Qdrant service handles overwrite in index_data, here we just configure path
    enable_pubchem: bool = False
) -> VectorStoreService:
    """
    Factory function to create a fully configured VectorStoreService.
    Centralizes the initialization of all infrastructure adapters.
    """
    
    # 1. Embeddings
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    
    # 2. Vector Store
    # Note: path=None means in-memory for QdrantImpl, but here we usually want persistence if path provided
    # If path is "memory" or None, it will be in-memory (handled by QdrantImpl logic if applicable or default behavior)
    # The current QdrantImpl seems to take 'path' for local storage.
    
    db_impl = QdrantImpl(collection_name=qdrant_collection, path=qdrant_path)
    
    # 3. Infrastructure Services
    llm_service = LocalLLMService()
    reranker_service = CrossEncoderRerankerService()
    
    # 4. Sparse Retriever
    bm25_service = BM25RetrieverImpl(storage_path=bm25_path)
    
    # 5. Optional Services
    pubchem_service = None
    if enable_pubchem:
        print("[Bootstrap] PubChem Service Enabled.")
        pubchem_service = PubChemAdapter()

    # 6. Service Injection
    service = VectorStoreService(
        embedder=embedder, 
        db_impl=db_impl,
        llm_service=llm_service,
        reranker_service=reranker_service,
        sparse_retriever=bm25_service,
        pubchem_service=pubchem_service
    )
    
    return service

