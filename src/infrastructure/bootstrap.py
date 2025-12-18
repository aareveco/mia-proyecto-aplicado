from typing import Optional

from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService
from src.infrastructure.llm.llm_factory import LLMFactory
from src.infrastructure.reranker.cross_encoder_reranker import CrossEncoderRerankerService
# from src.infrastructure.embeddings.bm25 import BM25API, BM25Adapter # Removed
from src.infrastructure.adapters.pubchem_adapter import PubChemAdapter

def create_rag_service(
    qdrant_path: str,
    qdrant_collection: str = "rag_chunks",
    enable_pubchem: bool = False,
    bm25_path: str = "bm25_index.pkl",
    llm_provider: str = "gemini"
) -> VectorStoreService:
    """
    Factory function to create a fully configured VectorStoreService.
    Centralizes the initialization of all infrastructure adapters.
    """
    
    # 1. Embeddings
    # 1. Embeddings
    # Centralized creation via Factory (defaults to local/HF, can be switched to gemini)
    embedder = LLMFactory.get_app_embeddings(provider="local", model="all-MiniLM-L6-v2")
    
    # 2. Vector Store
    # Use path for disk storage or None for memory.
    # Note: Application might pass qdrant_path="qdrant_storage"
    db_impl = QdrantImpl(collection_name=qdrant_collection, path=qdrant_path)
    
    # 3. Infrastructure Services
    # llm_service = LocalLLMService()
    print(f"[Bootstrap] Initializing {llm_provider} LLM Service...")
    llm_service = LLMFactory.get_app_llm(provider=llm_provider, model="gemini-2.0-flash-exp" if llm_provider == "gemini" else "qwen2.5:1.5b")
    reranker_service = CrossEncoderRerankerService()
    
    # 4. Sparse Embedding / Retrieval (Unified)
    from src.infrastructure.retrieval.bm25_service import BM25Service
    bm25_service = BM25Service(storage_path=bm25_path)
    
    # 5. Optional Services
    pubchem_service = None
    if enable_pubchem:
        print("[Bootstrap] PubChem Service Enabled.")
        pubchem_service = PubChemAdapter()

    # 6. Service Injection
    # We pass bm25_service as the keyword_retriever argument for Federated Hybrid Search
    service = VectorStoreService(
        embedder=embedder, 
        db_impl=db_impl,
        llm_service=llm_service,
        reranker_service=reranker_service,
        keyword_retriever=bm25_service, 
        pubchem_service=pubchem_service
    )
    
    
    return service

def create_evaluation_resources():
    """
    Creates LangChain-compatible LLM and Embeddings for Ragas evaluation.
    This ensures app.py relies on bootstrap for all infrastructure.
    """
    # Use Factory for consistency (defaulting to Gemini as per current preference)
    llm = LLMFactory.get_ragas_llm(provider="gemini", model="gemini-2.0-flash-exp")
    embeddings = LLMFactory.get_ragas_embeddings(provider="gemini", model="models/text-embedding-004")
    
    return llm, embeddings

