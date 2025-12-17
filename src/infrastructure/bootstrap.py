from typing import Optional
from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService
from src.infrastructure.llm.local_llm_service import LocalLLMService
from src.infrastructure.reranker.cross_encoder_reranker import CrossEncoderRerankerService
from src.infrastructure.embeddings.bm25 import BM25API, BM25Adapter
from src.infrastructure.adapters.pubchem_adapter import PubChemAdapter

def create_rag_service(
    qdrant_path: str,
    qdrant_collection: str = "rag_chunks",
    enable_pubchem: bool = False
) -> VectorStoreService:
    """
    Factory function to create a fully configured VectorStoreService.
    Centralizes the initialization of all infrastructure adapters.
    """
    
    # 1. Embeddings
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    
    # 2. Vector Store
    # Use path for disk storage or None for memory.
    # Note: Application might pass qdrant_path="qdrant_storage"
    db_impl = QdrantImpl(collection_name=qdrant_collection, path=qdrant_path)
    
    # 3. Infrastructure Services
    llm_service = LocalLLMService()
    reranker_service = CrossEncoderRerankerService()
    
    # 4. Sparse Embedding (Adapted)
    # Using the new BM25 architecture from rag_core
    bm25_api = BM25API()
    bm25_adapter = BM25Adapter(bm25_api)
    
    # 5. Optional Services
    pubchem_service = None
    if enable_pubchem:
        print("[Bootstrap] PubChem Service Enabled.")
        pubchem_service = PubChemAdapter()

    # 6. Service Injection
    # We pass bm25_adapter as the sparse_retriever argument because the service 
    # will use it for both indexing (embedding) and retrieval (hybrid strategy construction)
    service = VectorStoreService(
        embedder=embedder, 
        db_impl=db_impl,
        llm_service=llm_service,
        reranker_service=reranker_service,
        sparse_retriever=bm25_adapter, 
        pubchem_service=pubchem_service
    )
    
    
    return service

def create_evaluation_resources():
    """
    Creates LangChain-compatible LLM and Embeddings for Ragas evaluation.
    This ensures app.py relies on bootstrap for all infrastructure.
    """
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # Consistent config with create_rag_service
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return llm, embeddings

