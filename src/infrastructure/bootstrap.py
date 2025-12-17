# src/infrastructure/bootstrap.py
from typing import Optional
from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.embeddings.gemini import GeminiEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService
from src.infrastructure.llm.gemini_llm_service import GeminiLLMService
from src.infrastructure.reranker.cross_encoder_reranker import CrossEncoderRerankerService
from src.infrastructure.retrieval.bm25_service import BM25RetrieverImpl
from src.infrastructure.adapters.pubchem_adapter import PubChemAdapter


def create_rag_service(
    qdrant_path: str,
    qdrant_collection: str = "rag_chunks",
    bm25_path: str = "data/bm25_index.pkl",
    enable_pubchem: bool = False,
    enable_llm_extraction: bool = True,
    use_gemini_embeddings: bool = True,
    gemini_model: str = "gemini-2.0-flash-exp",
) -> VectorStoreService:
    """
    Factory function to create a fully configured VectorStoreService.
    
    Args:
        qdrant_path: Path for Qdrant persistence
        qdrant_collection: Collection name
        bm25_path: Path for BM25 index
        enable_pubchem: Enable PubChem integration
        enable_llm_extraction: Enable LLM-based metadata extraction
        use_gemini_embeddings: Use Gemini embeddings (True) or HuggingFace (False)
        gemini_model: Gemini model name for LLM
    """

    # 1. Embeddings
    if use_gemini_embeddings:
        print("[Bootstrap] Using Gemini Embeddings")
        embedder = GeminiEmbedder(model_name="models/text-embedding-004")
    else:
        print("[Bootstrap] Using HuggingFace Embeddings")
        embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")

    # 2. Vector Store
    db_impl = QdrantImpl(collection_name=qdrant_collection, path=qdrant_path)

    # 3. LLM Service (Gemini)
    print(f"[Bootstrap] Using Gemini LLM: {gemini_model}")
    llm_service = GeminiLLMService(model_name=gemini_model)

    # 4. Reranker
    reranker_service = CrossEncoderRerankerService()

    # 5. Sparse Retriever (BM25)
    bm25_service = BM25RetrieverImpl(storage_path=bm25_path)

    # 6. Optional PubChem
    pubchem_service = None
    if enable_pubchem:
        print("[Bootstrap] PubChem Service Enabled.")
        pubchem_service = PubChemAdapter()

    # 7. Service Injection
    service = VectorStoreService(
        embedder=embedder,
        db_impl=db_impl,
        llm_service=llm_service,
        reranker_service=reranker_service,
        sparse_retriever=bm25_service,
        pubchem_service=pubchem_service,
    )

    return service