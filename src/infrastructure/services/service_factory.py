from src.infrastructure.llm.gemini_factory import GeminiFactory
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService

class ServiceFactory:
    """
    Factory for creating application services and infrastructure components.
    This centralizes dependency injection.
    """

    @staticmethod
    def get_vector_service(qdrant_path: str = "qdrant_storage") -> VectorStoreService:
        """
        Creates and returns a configured VectorStoreService.
        """
        # 1. Get the Gemini Embedder
        embedder = GeminiFactory.get_app_embedder(model_name="models/text-embedding-004")

        # 2. Connect to Qdrant
        db_impl = QdrantImpl(collection_name="rag_chunks", path=qdrant_path)

        # 3. Create Service
        return VectorStoreService(embedder=embedder, db_impl=db_impl)

    @staticmethod
    def get_evaluation_components():
        """
        Returns the LLM and Embeddings needed for Ragas evaluation.
        """
        llm = GeminiFactory.get_generator_llm("gemini-2.0-flash")
        embeddings = GeminiFactory.get_embeddings()
        return llm, embeddings
