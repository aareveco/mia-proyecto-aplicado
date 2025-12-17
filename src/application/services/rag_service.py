from typing import List, Dict, Optional, Any
import numpy as np

from src.domain.models import ProcessedChunk
from src.application.ports.embedder_port import AbstractEmbedder
from src.application.ports.vector_store_port import VectorStoreImpl, RetrievalStrategy
from src.application.ports.llm_port import LLMService
from src.application.ports.reranker_port import RerankerService

from src.application.services.query_processing import QueryRewritingStrategy
from src.application.ports.pubchem_port import PubChemService
from src.application.services.generation_service import AugmentedGenerator






from src.application.ports.indexer_port import IndexerPort
from src.application.services.retrieval_factory import RetrievalStrategyFactory

class VectorStoreService:
    """
    Updated Service handling the full RAG pipeline (Advanced) with Dependency Injection.
    """

    def __init__(
        self, 
        embedder: AbstractEmbedder, 
        db_impl: VectorStoreImpl,
        llm_service: LLMService,
        reranker_service: RerankerService,
        keyword_retriever: RetrievalStrategy, # Injected BM25Service
        pubchem_service: Optional[PubChemService] = None, 
    ):
        self._embedder = embedder
        self._db_impl = db_impl
        self.llm_service = llm_service
        self.reranker_service = reranker_service
        self.keyword_retriever = keyword_retriever
        self.pubchem_service = pubchem_service
        
        # Public Query Processor for UI interaction (e.g. rewriting preview)
        self.query_processor = QueryRewritingStrategy(self.llm_service)
        
        # Factory for Strategy Creation
        self.strategy_factory = RetrievalStrategyFactory(
            embedder=embedder,
            vector_store=db_impl,
            llm_service=llm_service,
            reranker=reranker_service,
            keyword_retriever=keyword_retriever,
            pubchem_service=pubchem_service
        )
        
        # Initialize default strategies via Factory
        self.final_retriever = self.strategy_factory.create_optimized_pipeline()
        
        # Generation
        self.generator = AugmentedGenerator(self.llm_service)

    def get_retrieval_strategy(self, mode: str = "hybrid", use_pubchem: bool = True) -> RetrievalStrategy:
        """
        Factory method to get a strategy based on configuration.
        mode: 'hybrid' | 'dense'
        """
        return self.strategy_factory.create_strategy(mode=mode, use_pubchem=use_pubchem)

    def index_chunks(self, chunks: List[ProcessedChunk], overwrite: bool = False) -> None:
        print("[Service] Indexing chunks in Dense Vector Store...")
        vectors = self._embedder.embed_chunks(chunks)
        for chunk, vec in zip(chunks, vectors):
            chunk.dense_vector = vec.tolist()
        metadatas = [c.model_dump() for c in chunks]
        self._db_impl.index_data(vectors, metadatas, overwrite=overwrite)
        
        # Index in Keyword Retriever (BM25) SAFELY using IndexerPort
        if isinstance(self.keyword_retriever, IndexerPort):
            print("[Service] Indexing in Keyword Retriever (BM25)...")
            self.keyword_retriever.index_documents(chunks, overwrite=overwrite)
        else:
            print("[Service] Keyword Retriever does not support indexing (IndexerPort not implemented). Skipping.")

    def query(self, query_text: str, top_k: int = 5) -> List[ProcessedChunk]:
        """
        Executes the full retrieval pipeline.
        """
        return self.final_retriever.retrieve_context(query_text, {}, top_k=top_k)

    def generate(self, query: str, context: List[ProcessedChunk]) -> str:
        """
        Invokes the Augmented Generator.
        """
        return self.generator.generate_answer(query, context)




