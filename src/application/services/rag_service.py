from typing import List, Dict, Optional, Any
import numpy as np

from src.domain.models import ProcessedChunk
from src.application.ports.embedder_port import AbstractEmbedder
from src.application.ports.vector_store_port import VectorStoreImpl, RetrievalStrategy
from src.application.ports.llm_port import LLMService
from src.application.ports.reranker_port import RerankerService

from src.application.services.query_processing import QueryRewritingStrategy
from src.application.ports.pubchem_port import PubChemService
from src.application.services.retrieval_strategies import (
    CompositionalHybridSearchRetriever,
    FederatedRetriever,
    PubChemRetriever,
    QueryOptimizerRetriever,
    RerankingDecorator,
    ContextRepackerDecorator,
)
from src.application.services.generation_service import AugmentedGenerator





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
        sparse_retriever: Any, # Should implement RetrievalStrategy AND SparseEncoder
        pubchem_service: Optional[PubChemService] = None, 
        # Optional: Allow overriding the composition logic or strategies if needed, 
        # but for now we compose them here using the injected components.
    ):
        self._embedder = embedder
        self._db_impl = db_impl
        self.llm_service = llm_service
        self.reranker_service = reranker_service
        self.sparse_retriever = sparse_retriever
        self.pubchem_service = pubchem_service
        
        # Build Retrieval Chain
        
        # 1. Retrieval Strategies
        # using the injected sparse_retriever (BM25Adapter) as the sparse encoder
        from src.application.services.retrieval_strategies import QdrantHybridStrategy, DenseRetriever
        
        # A. Hybrid (Dense + Sparse)
        self.hybrid_strategy = QdrantHybridStrategy(
            vector_store=self._db_impl,
            embedder=self._embedder,
            sparse_encoder=self.sparse_retriever
        )
        
        # B. Dense Only (Semantic)
        self.dense_strategy = DenseRetriever(
            vector_store=self._db_impl,
            embedder=self._embedder
        )
        
        # C. PubChem (if enabled)
        self.pubchem_retriever = None
        if self.pubchem_service:
            print("[RAG Service] Integrating PubChem Retriever...")
            self.pubchem_retriever = PubChemRetriever(self.pubchem_service)
            
        # Default Federated (Hybrid + PubChem) for backward compatibility
        strategies = [self.hybrid_strategy]
        if self.pubchem_retriever:
             strategies.append(self.pubchem_retriever)
        self.federated_strategy = FederatedRetriever(strategies)
        
        # 3. Query Optimization
        self.query_processor = QueryRewritingStrategy(self.llm_service)
        self.optimizer_retriever = QueryOptimizerRetriever(
            query_processor=self.query_processor,
            retrieval_strategy=self.federated_strategy 
        )
        
        # 4. Reranking
        self.reranking_retriever = RerankingDecorator(
            wrapped_strategy=self.optimizer_retriever,
            reranker=self.reranker_service
        )
        
        # 5. Repacking
        self.final_retriever = ContextRepackerDecorator(
            wrapped_strategy=self.reranking_retriever
        )
        
        # Generation
        self.generator = AugmentedGenerator(self.llm_service)

    def get_retrieval_strategy(self, mode: str = "hybrid", use_pubchem: bool = True) -> RetrievalStrategy:
        """
        Factory method to get a strategy based on configuration.
        mode: 'hybrid' | 'dense'
        """
        # 1. Select Base
        if mode == "dense":
            base = self.dense_strategy
        else: # default hybrid
            base = self.hybrid_strategy
        
        # 2. Combine with PubChem?
        if use_pubchem and self.pubchem_retriever:
            return FederatedRetriever([base, self.pubchem_retriever])
        
        return base

    def index_chunks(self, chunks: List[ProcessedChunk], overwrite: bool = False) -> None:
        vectors = self._embedder.embed_chunks(chunks)
        for chunk, vec in zip(chunks, vectors):
            chunk.dense_vector = vec.tolist()
        metadatas = [c.model_dump() for c in chunks]
        self._db_impl.index_data(vectors, metadatas, overwrite=overwrite)
        
        # Index in Sparse Retriever (if it supports indexing interface)
        # Assuming sparse_retriever has index_documents method (it might need a separate Port definition for Indexing vs Retrieval)
        # For now, we assume it's the BM25RetrieverImpl which has it.
        # In a strict port sense, we should have an Indexable interface.
        if hasattr(self.sparse_retriever, "index_documents"):
            print("[Service] Indexing in Sparse Retriever...")
            self.sparse_retriever.index_documents(chunks, overwrite=overwrite)

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




