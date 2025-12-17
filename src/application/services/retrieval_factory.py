from typing import List, Optional
from src.application.ports.vector_store_port import VectorStoreImpl, RetrievalStrategy
from src.application.ports.embedder_port import AbstractEmbedder
from src.application.ports.llm_port import LLMService
from src.application.ports.reranker_port import RerankerService
from src.application.ports.pubchem_port import PubChemService
from src.application.services.retrieval_strategies import (
    DenseRetriever,
    FederatedRetriever,
    PubChemRetriever,
    QueryOptimizerRetriever,
    RerankingDecorator,
    ContextRepackerDecorator,
)
from src.application.services.query_processing import QueryRewritingStrategy

class RetrievalStrategyFactory:
    """
    Factory to create and configure retrieval strategies.
    Decouples construction logic from the main service.
    """
    def __init__(
        self,
        embedder: AbstractEmbedder,
        vector_store: VectorStoreImpl,
        llm_service: LLMService,
        reranker: RerankerService,
        keyword_retriever: RetrievalStrategy, 
        pubchem_service: Optional[PubChemService] = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.reranker = reranker
        self.keyword_retriever = keyword_retriever
        self.pubchem_service = pubchem_service

    def create_dense_strategy(self) -> RetrievalStrategy:
        return DenseRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder
        )

    def create_hybrid_strategy(self) -> RetrievalStrategy:
        dense = self.create_dense_strategy()
        return FederatedRetriever(strategies=[dense, self.keyword_retriever])

    def create_pubchem_strategy(self) -> Optional[RetrievalStrategy]:
        if not self.pubchem_service:
            return None
        return PubChemRetriever(self.pubchem_service)

    def create_federated_strategy(self, use_pubchem: bool = True) -> RetrievalStrategy:
        """
        Creates the main strategy: Hybrid (Dense+Keyword) + PubChem (if available).
        """
        strategies = [self.create_dense_strategy(), self.keyword_retriever]
        
        if use_pubchem:
            pubchem = self.create_pubchem_strategy()
            if pubchem:
                strategies.append(pubchem)
            
        return FederatedRetriever(strategies=strategies)

    def create_strategy(self, mode: str = "hybrid", use_pubchem: bool = True) -> RetrievalStrategy:
        """
        Dynamic creation based on parameters.
        """
        strategies = []
        # Dense is always included in current logic (Semantic or Hybrid)
        strategies.append(self.create_dense_strategy())
        
        if mode == "hybrid":
            strategies.append(self.keyword_retriever)
            
        if use_pubchem:
            pubchem = self.create_pubchem_strategy()
            if pubchem:
                strategies.append(pubchem)
                
        # If only one strategy, return it directly? 
        # But DenseRetriever might not be wrapped. 
        # However, to be consistent with "FederatedRetriever" return type if multiple.
        # If single, we can return single.
        if len(strategies) == 1:
            return strategies[0]
            
        return FederatedRetriever(strategies=strategies)

    def create_optimized_pipeline(self) -> RetrievalStrategy:
        """
        Creates the full pipeline: 
        Query Optimization -> Federated Retrieval -> Reranking -> Repacking
        """
        # 1. Base Retrieval (Federated)
        base_strategy = self.create_federated_strategy()

        # 2. Query Optimization
        query_processor = QueryRewritingStrategy(self.llm_service)
        optimized_strategy = QueryOptimizerRetriever(
            query_processor=query_processor,
            retrieval_strategy=base_strategy
        )

        # 3. Reranking
        reranked_strategy = RerankingDecorator(
            wrapped_strategy=optimized_strategy,
            reranker=self.reranker
        )

        # 4. Repacking
        final_strategy = ContextRepackerDecorator(
            wrapped_strategy=reranked_strategy
        )

        return final_strategy
