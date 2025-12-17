from typing import List, Dict, Optional
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
    ContextRepackerDecorator
)
from src.application.services.generation_service import AugmentedGenerator


class VectorRetrievalStrategy(RetrievalStrategy):
    """
    Simple adapter to use VectorStoreImpl as a RetrievalStrategy.
    """
    def __init__(self, vector_store: VectorStoreImpl, embedder: AbstractEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        # Embed query
        query_chunk = ProcessedChunk(content=query)
        # Note: embed_chunks usually expects list
        query_vector = self.embedder.embed_chunks([query_chunk])[0]
        
        # Query DB
        results_dict = self.vector_store.query_data(query_vector, top_k=top_k, filters=filters)
        return [ProcessedChunk(**r) for r in results_dict]


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
        sparse_retriever: RetrievalStrategy, 
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
        
        # 1. Base Strategies
        self.dense_strategy = VectorRetrievalStrategy(self._db_impl, self._embedder)
        
        # 2. Hybrid / Federated
        strategies = [self.dense_strategy, self.sparse_retriever]
        if self.pubchem_service:
            print("[Service] PubChem Service enabled. Adding PubChemRetriever.")
            self.pubchem_retriever = PubChemRetriever(self.pubchem_service)
            strategies.append(self.pubchem_retriever)
            
        self.hybrid_strategy = FederatedRetriever(strategies=strategies)
        
        # 3. Query Optimization
        self.query_processor = QueryRewritingStrategy(self.llm_service)
        self.optimizer_retriever = QueryOptimizerRetriever(
            query_processor=self.query_processor,
            retrieval_strategy=self.hybrid_strategy
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


def run_retrieval_service(
    query: str,
    vector_store: VectorStoreService,
    top_k: int = 5,
) -> List[ProcessedChunk]:
    print(f"[Retrieval] Ejecutando búsqueda avanzada para: {query!r}")
    results = vector_store.query(query, top_k=top_k)
    return results

# run_indexing_service helper might interact with DocumentLoaderFactory which is infrastructure.
# To fail safely, we can keep it here IF we import factory ONLY inside the function, 
# or better, move this helper to a script or the infrastructure layer. 
# But let's check imports. loading factory is infra.
# So this function logically belongs to a higher level (like a CLI or Orchestrator in Infra), not Domain/App services.
# However, for now, to avoid breaking too much, I will remove the Import from top level and import locally, 
# OR move it to app.py/scripts. 
# Best Clean Code practice: Move `run_indexing_service` to `app.py` or a dedicated `pipeline_runner.py` in infra.
# I will REMOVE it from here to enforce separation. 
# Note: app.py used it. I will move it to app.py.

