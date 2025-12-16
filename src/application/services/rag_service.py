from typing import List, Dict
import numpy as np

from src.domain.models import ProcessedChunk
from src.application.ports.embedder_port import AbstractEmbedder
from src.application.ports.vector_store_port import VectorStoreImpl, RetrievalStrategy
from src.infrastructure.loaders.factory import DocumentLoaderFactory

# New imports
from src.application.ports.llm_port import LLMService
from src.application.ports.reranker_port import RerankerService
from src.application.services.query_processing import QueryRewritingStrategy
from src.application.services.retrieval_strategies import (
    CompositionalHybridSearchRetriever,
    QueryOptimizerRetriever,
    RerankingDecorator,
    ContextRepackerDecorator
)
from src.application.services.generation_service import AugmentedGenerator
from src.infrastructure.llm.local_llm_service import LocalLLMService
from src.infrastructure.reranker.dummy_reranker import DummyRerankerService

# Temporary simpler implementations for strategies if not injected
class VectorRetrievalStrategy(RetrievalStrategy):
    def __init__(self, vector_store: VectorStoreImpl, embedder: AbstractEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        # Embed query
        query_chunk = ProcessedChunk(content=query)
        # Note: embed_chunks usually expects list
        query_vector = self.embedder.embed_chunks([query_chunk])[0]
        
        # Query DB
        # TODO: Pass filters if supported by vector_store
        # db_impl.query_data currently doesn't take filters in signature shown in qdrant_db.py, 
        # but VectorStoreImpl signature in 'vector_store_port.py' (viewed earlier) had 
        # abstract methods. Wait, I saw 'vector_store_port.py' earlier and it had:
        # def query_data(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        # It didn't obviously show filters argument in abstract method? 
        # Let's check line 14 of 'vector_store_port.py'.
        # Assuming no filters support in base for now, or we handle it in implementation.
        
        results_dict = self.vector_store.query_data(query_vector, top_k=top_k)
        return [ProcessedChunk(**r) for r in results_dict]

from src.infrastructure.retrieval.bm25_service import BM25RetrieverImpl

class VectorStoreService:
    """
    Updated Service handling the full RAG pipeline (Advanced).
    """

    def __init__(self, embedder: AbstractEmbedder, db_impl: VectorStoreImpl):
        self._embedder = embedder
        self._db_impl = db_impl
        
        # Initialize Services (In a real app, DI container handles this)
        # Assuming persistence path for BM25
        self.bm25_retriever = BM25RetrieverImpl(storage_path="data/bm25_index.pkl")
        
        self.llm_service = LocalLLMService()
        self.reranker_service = DummyRerankerService()
        
        # Build Retrieval Chain
        
        # 1. Base Strategies
        self.dense_strategy = VectorRetrievalStrategy(self._db_impl, self._embedder)
        self.sparse_strategy = self.bm25_retriever # Replaced Placeholder with Real BM25
        
        # 2. Hybrid
        self.hybrid_strategy = CompositionalHybridSearchRetriever(
            dense_strategy=self.dense_strategy, 
            sparse_strategy=self.sparse_strategy
        )
        
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
        
        # Index in BM25
        print("[Service] Indexing in BM25...")
        self.bm25_retriever.index_documents(chunks, overwrite=overwrite)

    def query(self, query_text: str, top_k: int = 5) -> List[ProcessedChunk]:
        """
        Legacy/Simple query (wraps advanced pipeline or stays simple?)
        The prompt asked to REFACTOR retrieve.
        Let's use the advanced pipeline here.
        """
        return self.final_retriever.retrieve_context(query_text, {}, top_k=top_k)

    def generate(self, query: str, context: List[ProcessedChunk]) -> str:
        """
        Invokes the Augmented Generator.
        """
        return self.generator.generate_answer(query, context)


def run_indexing_service(
    file_path: str,
    vector_store: VectorStoreService,
    overwrite: bool = False,
) -> None:
    loader = DocumentLoaderFactory.get_loader(file_path)
    chunks = loader.load_and_chunk(file_path)
    print("[Index] Generando embeddings e indexando en Qdrant...")
    vector_store.index_chunks(chunks, overwrite=overwrite)
    print("[Index] Listo.")


def run_retrieval_service(
    query: str,
    vector_store: VectorStoreService,
    top_k: int = 5,
) -> List[ProcessedChunk]:
    print(f"[Retrieval] Ejecutando búsqueda avanzada para: {query!r}")
    results = vector_store.query(query, top_k=top_k)
    return results
