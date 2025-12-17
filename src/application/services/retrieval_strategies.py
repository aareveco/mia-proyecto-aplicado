from typing import List, Dict, Any
from collections import defaultdict
from src.application.ports.vector_store_port import VectorStoreImpl, RetrievalStrategy
from src.application.ports.embedder_port import AbstractEmbedder
from src.application.ports.reranker_port import RerankerService
from src.application.services.query_processing import QueryProcessingStrategy
from src.domain.models import ProcessedChunk
from src.application.ports.pubchem_port import PubChemService

class DenseRetriever(RetrievalStrategy):
    """
    Realiza búsqueda puramente semántica (Dense Only) usando embeddings.
    """
    def __init__(self, vector_store: VectorStoreImpl, embedder: AbstractEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        # 1. Embed query (Dense)
        query_chunk = ProcessedChunk(content=query)
        query_dense = self.embedder.embed_chunks([query_chunk])[0]
        
        # 2. Query Vector Store (Dense)
        # Assuming database impl has query_data which does dense search or generic search
        results_dict = self.vector_store.query_data(query_dense, top_k=top_k, filters=filters)
            
        return [ProcessedChunk(**r) for r in results_dict]





class PubChemRetriever(RetrievalStrategy):
    """
    Retrieves compound information from PubChem if 'mz' filter is present.
    """
    def __init__(self, pubchem_service: PubChemService):
        self.pubchem = pubchem_service

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        mz = filters.get("mz")
        if not mz:
            return []
        
        try:
            # Cast to float if it's a string/number
            mz_val = float(mz)
        except (ValueError, TypeError):
            return []

        print(f"[PubChemRetriever] Searching for m/z: {mz_val}")
        result = self.pubchem.get_compound_by_mz(mz_val)
        
        if not result:
            return []
            
        # Convert result to ProcessedChunk
        # We create a synthetic chunk with the compound info
        compounds = result.get("compounds", [])
        content_lines = ["**PubChem Search Results**"]
        for c in compounds:
            name = c.get("Title", "Unknown")
            formula = c.get("MolecularFormula", "")
            bio = c.get("Bioactivity", [])
            bio_str = "; ".join(bio) if bio else "No specific assays found."
            content_lines.append(f"- Name: {name}, Formula: {formula}")
            content_lines.append(f"  Bioactivities: {bio_str}")
            
        full_content = "\n".join(content_lines)
        
        chunk = ProcessedChunk(
            content=full_content,
            metadata={
                "source": "PubChem",
                "mz_query": mz_val,
                "raw_result": str(result)
            }
        )
        return [chunk]


def reciprocal_rank_fusion(results_lists: List[List[ProcessedChunk]], k=60) -> List[ProcessedChunk]:
    """
    Combines multiple lists of ranked results using Reciprocal Rank Fusion (RRF).
    score = sum(1 / (k + rank))
    """
    scores = defaultdict(float)
    chunk_map = {}

    for results in results_lists:
        for rank, chunk in enumerate(results):
            # Identify chunk by ID or content hash if ID missing
            # Assuming chunk_id is present and unique
            cid = chunk.chunk_id if chunk.chunk_id else chunk.content[:50] 
            
            # Store chunk obj
            if cid not in chunk_map:
                chunk_map[cid] = chunk
            
            scores[cid] += 1.0 / (k + rank + 1)

    # Sort by score desc
    sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    final_results = []
    for cid in sorted_cids:
        # We could attach the merged score to metadata
        chunk = chunk_map[cid]
        # Duplicate to avoid mutating original references if used elsewhere
        # (shallow copy might be enough)
        # chunk.metadata['rrf_score'] = scores[cid] 
        final_results.append(chunk)

    return final_results



class FederatedRetriever(RetrievalStrategy):
    """
    Executes multiple retrieval strategies in parallel (or sequentially) and merges results using RRF.
    """
    def __init__(self, strategies: List[RetrievalStrategy]):
        self.strategies = strategies

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        results_lists = []
        for strategy in self.strategies:
            # We could do this in parallel threads
            results_lists.append(strategy.retrieve_context(query, filters, top_k))
        
        # Merge using RRF
        combined = reciprocal_rank_fusion(results_lists)
        return combined[:top_k]

# Redefining to use composition of strategies
class CompositionalHybridSearchRetriever(FederatedRetriever):
    """
    Hybrid Search using two underlying strategies.
    (Backwards compatibility wrapper around FederatedRetriever)
    """
    def __init__(self, dense_strategy: RetrievalStrategy, sparse_strategy: RetrievalStrategy):
        super().__init__([dense_strategy, sparse_strategy])



class QueryOptimizerRetriever(RetrievalStrategy):
    """
    Orchestrates the retrieval process:
    1. Optimizes the query (rewriting + filter extraction).
    2. Delegates to the underlying retrieval strategy (e.g., HybridSearch).
    """
    def __init__(
        self,
        query_processor: QueryProcessingStrategy,
        retrieval_strategy: RetrievalStrategy,
    ):
        self.processor = query_processor
        self.retriever = retrieval_strategy

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        # 1. Process/Optimize Query
        structured_query_output = self.processor.process_query(query)

        # 2. Merge user-provided filters with extracted filters
        combined_filters = {**filters, **structured_query_output.metadata_filters}
        
        optimized_query = structured_query_output.rewritten_query
        
        print(f"Query reescrita: '{optimized_query}'. Filtros: {combined_filters}")

        # 3. Retrieve using the optimized query
        return self.retriever.retrieve_context(
            optimized_query,
            combined_filters,
            top_k=top_k
        )


class RetrievalDecorator(RetrievalStrategy):
    """Base Decorator for Retrieval Strategies"""
    def __init__(self, wrapped_strategy: RetrievalStrategy):
        self._wrapped_strategy = wrapped_strategy

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        return self._wrapped_strategy.retrieve_context(query, filters, top_k=top_k)


class RerankingDecorator(RetrievalDecorator):
    """
    Fetches more candidates than needed (top_k * N), then reranks them using a Cross-Encoder.
    """
    def __init__(
        self,
        wrapped_strategy: RetrievalStrategy,
        reranker: RerankerService,
        fetch_k_multiplier: int = 2
    ):
        super().__init__(wrapped_strategy)
        self.reranker_service = reranker
        self.fetch_k_multiplier = fetch_k_multiplier

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        # 1. Fetch more candidates
        candidate_k = top_k * self.fetch_k_multiplier
        candidates: List[ProcessedChunk] = self._wrapped_strategy.retrieve_context(
            query, filters, top_k=candidate_k
        )
        
        if not candidates:
            return []

        # 2. Rerank
        texts_to_rank = [c.content for c in candidates]
        scores = self.reranker_service.rerank(query, texts_to_rank)

        # 3. Assign scores and sort
        scored_chunks = []
        for chunk, score in zip(candidates, scores):
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["rerank_score"] = float(score) 
            chunk.metadata["score"] = float(score) # Update main score too for UI
            scored_chunks.append((chunk, score))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Reordenados. Top score: {scored_chunks[0][1]:.2f}")

        # Return top_k chunks
        return [c for c, _ in scored_chunks[:top_k]]


class ContextRepackerDecorator(RetrievalDecorator):
    """
    Reorders the final context to place the most relevant information at the beginning and end ("Sides").
    """
    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5) -> List[ProcessedChunk]:
        chunks: List[ProcessedChunk] = self._wrapped_strategy.retrieve_context(
            query, filters, top_k=top_k
        )

        if len(chunks) < 3:
            return chunks

        print(f"Aplicando Sides Repacking para {len(chunks)} chunks.")

        # "Sides" packing: Best first, Second Best last, Third 2nd, Fourth 2nd-to-last...
        # Input assumed sorted by relevance: [0, 1, 2, 3, 4]
        # Output: [0, 2, 4, 3, 1]
        
        reordered = [None] * len(chunks)
        left, right = 0, len(chunks) - 1
        
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                reordered[left] = chunk
                left += 1
            else:
                reordered[right] = chunk
                right -= 1
        
        return reordered
