# src/application/services/retrieval_strategies.py
from typing import List, Dict, Any
from collections import defaultdict
from src.application.ports.vector_store_port import VectorStoreImpl, RetrievalStrategy
from src.application.ports.embedder_port import AbstractEmbedder
from src.application.ports.reranker_port import RerankerService
from src.application.services.query_processing import QueryProcessingStrategy
from src.domain.models import ProcessedChunk
from src.application.ports.pubchem_port import PubChemService


class VectorRetrievalStrategy(RetrievalStrategy):
    """
    Adapter para usar VectorStoreImpl como RetrievalStrategy con post-filtering.
    """

    def __init__(self, vector_store: VectorStoreImpl, embedder: AbstractEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve_context(
        self, query: str, filters: Dict, top_k: int = 5
    ) -> List[ProcessedChunk]:
        # 1. Embed query
        query_chunk = ProcessedChunk(content=query)
        query_vector = self.embedder.embed_chunks([query_chunk])[0]

        # 2. Separar filtros numéricos de filtros exactos
        exact_filters = {}
        target_mz = filters.get("target_mz")
        target_rt = filters.get("target_rt")

        for key, value in filters.items():
            if key not in ["target_mz", "target_rt"]:
                exact_filters[key] = value

        # 3. Query DB (recuperar más si hay filtros numéricos)
        retrieval_k = top_k * 3 if (target_mz or target_rt) else top_k
        results_dict = self.vector_store.query_data(
            query_vector, top_k=retrieval_k, filters=exact_filters
        )

        # 4. Convertir a ProcessedChunk
        chunks = [ProcessedChunk(**r) for r in results_dict]

        # 5. Post-filtering por m/z y RT
        if target_mz or target_rt:
            chunks = self._post_filter_by_metadata(chunks, target_mz, target_rt)
            chunks = chunks[:top_k]
            print(f"[Post-Filter]: {len(chunks)} chunks ordenados por similitud m/z/RT")

        return chunks

    def _post_filter_by_metadata(
        self,
        chunks: List[ProcessedChunk],
        target_mz: float | None,
        target_rt: float | None,
    ) -> List[ProcessedChunk]:
        """
        Ordena chunks por coincidencia con target_mz y target_rt.
        Usa getattr seguro para backward compatibility.
        """
        scored_chunks = []

        for chunk in chunks:
            score = 0

            # Boost por m/z match (usar getattr seguro)
            if target_mz:
                mz_values = getattr(chunk, 'mz_values', None)
                if mz_values:
                    for mz in mz_values:
                        if abs(mz - target_mz) <= 0.01:  # ±0.01 Da
                            score += 10
                            break

            # Boost por RT match (usar getattr seguro)
            if target_rt:
                rt_values = getattr(chunk, 'rt_values', None)
                if rt_values:
                    for rt in rt_values:
                        if abs(rt - target_rt) <= 0.5:  # ±0.5 min
                            score += 5
                            break

            # Temporal score para ordenamiento
            chunk.rerank_score = score
            scored_chunks.append(chunk)

        # Ordenar por score descendente
        scored_chunks.sort(key=lambda c: c.rerank_score or 0, reverse=True)
        return scored_chunks


class PubChemRetriever(RetrievalStrategy):
    """
    Retrieves compound information from PubChem if 'target_mz' filter is present.
    """

    def __init__(self, pubchem_service: PubChemService):
        self.pubchem = pubchem_service

    def retrieve_context(
        self, query: str, filters: Dict, top_k: int = 5
    ) -> List[ProcessedChunk]:
        mz = filters.get("target_mz") or filters.get("mz")
        if not mz:
            return []

        try:
            mz_val = float(mz)
        except (ValueError, TypeError):
            return []

        print(f"[PubChemRetriever] Searching for m/z: {mz_val}")
        result = self.pubchem.get_compound_by_mz(mz_val)

        if not result:
            return []

        # Convert result to ProcessedChunk
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
                "raw_result": str(result),
            },
        )
        return [chunk]


def reciprocal_rank_fusion(
    results_lists: List[List[ProcessedChunk]], k=60
) -> List[ProcessedChunk]:
    """
    Combines multiple lists of ranked results using Reciprocal Rank Fusion (RRF).
    """
    scores = defaultdict(float)
    chunk_map = {}

    for results in results_lists:
        for rank, chunk in enumerate(results):
            cid = chunk.chunk_id if chunk.chunk_id else chunk.content[:50]

            if cid not in chunk_map:
                chunk_map[cid] = chunk

            scores[cid] += 1.0 / (k + rank + 1)

    sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    final_results = []
    for cid in sorted_cids:
        chunk = chunk_map[cid]
        final_results.append(chunk)

    return final_results


class FederatedRetriever(RetrievalStrategy):
    """
    Executes multiple retrieval strategies and merges results using RRF.
    """

    def __init__(self, strategies: List[RetrievalStrategy]):
        self.strategies = strategies

    def retrieve_context(
        self, query: str, filters: Dict, top_k: int = 5
    ) -> List[ProcessedChunk]:
        results_lists = []
        for strategy in self.strategies:
            results_lists.append(strategy.retrieve_context(query, filters, top_k))

        combined = reciprocal_rank_fusion(results_lists)
        return combined[:top_k]


class CompositionalHybridSearchRetriever(FederatedRetriever):
    """
    Hybrid Search usando dos estrategias subyacentes.
    """

    def __init__(
        self, dense_strategy: RetrievalStrategy, sparse_strategy: RetrievalStrategy
    ):
        super().__init__([dense_strategy, sparse_strategy])


class QueryOptimizerRetriever(RetrievalStrategy):
    """
    Orquesta el proceso de retrieval:
    1. Optimiza el query (rewriting + extracción de filtros).
    2. Delega a la estrategia de retrieval subyacente.
    """

    def __init__(
        self,
        query_processor: QueryProcessingStrategy,
        retrieval_strategy: RetrievalStrategy,
    ):
        self.processor = query_processor
        self.retriever = retrieval_strategy

    def retrieve_context(
        self, query: str, filters: Dict, top_k: int = 5
    ) -> List[ProcessedChunk]:
        # 1. Process/Optimize Query
        structured_query_output = self.processor.process_query(query)

        # 2. Merge user-provided filters with extracted filters
        combined_filters = {**filters, **structured_query_output.metadata_filters}

        optimized_query = structured_query_output.rewritten_query

        print(f"Query reescrita: '{optimized_query}'. Filtros: {combined_filters}")

        # Mostrar metadata extraída (usar getattr seguro)
        target_mz = getattr(structured_query_output, 'target_mz', None)
        target_rt = getattr(structured_query_output, 'target_rt', None)
        
        if target_mz or target_rt:
            print(
                f"[Metadata Extraída]: target_mz={target_mz}, "
                f"target_rt={target_rt}"
            )

        # 3. Retrieve usando la query optimizada
        return self.retriever.retrieve_context(optimized_query, combined_filters, top_k=top_k)


class RetrievalDecorator(RetrievalStrategy):
    """Base Decorator para Retrieval Strategies"""

    def __init__(self, wrapped_strategy: RetrievalStrategy):
        self._wrapped_strategy = wrapped_strategy

    def retrieve_context(
        self, query: str, filters: Dict, top_k: int = 5
    ) -> List[ProcessedChunk]:
        return self._wrapped_strategy.retrieve_context(query, filters, top_k=top_k)


class RerankingDecorator(RetrievalDecorator):
    """
    Fetches more candidates, then reranks using Cross-Encoder.
    """

    def __init__(
        self,
        wrapped_strategy: RetrievalStrategy,
        reranker: RerankerService,
        fetch_k_multiplier: int = 2,
    ):
        super().__init__(wrapped_strategy)
        self.reranker_service = reranker
        self.fetch_k_multiplier = fetch_k_multiplier

    def retrieve_context(
        self, query: str, filters: Dict, top_k: int = 5
    ) -> List[ProcessedChunk]:
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
            chunk.metadata["score"] = float(score)
            chunk.rerank_score = float(score)  # También en el objeto
            scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        print(f"Reordenados. Top score: {scored_chunks[0][1]:.2f}")

        return [c for c, _ in scored_chunks[:top_k]]


class ContextRepackerDecorator(RetrievalDecorator):
    """
    Reorders final context (Sides packing).
    """

    def retrieve_context(
        self, query: str, filters: Dict, top_k: int = 5
    ) -> List[ProcessedChunk]:
        chunks: List[ProcessedChunk] = self._wrapped_strategy.retrieve_context(
            query, filters, top_k=top_k
        )

        if len(chunks) < 3:
            return chunks

        print(f"Aplicando Sides Repacking para {len(chunks)} chunks.")

        # Sides packing: [0, 2, 4, 3, 1]
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