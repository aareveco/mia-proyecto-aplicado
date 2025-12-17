"""
RAG Core - Clases y funciones compartidas
Mantiene 100% la arquitectura del profesor
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Set, Any
import os
from pydantic import BaseModel, Field
import numpy as np

# ============== IMPORTS PARA IMPLEMENTACIONES REALES ==============
import PyPDF2
from pathlib import Path

# OpenAI para LLM
from openai import OpenAI

# Transformers para embeddings y reranking
from sentence_transformers import SentenceTransformer, CrossEncoder

# Qdrant para base de datos vectorial
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    SparseVectorParams,
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue
)

# Chunking
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("ADVERTENCIA: langchain-text-splitters no instalado. Usando chunking básico.")

# BM25 para sparse vectors
from rank_bm25 import BM25Okapi
import json

# ==================================================================

# --- IMPLEMENTACIONES REALES (NO MOCKS) ---

class OpenAILLM:
    """Implementación REAL de LLM usando OpenAI para Query Processing (C6)."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate_structured(self, prompt: str) -> "FilterSuggestion":
        """Llama a OpenAI y extrae query reescrita + filtros"""
        print(f"-> [C6/OpenAI]: Generando Pydantic FilterSuggestion con {self.model}...")
        
        system_prompt = """Eres un asistente especializado en metabolómica y anotación de compuestos bioactivos.
Dado un query sobre features metabolómicas (m/z, RT, bioactividades), debes:
1. Reescribir el query para hacerlo más específico técnicamente (incluir términos como "mass-to-charge ratio", "retention time", "bioactivity", nombres de compuestos)
2. Sugerir filtros de metadata relevantes SOLO de estos campos disponibles:
   - experimental_method (valores posibles: LC-MS, GC-MS, NMR, HPLC)
   - publication_year (año numérico, ejemplo: 2024)

NO sugieras filtros para campos que no están en la lista anterior (como source_database, sample_type, compound_class).

Responde SOLO en formato JSON con esta estructura:
{
    "rewritten_query": "query optimizado con terminología metabolómica",
    "metadata_filters": {"campo": "valor"}
}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query original: {prompt}"}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return FilterSuggestion(
                rewritten_query=result.get("rewritten_query", prompt),
                metadata_filters=result.get("metadata_filters", {})
            )
        except Exception as e:
            print(f"Error en OpenAI: {e}")
            # Fallback
            return FilterSuggestion(rewritten_query=prompt, metadata_filters={})


class CrossEncoderReranker:
    """Implementación REAL de Cross-Encoder para Reranking (Clase 7)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"-> [C7/Cross-Encoder]: Cargando modelo {model_name}...")
        self.model = CrossEncoder(model_name)

    def rank(self, query: str, texts: List[str]) -> List[float]:
        """Reranking REAL usando Cross-Encoder"""
        print(f"-> [C7/Cross-Encoder]: Rerankendo {len(texts)} chunks con modelo real.")
        
        # Crear pares [query, texto]
        pairs = [[query, text] for text in texts]
        
        # Obtener scores del modelo
        scores = self.model.predict(pairs)
        
        return scores.tolist()


# --- 1. CONTRATOS PYDANTIC (Clases 2, 3, 5, 6 y 7) ---

class ProcessedChunk(BaseModel):
    """El contrato de datos central (C2) que se enriquece (C3 y C7)."""

    content: str
    source_file: str
    publication_year: int
    experimental_method: str  # LC-MS, GC-MS, NMR, etc.
    chunk_id: str
    dense_vector: list[float] | None = None
    sparse_vector: tuple[list[int], list[float]] | None = None  # (indices, values)
    rerank_score: float | None = None  # <- ENRIQUECIMIENTO C7


class BenchmarkEntry(BaseModel):
    """El contrato del Golden Dataset para la evaluación (C5)."""

    query: str
    reference_answer: str
    relevant_chunk_ids: Set[str]


class FilterSuggestion(BaseModel):
    """Contrato Pydantic para el Output de Query Processing (C6)."""

    rewritten_query: str
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


# --- 2. CLASE: PATRÓN FACTORY (Creación) ---


class AbstractLoader(ABC):
    @abstractmethod
    def load_and_chunk(self, path: str) -> List[ProcessedChunk]:
        pass


class PDFLoader(AbstractLoader):
    """Loader REAL para PDFs con extracción de metadata"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_and_metadata(self, path: str) -> tuple[str, dict]:
        """Extrae texto y metadata del PDF"""
        text = ""
        metadata = {
            "publication_year": 2024,  # Default
            "experimental_method": "Unknown"
        }
        
        try:
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Intentar extraer metadata
                if pdf_reader.metadata:
                    # Buscar año en metadata
                    if pdf_reader.metadata.creation_date:
                        metadata["publication_year"] = pdf_reader.metadata.creation_date.year
                
                # Extraer texto
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Buscar método experimental en el texto (heurística para metabolómica)
                text_lower = text.lower()
                if "lc-ms" in text_lower or "liquid chromatography" in text_lower or "uplc" in text_lower:
                    metadata["experimental_method"] = "LC-MS"
                elif "gc-ms" in text_lower or "gas chromatography" in text_lower:
                    metadata["experimental_method"] = "GC-MS"
                elif "nmr" in text_lower or "resonancia magnética" in text_lower:
                    metadata["experimental_method"] = "NMR"
                elif "hplc" in text_lower or "high performance" in text_lower:
                    metadata["experimental_method"] = "HPLC"
                    
        except Exception as e:
            print(f"Error leyendo PDF {path}: {e}")
            return "", metadata
        
        return text, metadata
    
    def chunk_text(self, text: str, source_file: str, metadata: dict) -> List[ProcessedChunk]:
        """Divide el texto en chunks"""
        chunks = []
        
        if LANGCHAIN_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            text_chunks = text_splitter.split_text(text)
        else:
            # Chunking básico
            text_chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                text_chunks.append(text[i:i + self.chunk_size])
        
        # Crear ProcessedChunk para cada chunk
        for idx, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                chunks.append(
                    ProcessedChunk(
                        content=chunk_text.strip(),
                        source_file=source_file,
                        publication_year=metadata["publication_year"],
                        experimental_method=metadata["experimental_method"],
                        chunk_id=f"{Path(source_file).stem}-chunk-{idx}",
                    )
                )
        
        return chunks
    
    def load_and_chunk(self, path: str) -> List[ProcessedChunk]:
        print(f"-> [C2/PDFLoader]: Aplicando Layout-Aware Chunking a {path}")
        
        # 1. Extraer texto y metadata del PDF
        full_text, metadata = self.extract_text_and_metadata(path)
        
        if not full_text:
            print(f"ADVERTENCIA: No se pudo extraer texto de {path}")
            return []
        
        # 2. Dividir en chunks
        chunks = self.chunk_text(full_text, path, metadata)
        
        print(f"   -> Extraídos {len(chunks)} chunks de {path}")
        return chunks

    
class MarkdownLoader(AbstractLoader):
    """Loader REAL para Markdown"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_and_chunk(self, path: str) -> List[ProcessedChunk]:
        print(f"-> [C2/MarkdownLoader]: Aplicando Header-Split Chunking a {path}")
        
        chunks = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Dividir por headers (##)
            sections = text.split('\n## ')
            
            for idx, section in enumerate(sections):
                if section.strip():
                    chunks.append(
                        ProcessedChunk(
                            content=section.strip(),
                            source_file=path,
                            publication_year=2024,
                            experimental_method="Unknown",
                            chunk_id=f"{Path(path).stem}-md-{idx}",
                        )
                    )
        except Exception as e:
            print(f"Error leyendo Markdown {path}: {e}")
        
        return chunks


class DocumentLoaderFactory:
    @staticmethod
    def get_loader(path: str, **kwargs) -> AbstractLoader:
        match path:
            case path if path.endswith(".pdf"):
                return PDFLoader(**kwargs)
            case path if path.endswith(".md"):
                return MarkdownLoader(**kwargs)
            case _:
                raise ValueError(f"Formato no soportado: {path}")
            

# --- 3. CLASE 3: PATRÓN ADAPTER (Traducción de Embeddings) ---

class SentenceTransformerAPI:
    """API REAL de Sentence Transformers"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"-> Cargando modelo de embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


class BioBERTAdapter:
    """Adapter REAL para embeddings densos"""
    def __init__(self, model: SentenceTransformerAPI):
        self.model = model
    
    def embed_chunks(self, chunks: list[ProcessedChunk]) -> np.ndarray:
        texts = [c.content for c in chunks]
        print(f"-> [C3/Adapter]: Generando embeddings densos para {len(texts)} chunks")
        return self.model.encode(texts)


class BM25API:
    """API REAL de BM25 para sparse vectors"""
    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = []
        
    def fit(self, texts: list[str]):
        """Ajustar BM25 con el corpus"""
        self.tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def get_sparse_vector(self, texts: list[str]) -> np.ndarray:
        """Obtener sparse vectors (scores BM25 contra el corpus)"""
        if self.bm25 is None:
            # Si no está fitted, retornar vectores vacíos
            return np.zeros((len(texts), 100))
        
        vectors = []
        for text in texts:
            tokenized = text.lower().split()
            scores = self.bm25.get_scores(tokenized)
            vectors.append(scores)
        
        return np.array(vectors)
    
    def encode(self, query: str) -> dict:
        """Codificar query a formato sparse para Qdrant"""
        if self.bm25 is None:
            return {"indices": [0], "values": [0.0]}
        
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        
        # Obtener índices no-cero
        non_zero_indices = np.where(scores > 0)[0]
        non_zero_values = scores[non_zero_indices]
        
        if len(non_zero_indices) == 0:
            return {"indices": [0], "values": [0.0]}
        
        return {
            "indices": non_zero_indices.tolist(),
            "values": non_zero_values.tolist()
        }


class BM25Adapter:
    """Adapter REAL para BM25"""
    def __init__(self, model: BM25API):
        self.model = model
    
    def embed_chunks(self, chunks: list[ProcessedChunk]) -> list[tuple]:
        texts = [c.content for c in chunks]
        print(f"-> [C3/Adapter]: Generando sparse vectors BM25 para {len(texts)} chunks")
        
        # Primero ajustamos BM25 con el corpus
        self.model.fit(texts)
        
        # Convertir a formato sparse (índices, valores)
        dense_vectors = self.model.get_sparse_vector(texts)
        sparse_vectors = []
        
        for vec in dense_vectors:
            # Obtener índices de valores no-cero y sus valores
            non_zero_indices = np.where(vec > 0)[0]
            non_zero_values = vec[non_zero_indices]
            
            # Si no hay valores, usar al menos un índice
            if len(non_zero_indices) == 0:
                non_zero_indices = np.array([0])
                non_zero_values = np.array([0.0])
            
            sparse_vectors.append((non_zero_indices.tolist(), non_zero_values.tolist()))
        
        return sparse_vectors
    
# --- 4. CLASE 4: PATRÓN STRATEGY (Algoritmos de Búsqueda) ---

class QdrantVectorStore:
    """Implementación REAL con Qdrant"""
    
    def __init__(self, collection_name: str = "rag_collection", url: str = "http://localhost:6333", api_key: str = None, bm25_encoder=None):
        print(f"-> [C4/Qdrant]: Conectando a Qdrant en {url}")
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = None
        self.bm25_encoder = bm25_encoder

        
    def index_data(self, chunks: List[ProcessedChunk]):
        """Indexa chunks en Qdrant"""
        if not chunks:
            return
        
        # Determinar tamaño del vector del primer chunk
        if chunks[0].dense_vector:
            self.vector_size = len(chunks[0].dense_vector)
        else:
            print("ERROR: Los chunks no tienen vectores densos")
            return
        
        # Crear o recrear colección
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except:
            pass
        
        # Crear colección con dense + sparse vectors
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )
        
        # Crear índices para los campos de filtrado
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="experimental_method",
            field_schema="keyword",
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="publication_year",
            field_schema="integer",
        )
        
        print(f"[C4/Qdrant]: Índices creados para experimental_method y publication_year")
        
        # Preparar puntos para Qdrant
        points = []
        for chunk in chunks:
            points.append(
                PointStruct(
                    id=hash(chunk.chunk_id) & 0x7FFFFFFF,
                    vector={
                        "dense": chunk.dense_vector,
                        "sparse": models.SparseVector(
                            indices=chunk.sparse_vector[0],
                            values=chunk.sparse_vector[1],
                        ),
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "source_file": chunk.source_file,
                        "publication_year": chunk.publication_year,
                        "experimental_method": chunk.experimental_method,
                    },
                )
            )

        
        # Insertar en batch
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"[C4/Qdrant]: Indexados {len(chunks)} chunks en colección '{self.collection_name}'")

    def query_hybrid(self, query: str, query_vector: list[float], filters: Dict, k: int = 10) -> List[Dict]:
        """Búsqueda híbrida REAL en Qdrant"""
        print(f"[Qdrant Query]: Buscando con filtros: {filters}")
        
        # Construir filtro de Qdrant
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Obtener sparse vector del query
        query_sparse = self.bm25_encoder.encode(query)
        
        # Búsqueda híbrida con query_points + prefetch + RRF
        result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse["indices"],
                        values=query_sparse["values"],
                    ),
                    using="sparse",
                    filter=qdrant_filter,
                    limit=k * 2,
                ),
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    filter=qdrant_filter,
                    limit=k * 2,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=k,
            with_payload=True,
        )
        
        # Convertir resultados
        results = []
        for point in result.points:
            results.append(point.payload)
        
        return results


class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve_context(
        self, query: str, filters: Dict, k: int
    ) -> List[ProcessedChunk]:
        pass


class HybridSearchStrategy(RetrievalStrategy):
    """Strategy REAL con Qdrant y embeddings"""
    
    def __init__(self, vector_store: QdrantVectorStore, embedding_model: SentenceTransformerAPI):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve_context(
        self, query: str, filters: Dict, k: int = 3
    ) -> List[ProcessedChunk]:
        print(f"-> [C4/Strategy]: Ejecutando Búsqueda HÍBRIDA (componente base).")
        
        # 1. Embeddings del query
        query_vector = self.embedding_model.encode([query])[0].tolist()
        
        # 2. Búsqueda en Qdrant
        results_dict = self.vector_store.query_hybrid(query, query_vector, filters, k=k)
        
        # 3. Convertir a ProcessedChunk
        chunks = []
        for r in results_dict:
            chunks.append(
                ProcessedChunk(
                    content=r["content"],
                    source_file=r["source_file"],
                    publication_year=r["publication_year"],
                    experimental_method=r["experimental_method"],
                    chunk_id=r["chunk_id"]
                )
            )
        
        return chunks


# --- 5. CLASE 6: PATRÓN STRATEGY REFINADO (Query Processing) + C4 INTEGRATION ---

class QueryProcessingStrategy(ABC):
    @abstractmethod
    def process_query(self, query: str) -> FilterSuggestion:
        pass


class QueryRewritingStrategy(QueryProcessingStrategy):
    """Strategy REAL con OpenAI"""
    
    def __init__(self, llm: OpenAILLM):
        self.llm = llm

    def process_query(self, query: str) -> FilterSuggestion:
        """Retorna un objeto Pydantic con la query optimizada y filtros."""
        return self.llm.generate_structured(
            prompt=f"Optimiza esta consulta científica: {query}"
        )
    
class QueryOptimizerRetriever(RetrievalStrategy):
    """
    Strategy Refinado (C6) + Contexto (C4): Une el output Pydantic del LLM
    con el input del Retrieval Strategy (C4).
    """

    def __init__(
        self,
        query_processor: QueryProcessingStrategy,
        retrieval_strategy: RetrievalStrategy,
    ):
        self.query_processor = query_processor
        self.retriever = retrieval_strategy

    def retrieve_context(
        self, query: str, filters: Dict, k: int
    ) -> tuple[List[ProcessedChunk], FilterSuggestion]:
        """Retorna (chunks, filter_suggestion) para tracking"""
        # 1. Ejecutar el Strategy Refinado (C6) para obtener el Pydantic FilterSuggestion
        structured_query_output = self.query_processor.process_query(query)

        # 2. Combinar filtros iniciales con los filtros sugeridos por el LLM
        combined_filters = {**filters, **structured_query_output.metadata_filters}
        print(
            f"[C6 Output]: Query reescrita: '{structured_query_output.rewritten_query}'. Filtros: {combined_filters}"
        )

        # 3. Delegar al Retriever base, usando la query reescrita y los filtros combinados
        chunks = self.retriever.retrieve_context(
            structured_query_output.rewritten_query, combined_filters, k
        )
        
        return chunks, structured_query_output
    
# --- 6. CLASE 7: PATRÓN DECORADOR AVANZADO (Apilamiento) ---

# Clase Base del Decorador (Respeta la interfaz RetrievalStrategy)
class RetrievalDecorator(RetrievalStrategy):
    def __init__(self, wrapped_strategy: RetrievalStrategy):
        self._wrapped = wrapped_strategy

    def retrieve_context(
        self, query: str, filters: Dict, k: int
    ) -> List[ProcessedChunk]:
        # Por defecto, solo delega. La funcionalidad se añade en las subclases.
        return self._wrapped.retrieve_context(query, filters, k)
    

# Decorator A: Reranking (Post-Retrieval) - REAL
class RerankingDecorator(RetrievalDecorator):
    def __init__(
        self,
        wrapped_strategy: RetrievalStrategy,
        reranker: CrossEncoderReranker
    ):
        super().__init__(wrapped_strategy)
        self._reranker = reranker

    def retrieve_context(
        self, query: str, filters: Dict, k: int
    ) -> List[ProcessedChunk]:
        # 1. Delegación: Obtiene los candidatos del componente envuelto (C4+C6)
        result = self._wrapped.retrieve_context(query, filters, k=k * 2)
        
        # Manejar si retorna tupla (QueryOptimizerRetriever) o lista (otros)
        if isinstance(result, tuple):
            chunks_candidates, filter_suggestion = result
        else:
            chunks_candidates = result
            filter_suggestion = None

        if not chunks_candidates:
            return ([], filter_suggestion) if filter_suggestion else []

        # 2. Adición de funcionalidad: Reranking REAL
        texts_to_rank = [c.content for c in chunks_candidates]
        scores = self._reranker.rank(query, texts_to_rank)

        # 3. Asociar Scores a los objetos Pydantic (Enriquecimiento Pydantic)
        scored_chunks = []
        for chunk, score in zip(chunks_candidates, scores):
            chunk.rerank_score = float(score)  # Llenamos el campo C7
            scored_chunks.append(chunk)

        # Reordenamiento (del mayor score al menor)
        scored_chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        print(
            f"[C7 Reranking]: Reordenados. Top Score: {scored_chunks[0].rerank_score:.2f}"
        )

        final_chunks = scored_chunks[:k]
        return (final_chunks, filter_suggestion) if filter_suggestion else final_chunks
    
# Decorador B: Context Repacker (Optimización Posicional para el LLM)
class ContextRepackerDecorator(RetrievalDecorator):
    def retrieve_context(
        self, query: str, filters: Dict, k: int
    ) -> List[ProcessedChunk]:

        # 1. Delegación: Obtiene los chunks ya ordenados (del Reranker)
        result = self._wrapped.retrieve_context(query, filters, k)
        
        # Manejar si retorna tupla o lista
        if isinstance(result, tuple):
            chunks, filter_suggestion = result
        else:
            chunks = result
            filter_suggestion = None

        if len(chunks) < 3:
            return (chunks, filter_suggestion) if filter_suggestion else chunks

        print(f"[C7 Repacker]: Aplicando Sides Repacking para {len(chunks)} chunks.")

        # 2. Lógica de Reempaquetamiento (ej. [Top] + [Resto del medio])
        top_chunk = chunks[0]  # El mejor chunk (ahora con el score C7 más alto)
        rest = chunks[1:]

        # Nueva estructura para el prompt: [Top 1] + [Resto]
        repacked_chunks = [top_chunk] + rest

        return (repacked_chunks, filter_suggestion) if filter_suggestion else repacked_chunks
    
# --- 7. FUNCIONES DE UTILIDAD ---

def run_embedding_pipeline(chunks: List[ProcessedChunk], dense_adapter, sparse_adapter):
    """Pipeline REAL de embeddings"""
    print("\n-> Iniciando pipeline de embeddings...")
    dense_v = dense_adapter.embed_chunks(chunks)
    sparse_v = sparse_adapter.embed_chunks(chunks)  # Retorna lista de tuplas (indices, values)
    
    for i, chunk in enumerate(chunks):
        chunk.dense_vector = dense_v[i].tolist()
        chunk.sparse_vector = sparse_v[i]  # Ya es tupla (indices, values)
    
    print(f"-> Pipeline completado: {len(chunks)} chunks con vectores generados")
    return chunks


def run_indexing_service(
    file_path: str,
    dense_adapter,
    sparse_adapter,
    vector_store,
    **loader_kwargs
):
    """Servicio de indexación REAL"""
    loader = DocumentLoaderFactory.get_loader(file_path, **loader_kwargs)
    chunks_pydantic = loader.load_and_chunk(file_path)
    
    if not chunks_pydantic:
        print(f"ADVERTENCIA: No se generaron chunks para {file_path}")
        return
    
    chunks_with_vectors = run_embedding_pipeline(
        chunks_pydantic, dense_adapter, sparse_adapter
    )
    vector_store.index_data(chunks_with_vectors)
