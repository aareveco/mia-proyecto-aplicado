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
    MatchValue,
    Range
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
        
        system_prompt = """Eres un asistente especializado en metabolómica.

Analiza el query y extrae:
1. Reescribir con terminología técnica (m/z → mass-to-charge ratio, RT → retention time)
2. Extraer valores para búsqueda:
   - target_mz: valor m/z si se menciona (ej: 449.107)
   - target_rt: valor RT si se menciona (ej: 8.2)
3. Extraer filtros Qdrant (solo campos indexados):
   - publication_year: año si se menciona (campo indexado en Qdrant)

IMPORTANTE: target_mz y target_rt NO van en metadata_filters (no son campos indexados en Qdrant).

Responde SOLO en JSON:
{
    "rewritten_query": "query con terminología técnica",
    "metadata_filters": {
        "publication_year": 2024
    },
    "target_mz": 449.107,
    "target_rt": 8.2
}

Si no hay valores, usa null u omitir del objeto."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Query original: Feature con m/z 449.107 y RT 8.2 min en Té Verde"},
                    {"role": "assistant", "content": '{"rewritten_query": "Característica metabolómica con mass-to-charge ratio (m/z) de 449.107 y retention time (RT) de 8.2 minutos en Té Verde", "metadata_filters": {}, "target_mz": 449.107, "target_rt": 8.2}'},
                    {"role": "user", "content": f"Query original: {prompt}"}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Extraer valores
            target_mz = result.get("target_mz")
            target_rt = result.get("target_rt")
            metadata_filters = result.get("metadata_filters", {})
            
            # Asegurar que target_mz y target_rt estén en metadata_filters
            if target_mz is not None and "target_mz" not in metadata_filters:
                metadata_filters["target_mz"] = target_mz
            if target_rt is not None and "target_rt" not in metadata_filters:
                metadata_filters["target_rt"] = target_rt
            
            return FilterSuggestion(
                rewritten_query=result.get("rewritten_query", prompt),
                metadata_filters=metadata_filters,
                target_mz=target_mz,
                target_rt=target_rt
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
    chunk_id: str
    dense_vector: list[float] | None = None
    sparse_vector: tuple[list[int], list[float]] | None = None  # (indices, values)
    rerank_score: float | None = None  # <- ENRIQUECIMIENTO C7
    
    # Metadata estructurada (para metabolómica)
    mz_values: list[float] | None = None
    rt_values: list[float] | None = None
    compound_names: list[str] | None = None
    bioactivities: list[str] | None = None


class BenchmarkEntry(BaseModel):
    """El contrato del Golden Dataset para la evaluación (C5)."""

    query: str
    reference_answer: str
    relevant_chunk_ids: Set[str]


class FilterSuggestion(BaseModel):
    """Contrato Pydantic para el Output de Query Processing (C6)."""

    rewritten_query: str
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    
    # Metadata extraída del query para post-filtering
    target_mz: float | None = None
    target_rt: float | None = None


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
                    
        except Exception as e:
            print(f"Error leyendo PDF {path}: {e}")
            return "", metadata
        
        return text, metadata
    
    def extract_structured_metadata(self, text: str) -> dict:
        """Extrae metadata estructurada del chunk usando LLM"""
        
        # Si el chunk es muy corto o no relevante, retornar vacío
        if len(text.strip()) < 50:
            return {
                "mz_values": None,
                "rt_values": None,
                "compound_names": None,
                "bioactivities": None
            }
        
        try:
            from openai import OpenAI
            import json
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            system_prompt = """Eres un experto en metabolómica. Extrae información estructurada del texto científico.

Extrae:
1. mz_values: Lista de valores m/z (mass-to-charge ratio). Busca patrones como "m/z 449.107", "mz: 449.1", etc.
2. rt_values: Lista de valores RT (retention time) en minutos. Busca "RT 8.2", "retention time 8.2 min", etc.
3. compound_names: Lista de nombres de compuestos químicos mencionados (Myricetina, Quercetina, etc.)
4. bioactivities: Lista de bioactividades mencionadas (antioxidant, antidiabetic, anti-inflammatory, etc.)

Responde SOLO en formato JSON:
{
    "mz_values": [449.107, ...] o null,
    "rt_values": [8.2, ...] o null,
    "compound_names": ["Myricetina", ...] o null,
    "bioactivities": ["antioxidant", ...] o null
}

Si no encuentras información para algún campo, usa null."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Texto:\n{text[:1500]}"}  # Limitar a 1500 chars
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "mz_values": result.get("mz_values"),
                "rt_values": result.get("rt_values"),
                "compound_names": result.get("compound_names"),
                "bioactivities": result.get("bioactivities")
            }
            
        except Exception as e:
            # Fallback silencioso: retornar None para todos los campos
            return {
                "mz_values": None,
                "rt_values": None,
                "compound_names": None,
                "bioactivities": None
            }
    
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
        
        print(f"   -> Extrayendo metadata estructurada de {len(text_chunks)} chunks con LLM...")
        
        # Crear ProcessedChunk para cada chunk
        for idx, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                # Extraer metadata estructurada del chunk con LLM
                if idx % 10 == 0:  # Mostrar progreso cada 10 chunks
                    print(f"      Procesando chunk {idx+1}/{len(text_chunks)}...")
                
                structured_meta = self.extract_structured_metadata(chunk_text)
                
                chunks.append(
                    ProcessedChunk(
                        content=chunk_text.strip(),
                        source_file=source_file,
                        publication_year=metadata["publication_year"],
                        chunk_id=f"{Path(source_file).stem}-chunk-{idx}",
                        mz_values=structured_meta["mz_values"],
                        rt_values=structured_meta["rt_values"],
                        compound_names=structured_meta["compound_names"],
                        bioactivities=structured_meta["bioactivities"]
                    )
                )
        
        print(f"   -> Metadata estructurada extraída para {len(chunks)} chunks")
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

        
    def index_data(self, chunks: List[ProcessedChunk], recreate_collection: bool = False):
        """
        Indexa chunks en Qdrant
        
        Args:
            chunks: Lista de chunks a indexar
            recreate_collection: Si True, elimina y recrea la colección. Si False, agrega a la existente.
        """
        if not chunks:
            return
        
        # Determinar tamaño del vector del primer chunk
        if chunks[0].dense_vector:
            self.vector_size = len(chunks[0].dense_vector)
        else:
            print("ERROR: Los chunks no tienen vectores densos")
            return
        
        # Verificar si la colección existe
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
        except:
            collection_exists = False
        
        # Crear o recrear colección según parámetro
        if recreate_collection or not collection_exists:
            if collection_exists:
                print(f"[C4/Qdrant]: Eliminando colección existente '{self.collection_name}'...")
                self.client.delete_collection(collection_name=self.collection_name)
            
            print(f"[C4/Qdrant]: Creando nueva colección '{self.collection_name}'...")
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
            
            # Crear índice para publication_year
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="publication_year",
                field_schema="integer",
            )
            
            print(f"[C4/Qdrant]: Índice creado para publication_year")
        else:
            print(f"[C4/Qdrant]: Usando colección existente '{self.collection_name}'")
        
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
                        "mz_values": chunk.mz_values,
                        "rt_values": chunk.rt_values,
                        "compound_names": chunk.compound_names,
                        "bioactivities": chunk.bioactivities,
                    },
                )
            )

        
        # Insertar en batch
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"[C4/Qdrant]: Indexados {len(chunks)} chunks en colección '{self.collection_name}'")
    
    def collection_exists(self) -> bool:
        """Verifica si la colección existe"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except:
            return False
    
    def get_collection_info(self) -> dict:
        """Obtiene información de la colección"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "points_count": info.points_count if hasattr(info, 'points_count') else 0,
                "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else 0,
                "indexed_vectors_count": info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else 0,
                "status": str(info.status) if hasattr(info, 'status') else "unknown"
            }
        except Exception as e:
            print(f"[C4/Qdrant]: Error al obtener info de colección: {e}")
            return None
    
    def delete_collection(self):
        """Elimina la colección completa"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"[C4/Qdrant]: Colección '{self.collection_name}' eliminada")
            return True
        except Exception as e:
            print(f"[C4/Qdrant]: Error al eliminar colección: {e}")
            return False
    
    def list_documents(self) -> List[str]:
        """Lista todos los documentos únicos en la colección"""
        try:
            # Hacer scroll para obtener todos los puntos
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            # Extraer nombres únicos de archivos
            documents = set()
            for point in points:
                if 'source_file' in point.payload:
                    documents.add(point.payload['source_file'])
            
            return sorted(list(documents))
        except:
            return []
    
    def delete_document(self, source_file: str) -> int:
        """
        Elimina todos los chunks de un documento específico
        
        Args:
            source_file: Nombre del archivo a eliminar
            
        Returns:
            Número de chunks eliminados
        """
        try:
            # Buscar todos los puntos del documento
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=source_file)
                        )
                    ]
                ),
                limit=10000,
                with_payload=False,
                with_vectors=False
            )
            
            # Extraer IDs
            point_ids = [point.id for point in points]
            
            if point_ids:
                # Eliminar puntos
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                print(f"[C4/Qdrant]: Eliminados {len(point_ids)} chunks del documento '{source_file}'")
            
            return len(point_ids)
        except Exception as e:
            print(f"[C4/Qdrant]: Error al eliminar documento: {e}")
            return 0

    def query_hybrid(self, query: str, query_vector: list[float], filters: Dict, k: int = 10) -> List[Dict]:
        """Búsqueda híbrida REAL en Qdrant con filtros opcionales"""
        
        # Separar filtros numéricos (opcional) de filtros exactos (must)
        must_conditions = []
        should_conditions = []
        
        if filters:
            for key, value in filters.items():
                # Filtros numéricos opcionales (boost si coincide, pero no requerido)
                if key == "target_mz" and isinstance(value, (int, float)):
                    # Nota: No filtramos por m/z porque puede excluir resultados relevantes
                    # En su lugar, lo usamos en el reranking posterior
                    pass
                elif key == "target_rt" and isinstance(value, (int, float)):
                    # Nota: No filtramos por RT porque puede excluir resultados relevantes
                    pass
                # Filtros exactos (must)
                elif isinstance(value, (str, int)):
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
        
        # Construir filtro final
        qdrant_filter = None
        if must_conditions:
            qdrant_filter = Filter(must=must_conditions)
        
        # Log: Si hay filtros numéricos, informar que se usarán en post-procesamiento
        if filters.get("target_mz") or filters.get("target_rt"):
            print(f"[Qdrant Info]: Filtros numéricos (m/z, RT) se aplicarán en post-procesamiento")

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
        
        # 2. Búsqueda en Qdrant (sin filtros numéricos estrictos)
        results_dict = self.vector_store.query_hybrid(query, query_vector, filters, k=k*3)  # Recuperar más para filtrar
        
        # 3. Convertir a ProcessedChunk
        chunks = []
        for r in results_dict:
            chunks.append(
                ProcessedChunk(
                    content=r["content"],
                    source_file=r["source_file"],
                    publication_year=r["publication_year"],
                    chunk_id=r["chunk_id"],
                    mz_values=r.get("mz_values"),
                    rt_values=r.get("rt_values"),
                    compound_names=r.get("compound_names"),
                    bioactivities=r.get("bioactivities"),
                )
            )
        
        # 4. Post-filtering opcional por m/z y RT (si están en filtros)
        target_mz = filters.get("target_mz")
        target_rt = filters.get("target_rt")
        
        if target_mz or target_rt:
            filtered_chunks = []
            for chunk in chunks:
                score = 0
                
                # Boost por m/z match
                if target_mz and chunk.mz_values:
                    for mz in chunk.mz_values:
                        if abs(mz - target_mz) <= 0.01:  # Tolerancia ±0.01
                            score += 10
                            break
                
                # Boost por RT match
                if target_rt and chunk.rt_values:
                    for rt in chunk.rt_values:
                        if abs(rt - target_rt) <= 0.5:  # Tolerancia ±0.5 min
                            score += 5
                            break
                
                # Incluir todos los chunks pero dar prioridad a los que coinciden
                chunk.rerank_score = score  # Temporal score para ordenamiento
                filtered_chunks.append(chunk)
            
            # Ordenar por score (mayor primero)
            filtered_chunks.sort(key=lambda c: c.rerank_score or 0, reverse=True)
            chunks = filtered_chunks[:k]  # Limitar a k
            
            print(f"[C4 Post-Filter]: {len(chunks)} chunks ordenados por similitud m/z/RT")
        
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
        # Pasar el query directamente, el prompt está en generate_structured
        return self.llm.generate_structured(prompt=query)
    
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
        
        # Mostrar metadata extraída
        if structured_query_output.target_mz or structured_query_output.target_rt:
            print(f"[C6 Metadata]: target_mz={structured_query_output.target_mz}, target_rt={structured_query_output.target_rt}")

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
    recreate_collection: bool = False,
    **loader_kwargs
):
    """
    Servicio de indexación REAL
    
    Args:
        file_path: Ruta al archivo a indexar
        dense_adapter: Adapter para embeddings densos
        sparse_adapter: Adapter para embeddings sparse
        vector_store: QdrantVectorStore
        recreate_collection: Si True, elimina colección antes de indexar
        **loader_kwargs: Argumentos para el loader (chunk_size, etc.)
    """
    loader = DocumentLoaderFactory.get_loader(file_path, **loader_kwargs)
    chunks_pydantic = loader.load_and_chunk(file_path)
    
    if not chunks_pydantic:
        print(f"ADVERTENCIA: No se generaron chunks para {file_path}")
        return
    
    chunks_with_vectors = run_embedding_pipeline(
        chunks_pydantic, dense_adapter, sparse_adapter
    )
    vector_store.index_data(chunks_with_vectors, recreate_collection=recreate_collection)