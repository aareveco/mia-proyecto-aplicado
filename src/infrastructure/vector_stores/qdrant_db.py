import numpy as np
from typing import List, Dict, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams
from src.application.ports.vector_store_port import VectorStoreImpl

class QdrantImpl(VectorStoreImpl):

    def __init__(self, collection_name: str = "rag_chunks", path: Optional[str] = None, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Args:
            collection_name: Nombre de la colección en Qdrant.
            path: Ruta local donde persistir los datos. Si es None, usa RAM (:memory:).
            url: URL del servidor Qdrant (opcional, tiene prioridad sobre path).
            api_key: API Key de Qdrant Cloud/Local (opcional).
        """
        self.collection_name = collection_name
        
        if url:
            print(f"[Qdrant] Conectando a servidor: {url}")
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path:
            print(f"[Qdrant] Inicializando en DISCO: {path}")
            self.client = QdrantClient(path=path)
        else:
            print("[Qdrant] Inicializando en MEMORIA (Volátil)")
            self.client = QdrantClient(location=":memory:")

        self._collection_created = False
        
        # Intentamos verificar si la colección ya existe (para no reiniciar ID counter)
        self._next_id = 1 
        try:
            info = self.client.get_collection(self.collection_name)
            self._collection_created = True
            self._next_id = info.points_count + 1
            print(f"[Qdrant] Colección detectada. Próximo ID: {self._next_id}")
        except Exception:
            pass

    def _ensure_collection(self, vector_size: int):
        if self._collection_created:
            return

        # Si no existe, la creamos con configuración HÍBRIDA (Dense + Sparse)
        print(f"[Qdrant] Creando nueva colección HÍBRIDA '{self.collection_name}'...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )
        # Crear índices útiles
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="metadata.publication_year",
            field_schema="integer",
        )
        
        self._collection_created = True
        print(f"[Qdrant] Colección '{self.collection_name}' creada con dim={vector_size}")

    def index_data(self, vectors: np.ndarray, metadatas: List[Dict], overwrite: bool = False) -> None:
        """
        Indexa datos. Espera que 'vectors' sean los dense vectors.
        Busca 'sparse_vector' dentro de 'metadatas' para indexación híbrida.
        """
        if len(vectors) == 0:
            return

        vector_size = vectors.shape[1]

        if overwrite:
            print(f"[Qdrant] Overwrite=True. Recreando colección '{self.collection_name}'...")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                },
            )
            self._collection_created = True
            self._next_id = 1
        else:
            self._ensure_collection(vector_size=vector_size)

        points = []
        for vec, meta in zip(vectors, metadatas):
            # Extraer sparse vector si existe en metadata (inyectado por index_chunks)
            sparse_vec = meta.get("sparse_vector")
            
            # Limpiar sparse_vector de metadata para no duplicar data
            if "sparse_vector" in meta:
                del meta["sparse_vector"]
            
            # Construir vector struct
            # Construir vector struct
            vector_struct = {"dense": vec.tolist()}
            # Sparse vector logic removed as per refactoring plan
            
            points.append(
                PointStruct(
                    id=self._next_id,
                    vector=vector_struct,
                    payload=meta,
                )
            )
            self._next_id += 1

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[Qdrant] Indexados {len(points)} puntos (Dense). (Persistido: {True})")

    def query_data(self, query_vector: np.ndarray, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """
        Realiza búsqueda vectorial densa.
        """
        try:
            # Build Qdrant Filter
            qdrant_filter = None
            if filters:
                 from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
                 conditions = []
                 for key, value in filters.items():
                     # Map filter keys to actual payload fields
                     field_key = f"metadata.{key}" # Default assumption
                     
                     if key == "mz":
                         field_key = "mz_values"
                     elif key == "rt":
                         field_key = "rt_values"
                         
                     if isinstance(value, float):
                         # Range match (works for list containment in Qdrant too)
                         conditions.append(
                            FieldCondition(
                                key=field_key, 
                                range=Range(gte=value*0.99, lte=value*1.01)
                            )
                         )
                     elif isinstance(value, (int, bool, str)):
                         conditions.append(
                            FieldCondition(
                                key=field_key, 
                                match=MatchValue(value=value)
                            )
                         )
                 
                 if conditions:
                    qdrant_filter = Filter(must=conditions)

            # Búsqueda Densa Standard
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                using="dense", # Explicitly use dense vector
                limit=top_k,
                with_payload=True,
                query_filter=qdrant_filter
            )

        except Exception as e:
            print(f"[Qdrant Error] {e}")
            return []

        out: List[Dict] = []
        for p in result.points:
            payload = dict(p.payload or {})
            
            if "metadata" not in payload:
                payload["metadata"] = {}
            
            # Store score
            if isinstance(payload["metadata"], dict):
                payload["metadata"]["score"] = p.score
            else:
                payload["score"] = p.score 
            
            out.append(payload)

        return out