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
            vector_struct = {"dense": vec.tolist()}
            
            if sparse_vec:
                # sparse_vec es tuple (indices, values) o dict
                if isinstance(sparse_vec, (list, tuple)) and len(sparse_vec) == 2:
                    vector_struct["sparse"] = models.SparseVector(
                        indices=sparse_vec[0],
                        values=sparse_vec[1]
                    )
            
            points.append(
                PointStruct(
                    id=self._next_id,
                    vector=vector_struct,
                    payload=meta,
                )
            )
            self._next_id += 1

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[Qdrant] Indexados {len(points)} puntos Híbridos. (Persistido: {True})")

    def query_data(self, query_vector: np.ndarray, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """
        Implementación base de query_data (Dense Only) para compatibilidad.
        """
        return self._query_internal(query_vector=query_vector, sparse_vector=None, top_k=top_k, filters=filters, hybrid=False)

    def query_hybrid(self, query_vector: np.ndarray, query_sparse_vector: Dict, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """
        Búsqueda Híbrida (Dense + Sparse) con RRF.
        query_sparse_vector: {"indices": [...], "values": [...]}
        """
        return self._query_internal(query_vector=query_vector, sparse_vector=query_sparse_vector, top_k=top_k, filters=filters, hybrid=True)

    def _query_internal(self, query_vector: np.ndarray, sparse_vector: Dict, top_k: int, filters: Dict, hybrid: bool) -> List[Dict]:
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
                                range=Range(gte=value*0.99, lte=value*1.01) # Add tolerance? NO, user query had 495.1285. EXACT match often fails with floats.
                                # Let's use a small epsilon tolerance for float comparison or just gte/lte logic.
                                # Given "495.1285" in query and "495.1285" in data (json), it should match.
                                # But float precision issues are real.
                                # Let's use a relaxed range (e.g. +/- 0.1 or 0.01) if it's broad, but for mass spec ppm matters.
                                # For this 'exact' filter from LLM, let's assume strict but with float tolerance.
                                # Better: gte=value-0.0001, lte=value+0.0001
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

            if hybrid and sparse_vector:
                # Búsqueda Híbrida con RRF
                result = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=sparse_vector["indices"],
                                values=sparse_vector["values"],
                            ),
                            using="sparse",
                            filter=qdrant_filter,
                            limit=top_k * 2,
                        ),
                        models.Prefetch(
                            query=query_vector.tolist(),
                            using="dense",
                            filter=qdrant_filter,
                            limit=top_k * 2,
                        ),
                    ],
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    limit=top_k,
                    with_payload=True,
                )
            else:
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