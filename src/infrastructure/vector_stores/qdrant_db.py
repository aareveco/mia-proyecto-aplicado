import numpy as np
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.application.ports.vector_store_port import VectorStoreImpl

class QdrantImpl(VectorStoreImpl):

    def __init__(self, collection_name: str = "rag_chunks", path: Optional[str] = None):
        """
        Args:
            collection_name: Nombre de la colección en Qdrant.
            path: Ruta local donde persistir los datos. Si es None, usa RAM (:memory:).
        """
        self.collection_name = collection_name
        
        if path:
            print(f"[Qdrant] Inicializando en DISCO: {path}")
            self.client = QdrantClient(path=path)
        else:
            print("[Qdrant] Inicializando en MEMORIA (Volátil)")
            self.client = QdrantClient(location=":memory:")

        self._collection_created = False
        
        # Intentamos verificar si la colección ya existe (para no reiniciar ID counter)
        # Nota: En un entorno productivo real, gestionarías los IDs de otra forma (ej: UUIDs)
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

        # Si no existe, la creamos
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        self._collection_created = True
        print(f"[Qdrant] Colección '{self.collection_name}' creada con dim={vector_size}")

    def index_data(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        if len(vectors) == 0:
            return

        self._ensure_collection(vector_size=vectors.shape[1])

        points = []
        for vec, meta in zip(vectors, metadata):
            points.append(
                PointStruct(
                    id=self._next_id,
                    vector=vec.tolist(),
                    payload=meta,
                )
            )
            self._next_id += 1

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[Qdrant] Indexados {len(points)} puntos. (Persistido: {True})")

    def query_data(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            # Si consultamos antes de indexar nada
            print(f"[Qdrant Error] {e}")
            return []

        out: List[Dict] = []
        for p in result.points:
            payload = dict(p.payload or {})
            
            # --- START CHANGE ---
            # Create metadata dict if it doesn't exist (depending on your ingestion structure)
            if "metadata" not in payload:
                payload["metadata"] = {}
            
            # Store score INSIDE metadata so ProcessedChunk can see it
            if isinstance(payload["metadata"], dict):
                payload["metadata"]["score"] = p.score
            else:
                # Fallback if metadata is flattened
                payload["score"] = p.score 
            # --- END CHANGE ---
            
            out.append(payload)

        return out