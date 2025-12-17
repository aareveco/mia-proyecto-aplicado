# src/infrastructure/vector_stores/qdrant_db.py
import numpy as np
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, Range
from src.application.ports.vector_store_port import VectorStoreImpl


class QdrantImpl(VectorStoreImpl):
    def __init__(
        self, collection_name: str = "rag_chunks", path: Optional[str] = None
    ):
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

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        
        # Crear índices para filtros
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="publication_year",
                field_schema="integer",
            )
            print(f"[Qdrant] Índice creado para publication_year")
        except:
            pass
        
        self._collection_created = True
        print(
            f"[Qdrant] Colección '{self.collection_name}' creada con dim={vector_size}"
        )

    def index_data(
        self, vectors: np.ndarray, metadatas: List[Dict], overwrite: bool = False
    ) -> None:
        if len(vectors) == 0:
            return

        if overwrite:
            print(
                f"[Qdrant] Overwrite=True. Recreando colección '{self.collection_name}'..."
            )
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vectors.shape[1],
                    distance=Distance.COSINE,
                ),
            )
            self._collection_created = True
            self._next_id = 1
        else:
            self._ensure_collection(vector_size=vectors.shape[1])

        points = []
        for vec, meta in zip(vectors, metadatas):
            # Extraer campos de primer nivel para facilitar búsqueda
            payload = {
                "content": meta.get("content", ""),
                "source_file": meta.get("source_file", ""),
                "publication_year": meta.get("publication_year", 2024),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id", ""),
                "type": meta.get("type", "text"),
                # Metadata estructurada
                "mz_values": meta.get("mz_values"),
                "rt_values": meta.get("rt_values"),
                "compound_names": meta.get("compound_names"),
                "bioactivities": meta.get("bioactivities"),
                # Metadata general
                "metadata": meta.get("metadata", {}),
            }

            points.append(
                PointStruct(
                    id=self._next_id,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )
            self._next_id += 1

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[Qdrant] Indexados {len(points)} puntos.")

    def query_data(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Dict = None,
    ) -> List[Dict]:
        try:
            # Build Qdrant Filter
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    # Ignorar filtros numéricos (se manejan en post-filtering)
                    if key in ["target_mz", "target_rt"]:
                        continue

                    # Filtros exactos
                    if isinstance(value, float):
                        conditions.append(
                            FieldCondition(
                                key=key, range=Range(gte=value, lte=value)
                            )
                        )
                    elif isinstance(value, (int, bool, str)):
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

                if conditions:
                    qdrant_filter = Filter(must=conditions)

            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
                query_filter=qdrant_filter,
            )
        except Exception as e:
            print(f"[Qdrant Error] {e}")
            return []

        out: List[Dict] = []
        for p in result.points:
            payload = dict(p.payload or {})

            # Asegurar que todos los campos de ProcessedChunk existen
            # Esto es para backward compatibility con índices antiguos
            if "metadata" not in payload:
                payload["metadata"] = {}

            # Store score INSIDE metadata
            if isinstance(payload["metadata"], dict):
                payload["metadata"]["score"] = p.score
            else:
                payload["score"] = p.score
            
            # Asegurar campos de metadata estructurada existen (backward compatibility)
            if "mz_values" not in payload:
                payload["mz_values"] = None
            if "rt_values" not in payload:
                payload["rt_values"] = None
            if "compound_names" not in payload:
                payload["compound_names"] = None
            if "bioactivities" not in payload:
                payload["bioactivities"] = None
            
            # Asegurar campos básicos
            if "publication_year" not in payload:
                payload["publication_year"] = 2024
            if "type" not in payload:
                payload["type"] = "text"

            out.append(payload)

        return out