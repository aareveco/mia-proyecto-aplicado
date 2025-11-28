from typing import List
from src.domain.models import ProcessedChunk
from src.application.ports.embedder_port import AbstractEmbedder
from src.application.ports.vector_store_port import VectorStoreImpl
from src.infrastructure.loaders.factory import DocumentLoaderFactory

class VectorStoreService:
    """
    Conecta el generador de embeddings con la implementación de DB vectorial.
    """

    def __init__(self, embedder: AbstractEmbedder, db_impl: VectorStoreImpl):
        self._embedder = embedder
        self._db_impl = db_impl

    def index_chunks(self, chunks: List[ProcessedChunk]) -> None:
        vectors = self._embedder.embed_chunks(chunks)

        # Guardamos el embedding en cada chunk (opcional)
        for chunk, vec in zip(chunks, vectors):
            chunk.dense_vector = vec.tolist()

        metadatas = [c.model_dump() for c in chunks]
        self._db_impl.index_data(vectors, metadatas)

    def query(self, query_text: str, top_k: int = 5) -> List[ProcessedChunk]:
        """
        Realiza una búsqueda semántica simple: embede el query y consulta en Qdrant.
        """
        query_chunk = ProcessedChunk(content=query_text)
        query_vector = self._embedder.embed_chunks([query_chunk])[0]

        results_dict = self._db_impl.query_data(query_vector, top_k=top_k)

        # Reconstruimos ProcessedChunk desde payload
        return [ProcessedChunk(**r) for r in results_dict]




def run_indexing_service(
    file_path: str,
    vector_store: VectorStoreService,
) -> None:
    """
    Orquesta el proceso de indexación de un documento en el RAG.
    """
    loader = DocumentLoaderFactory.get_loader(file_path)
    chunks = loader.load_and_chunk(file_path)

    print("[Index] Generando embeddings e indexando en Qdrant...")
    vector_store.index_chunks(chunks)
    print("[Index] Listo.")


def run_retrieval_service(
    query: str,
    vector_store: VectorStoreService,
    top_k: int = 5,
) -> List[ProcessedChunk]:
    """
    Orquesta la consulta semántica.
    """
    print(f"[Retrieval] Ejecutando búsqueda para: {query!r}")
    results = vector_store.query(query, top_k=top_k)
    return results
