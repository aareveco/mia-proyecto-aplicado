import numpy as np 
from abc import ABC, abstractmethod
from typing import List,Dict
from src.domain.models import ProcessedChunk


class VectorStoreImpl(ABC):
    @abstractmethod
    @abstractmethod
    def index_data(self, vectors: np.ndarray, metadatas: List[Dict], overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def query_data(self, query_vector: np.ndarray, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        pass



class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve_context(self, query: str, filters: Dict, top_k: int
                         ) -> List[ProcessedChunk]:
        pass

class HybridSearchStrategy(RetrievalStrategy):
    def __init__(self, vector_store: VectorStoreImpl):
        self._vector_store = vector_store

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 5
                         ) -> List[ProcessedChunk]:
        # Lógica para combinar búsqueda vectorial y textual
        print(f"Realizando búsqueda híbrida (componente base).")
        results_dict = self._vector_store.query_data(query, filters)
