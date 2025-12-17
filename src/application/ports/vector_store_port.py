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

