import numpy as np 
from abc import ABC, abstractmethod
from typing import List,Dict
from src.domain.models import ProcessedChunk


class VectorStoreImpl(ABC):
    @abstractmethod
    def index_data(self, vectors: np.ndarray, metadatas: List[Dict]) -> None:
        pass

    @abstractmethod
    def query_data(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        pass