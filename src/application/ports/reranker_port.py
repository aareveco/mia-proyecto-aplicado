from abc import ABC, abstractmethod
from typing import List

class RerankerService(ABC):
    @abstractmethod
    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """
        Reranks a list of texts based on their relevance to the query.
        Returns a list of scores corresponding to the input texts.
        """
        pass
