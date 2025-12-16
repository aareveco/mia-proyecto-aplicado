from typing import List
import random
from src.application.ports.reranker_port import RerankerService

class DummyRerankerService(RerankerService):
    def rerank(self, query: str, texts: List[str]) -> List[float]:
        # Return random scores between 0 and 1 for now
        # In a real scenario, use cross-encoder
        return [random.random() for _ in texts]
