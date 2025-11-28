from abc import ABC, abstractmethod
from typing import List
from src.domain.models import ProcessedChunk

class AbstractLoader(ABC):
    @abstractmethod
    def load_and_chunk(self, path: str) -> List[ProcessedChunk]:
        pass
