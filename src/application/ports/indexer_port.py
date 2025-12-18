from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.domain.models import ProcessedChunk

class IndexerPort(ABC):
    """
    Port defining the capability to index documents/chunks.
    Implementing this interface guarantees the class can accept new chunks.
    """
    
    @abstractmethod
    def index_documents(self, chunks: List[ProcessedChunk], overwrite: bool = False) -> None:
        """
        Indexes the provided chunks.
        """
        pass
