from abc import ABC, abstractmethod
from typing import Dict, List, Any

class SparseEncoder(ABC):
    """
    Port defining the contract for Sparse Vector Encoders (e.g., BM25, SPLADE).
    This decouples the retrieval strategies from specific implementations.
    """
    
    @abstractmethod
    def encode(self, text: str) -> Dict[str, Any]:
        """
        Encodes a text into a sparse vector representation suitable for the vector store.
        
        Args:
            text: The input text to encode.
            
        Returns:
            A dictionary containing the sparse vector representation.
            Expected format for Qdrant: {"indices": [int], "values": [float]}
        """
        pass
