from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
from src.domain.models import ProcessedChunk
import numpy as np
class AbstractEmbedder(ABC):
    @abstractmethod
    def embed_chunks(self, chunks: List[ProcessedChunk]) -> np.ndarray:
        """
        Recibe una lista de chunks y devuelve un np.ndarray de
        shape (n_chunks, dim_embedding).
        """
        pass
