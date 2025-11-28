from src.domain.models import ProcessedChunk
from src.application.ports.embedder_port import AbstractEmbedder

import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
class HuggingFaceEmbedder(AbstractEmbedder):
    """
    Embedder concreto que usa modelos Open Source de Hugging Face
    vía la librería 'sentence-transformers'.
    
    No requiere API Key. Se ejecuta localmente.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Args:
            model_name: ID del modelo en Hugging Face. 
                        - 'all-MiniLM-L6-v2': Rápido y ligero (Inglés/General).
                        - 'paraphrase-multilingual-MiniLM-L12-v2': Recomendado si los PDFs están en Español.
            device: 'cpu', 'cuda', 'mps' (para Mac), o None (detecta automático).
        """
        print(f"[Embedder] Cargando modelo local: {model_name} ...")
        self.model_name = model_name
        
        # Carga el modelo (lo descarga si no existe en caché)
        self.model = SentenceTransformer(model_name, device=device)
        print("[Embedder] Modelo cargado exitosamente.")

    def embed_chunks(self, chunks: List[ProcessedChunk]) -> np.ndarray:
        # Extraer solo el texto de los chunks
        texts = [c.content for c in chunks]

   
        embeddings = self.model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )

        return embeddings.astype(np.float32)

