
from google import genai
import numpy as np
from typing import List, Optional
import os

# Import the Domain and Port
from src.domain.models import ProcessedChunk
from src.application.ports.embedder_port import AbstractEmbedder


DEFAULT_EMBED_MODEL = "models/text-embedding-004" 
class GeminiEmbedder(AbstractEmbedder):
    """
    Adapter para usar Gemini Embeddings sobre ProcessedChunk.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_EMBED_MODEL):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY no está configurada.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_chunks(self, chunks: List[ProcessedChunk]) -> np.ndarray:
        texts = [c.content for c in chunks]

        # Llamada a la API de embeddings de Gemini
        resp = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
        )

        # La respuesta puede venir con 'embeddings' como lista de objetos
        if hasattr(resp, "embeddings"):
            # Normalmente resp.embeddings es una lista de embeddings
            # Cada embedding tiene atributo 'values'
            vectors = [e.values for e in resp.embeddings]
        elif hasattr(resp, "embedding"):
            # En algunos casos puede venir un solo embedding
            vectors = [resp.embedding.values]
        else:
            raise RuntimeError("Formato de respuesta de Gemini desconocido para embeddings.")

        return np.array(vectors, dtype=np.float32)

