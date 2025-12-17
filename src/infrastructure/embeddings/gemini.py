# src/infrastructure/embeddings/gemini.py
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
    Implementa batching para respetar el límite de 100 chunks por request.
    """

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = DEFAULT_EMBED_MODEL
    ):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY no está configurada.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.batch_size = 100  # Límite de Gemini API

    def embed_chunks(self, chunks: List[ProcessedChunk]) -> np.ndarray:
        """
        Genera embeddings en batches de 100 para respetar límite de API.
        """
        texts = [c.content for c in chunks]
        
        all_vectors = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        print(f"[Gemini Embeddings] Procesando {len(texts)} chunks en {total_batches} batches...")
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            if total_batches > 1:
                print(f"  Batch {batch_num}/{total_batches}: {len(batch_texts)} chunks")
            
            # Llamada a la API de embeddings de Gemini
            resp = self.client.models.embed_content(
                model=self.model_name,
                contents=batch_texts,
            )
            
            # Extraer vectores
            if hasattr(resp, "embeddings"):
                batch_vectors = [e.values for e in resp.embeddings]
            elif hasattr(resp, "embedding"):
                batch_vectors = [resp.embedding.values]
            else:
                raise RuntimeError(
                    "Formato de respuesta de Gemini desconocido para embeddings."
                )
            
            all_vectors.extend(batch_vectors)
        
        print(f"[Gemini Embeddings] Completado: {len(all_vectors)} embeddings generados")
        return np.array(all_vectors, dtype=np.float32)