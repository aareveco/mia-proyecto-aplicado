from rank_bm25 import BM25Okapi
import numpy as np
from src.domain.models import ProcessedChunk

class BM25API:
    """API REAL de BM25 para sparse vectors (Ported from rag_core.py)"""
    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = []
        
    def fit(self, texts: list[str]):
        """Ajustar BM25 con el corpus"""
        self.tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def get_sparse_vector(self, texts: list[str]) -> np.ndarray:
        """Obtener sparse vectors (scores BM25 contra el corpus)"""
        if self.bm25 is None:
            # Si no está fitted, retornar vectores vacíos
            return np.zeros((len(texts), 100))
        
        vectors = []
        for text in texts:
            tokenized = text.lower().split()
            scores = self.bm25.get_scores(tokenized)
            vectors.append(scores)
        
        return np.array(vectors)
    
    def encode(self, query: str) -> dict:
        """Codificar query a formato sparse para Qdrant"""
        if self.bm25 is None:
            return {"indices": [0], "values": [0.0]}
        
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        
        # Obtener índices no-cero
        non_zero_indices = np.where(scores > 0)[0]
        non_zero_values = scores[non_zero_indices]
        
        if len(non_zero_indices) == 0:
            return {"indices": [0], "values": [0.0]}
        
        return {
            "indices": non_zero_indices.tolist(),
            "values": non_zero_values.tolist()
        }


class BM25Adapter:
    """Adapter REAL para BM25 (Ported from rag_core.py)"""
    def __init__(self, model: BM25API):
        self.model = model
    
    def embed_chunks(self, chunks: list[ProcessedChunk]) -> list[tuple]:
        texts = [c.content for c in chunks]
        print(f"-> [BM25Adapter]: Generando sparse vectors para {len(texts)} chunks")
        
        # Primero ajustamos BM25 con el corpus
        self.model.fit(texts)
        
        # Convertir a formato sparse (índices, valores)
        dense_vectors = self.model.get_sparse_vector(texts)
        sparse_vectors = []
        
        for vec in dense_vectors:
            # Obtener índices de valores no-cero y sus valores
            non_zero_indices = np.where(vec > 0)[0]
            non_zero_values = vec[non_zero_indices]
            
            # Si no hay valores, usar al menos un índice
            if len(non_zero_indices) == 0:
                non_zero_indices = np.array([0])
                non_zero_values = np.array([0.0])
            
            sparse_vectors.append((non_zero_indices.tolist(), non_zero_values.tolist()))
        
        return sparse_vectors
