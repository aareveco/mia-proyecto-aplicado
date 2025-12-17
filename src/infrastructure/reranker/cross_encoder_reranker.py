# src/infrastructure/reranker/cross_encoder_reranker.py
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from src.application.ports.reranker_port import RerankerService
import torch

class CrossEncoderRerankerService(RerankerService):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the CrossEncoder model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Specifically for mps on mac if available, though sentence-transformers handles generic 'auto' well usually
        if torch.backends.mps.is_available():
             device = "mps"
             
        print(f"[Reranker] Loading model {model_name} on {device}...")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
            
        # CrossEncoder expects a list of pairs: [(query, text1), (query, text2), ...]
        pairs = [(query, text) for text in texts]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Ensure it returns a list of floats
        if isinstance(scores, (list, tuple)):
             return [float(s) for s in scores]
        else:
             return scores.tolist()
