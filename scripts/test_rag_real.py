import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService
from src.domain.models import ProcessedChunk

def main():
    print("Initializing RAG Components (REAL LLM)...")
    
    # 1. Embedder
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    
    # 2. Vector Store (In-Memory for this test, or use persisted if you prefer)
    db_impl = QdrantImpl(collection_name="test_rag_real", path=None)
    
    # 3. Service (This initializes LocalLLMService connecting to Ollama)
    service = VectorStoreService(embedder=embedder, db_impl=db_impl)
    
    # Index Data (Metabolomics Example)
    print("\n--- Indexing Metabolomics Data ---")
    chunks = [
        ProcessedChunk(content="Fórmula C21H20O12. Compuesto: Myricetina 3-galactósido. Masa exacta: 449.107.", metadata={"mz": 449.107, "type": "public_db"}, chunk_id="pubchem_01"),
        ProcessedChunk(content="Ensayo ID 5678: Myricetina inhibe la agregación plaquetaria significativamente.", metadata={"compound": "myricetin"}, chunk_id="bioassay_5678"),
        ProcessedChunk(content="Feature mz449.1_rt8.1 anotada como Myricetina-derivado en 'Muestra Arándano 004'.", metadata={"mz": 449.1, "rt": 8.1, "type": "internal_exp"}, chunk_id="internal_exp_004"),
    ]
    service.index_chunks(chunks)
    
    # Query
    query = "Tengo una feature con m/z 449.107, RT 8.2 min, detectada en mi muestra de 'Té Verde'. ¿Qué es y qué bioactividad tiene?"
    print(f"\n--- Running Pipeline for: '{query}' ---")
    
    # Retrieve
    retrieved = service.query(query, top_k=3)
    print(f"Retrieved {len(retrieved)} chunks.")
    
    # Generate
    print("Generating answer (calling Ollama)...")
    answer = service.generate(query, retrieved)
    
    print("\n=== FINAL ANSWER ===")
    print(answer)
    print("====================")

if __name__ == "__main__":
    main()
