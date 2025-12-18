import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

# from src.application.services.rag_service import get_vector_service # Removed as it is not in rag_service.py
# In app.py it was defined inside app.py. I should probably move it to rag_service.py or duplicate initialization logic.
# rag_service.py doesn't have get_vector_service. It was in app.py.
# I'll manually initialize in the script for cleaner test logic or import from app.py if possible (might run st logic).
# Best to manually initialize.

from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService, run_indexing_service

# Mock LLM for testing without running Ollama
class MockLLMService:
    def generate_text(self, prompt: str) -> str:
        return "This is a mocked answer based on the context."

    def generate_structured(self, prompt: str, response_model):
        # Return a dummy object matching the response_model (FilterSuggestion)
        from src.domain.models import FilterSuggestion
        return FilterSuggestion(rewritten_query="france capital", metadata_filters={})

def main():
    print("Initializing components...")
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    
    # Use memory for test
    db_impl = QdrantImpl(collection_name="test_rag_full", path=None)
    
    service = VectorStoreService(embedder=embedder, db_impl=db_impl)
    
    # Inject Mock LLM
    service.llm_service = MockLLMService()
    # Re-initialize components that depend on LLM
    from src.application.services.query_processing import QueryRewritingStrategy
    from src.application.services.retrieval_strategies import QueryOptimizerRetriever
    from src.application.services.generation_service import AugmentedGenerator
    
    service.query_processor = QueryRewritingStrategy(service.llm_service)
    service.optimizer_retriever.processor = service.query_processor
    service.generator = AugmentedGenerator(service.llm_service)
    
    # 1. Index some dummy data
    print("\n--- 1. Indexing Data ---")
    # Make a dummy PDF or text file? Or just call index_chunks directly if exposed?
    # service.index_chunks is public.
    from src.domain.models import ProcessedChunk
    
    chunks = [
        ProcessedChunk(content="The capital of France is Paris.", metadata={"topic": "geography"}, chunk_id="doc_1"),
        ProcessedChunk(content="Python is a programming language.", metadata={"topic": "tech"}, chunk_id="doc_2"),
        ProcessedChunk(content="The Eiffel Tower is in Paris.", metadata={"topic": "geography"}, chunk_id="doc_3"),
        ProcessedChunk(content="Machine Learning is a subset of AI.", metadata={"topic": "tech"}, chunk_id="doc_4"),
        ProcessedChunk(content="Croissants are a popular French pastry.", metadata={"topic": "food"}, chunk_id="doc_5"),
    ]
    
    service.index_chunks(chunks)
    
    # 2. Test Retrieval
    query = "Tell me about France capital"
    print(f"\n--- 2. Testing Retrieval for query: '{query}' ---")
    
    # This should trigger: QueryRewrite -> Hybrid -> Rerank -> Repack
    results = service.query(query, top_k=3)
    
    print(f"Retrieved {len(results)} chunks:")
    for i, r in enumerate(results):
        score = r.metadata.get("score", "N/A")
        print(f"[{i}] Score: {score} | Content: {r.content}")
        
    # 3. Test Generation
    print(f"\n--- 3. Testing Generation ---")
    answer = service.generate(query, results)
    print(f"Answer:\n{answer}")

if __name__ == "__main__":
    main()
