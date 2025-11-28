# src/infrastructure/llm/local_llm_factory.py
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

class LocalResourcesFactory:
    """
    Provee los modelos LLM y Embeddings configurados para Ragas
    usando recursos locales.
    """
    
    @staticmethod
    def get_generator_llm(model_name: str = "qwen2.5:1.5b"):
        # Asegúrate de tener Ollama corriendo: `ollama run llama3.1`
        print(f"[Factory] Conectando a Ollama model={model_name}...")
        ollama_model = ChatOllama(model=model_name, temperature=0.6)
        return LangchainLLMWrapper(ollama_model)

    @staticmethod
    def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Factory] Cargando Embeddings HF model={model_name}...")
        hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return LangchainEmbeddingsWrapper(hf_embeddings)