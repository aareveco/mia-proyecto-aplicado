from typing import Literal, Optional
import os

# Ragas / LangChain Dependencies
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Application Dependencies
from src.application.ports.llm_port import LLMService
from src.infrastructure.llm.local_llm_service import LocalLLMService
from src.infrastructure.llm.gemini_llm_service import GeminiLLMService
from src.application.ports.embedder_port import AbstractEmbedder
from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.embeddings.gemini import GeminiEmbedder

class LLMFactory:
    """
    Unified Factory to provide LLM resources for both the Application (RAG Pipeline)
    and Evaluation (Ragas). Supports switching between 'local' and 'gemini'.
    """
    
    @staticmethod
    def get_app_llm(provider: Literal["local", "gemini"] = "local", **kwargs) -> LLMService:
        """
        Returns the concrete implementation of LLMService for the application.
        """
        if provider == "gemini":
            model = kwargs.get("model", "gemini-2.0-flash-exp")
            print(f"[LLMFactory] Initializing Gemini LLM Service ({model})...")
            return GeminiLLMService(model_name=model)
        else:
            model = kwargs.get("model", "qwen2.5:1.5b")
            print(f"[LLMFactory] Initializing Local LLM Service ({model})...")
            return LocalLLMService(model_name=model)

    @staticmethod
    def get_app_embeddings(provider: Literal["local", "gemini"] = "local", **kwargs) -> AbstractEmbedder:
        """
        Returns the concrete implementation of AbstractEmbedder for the application.
        """
        if provider == "gemini":
            model = kwargs.get("model", "models/text-embedding-004")
            print(f"[LLMFactory] Initializing Gemini Embeddings ({model})...")
            return GeminiEmbedder(model_name=model)
        else:
            # Default to local HuggingFace
            model = kwargs.get("model", "all-MiniLM-L6-v2")
            print(f"[LLMFactory] Initializing Local HuggingFace Embeddings ({model})...")
            return HuggingFaceEmbedder(model_name=model)

    @staticmethod
    def get_ragas_llm(provider: Literal["local", "gemini"] = "local", **kwargs):
        """
        Returns the LLM wrapped for Ragas evaluation.
        """
        if provider == "gemini":
            if "GOOGLE_API_KEY" not in os.environ:
                 raise ValueError("❌ GOOGLE_API_KEY environment variable is missing.")
            
            model = kwargs.get("model", "gemini-2.0-flash-exp")
            print(f"[LLMFactory] Connecting to Gemini for Ragas: {model}")
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0,
                max_output_tokens=2048
            )
            return LangchainLLMWrapper(llm)
        else:
            model = kwargs.get("model", "qwen2.5:1.5b")
            print(f"[LLMFactory] Connecting to Local Ollama for Ragas: {model}")
            ollama_model = ChatOllama(model=model, temperature=0)
            return LangchainLLMWrapper(ollama_model)

    @staticmethod
    def get_ragas_embeddings(provider: Literal["local", "gemini"] = "local", **kwargs):
        """
        Returns Embeddings wrapped for Ragas evaluation.
        """
        if provider == "gemini":
            model = kwargs.get("model", "models/text-embedding-004")
            print(f"[LLMFactory] Using Gemini Embeddings for Ragas: {model}")
            embeddings = GoogleGenerativeAIEmbeddings(model=model)
            return LangchainEmbeddingsWrapper(embeddings)
        else:
            model = kwargs.get("model", "all-MiniLM-L6-v2")
            print(f"[LLMFactory] Using HF Embeddings for Ragas: {model}")
            hf_embeddings = HuggingFaceEmbeddings(model_name=model)
            return LangchainEmbeddingsWrapper(hf_embeddings)
