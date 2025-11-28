# src/infrastructure/llm/gemini_factory.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.infrastructure.embeddings.gemini import GeminiEmbedder
from ragas.run_config import RunConfig

class GeminiFactory:
    """
    Factory to provide Gemini resources for both Ragas (Evaluation) 
    and the Application (RAG Pipeline).
    """

    @staticmethod
    def _check_key():
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("❌ GOOGLE_API_KEY environment variable is missing.")

    @staticmethod

    def get_generator_llm(model_name: str = "gemini-2.5-flash"):
        GeminiFactory._check_key()

        print(f"[Factory] Connecting to Gemini LLM: {model_name}")

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_output_tokens=2048,
            transport="rest",            # 👈 ¡ESTE CAMBIO ES LA CLAVE!
            # transport="rest_asyncio",  # (opcional, si quieres mantener async)
        )

        return LangchainLLMWrapper(llm)



    @staticmethod
    def get_embeddings(model_name: str = "models/text-embedding-004"):
        GeminiFactory._check_key()

        embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            transport="rest",  # 👈 evitar gRPC
        )
        return LangchainEmbeddingsWrapper(embeddings)

    @staticmethod
    def get_app_embedder(model_name: str = "models/text-embedding-004"):
        """
        Returns your custom GeminiEmbedder (Port Implementation) 
        for use in the RAG ingestion/retrieval pipeline.
        """
        GeminiFactory._check_key()
        # Uses your existing src/infrastructure/embeddings/gemini.py
        return GeminiEmbedder(model_name=model_name)