import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from src.application.ports.llm_port import LLMService
from src.domain.models import FilterSuggestion

class GeminiLLMService(LLMService):
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        if "GOOGLE_API_KEY" not in os.environ:
            # Check if streamlit secrets are available as fallback? 
            # Better to enforce env var as per factory pattern
            print("⚠️ WARNING: GOOGLE_API_KEY not found in environment.")
            
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_output_tokens=2048,
        )

    def generate_text(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    def generate_structured(self, prompt: str, response_model: type[FilterSuggestion]) -> FilterSuggestion:
        """
        Uses Gemini's native structured output capabilities (or LangChain's wrapper)
        to reliably return the Pydantic model.
        """
        # Gemini/LangChain supports .with_structured_output(PydanticModel)
        structured_llm = self.model.with_structured_output(response_model)
        
        try:
            return structured_llm.invoke(prompt)
        except Exception as e:
            print(f"Error in Gemini structured generation: {e}")
            # Fallback to manual parsing if native fails?
            # Ideally native works best.
            return response_model(rewritten_query=prompt, metadata_filters={})
