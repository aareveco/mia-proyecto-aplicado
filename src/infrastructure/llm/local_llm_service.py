from langchain_ollama import ChatOllama
from src.application.ports.llm_port import LLMService
from src.domain.models import FilterSuggestion
import json

class LocalLLMService(LLMService):
    def __init__(self, model_name: str = "qwen2.5:1.5b"):
        self.model = ChatOllama(model=model_name, temperature=0.7)

    def generate_text(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        # response is an AIMessage
        return response.content

    def generate_structured(self, prompt: str, response_model: type[FilterSuggestion]) -> FilterSuggestion:
        # Prompt engineering to force JSON
        json_prompt = (
            prompt 
            + "\n\nIMPORTANT: Return ONLY valid JSON matching the schema. "
            + "Do not use markdown code blocks. "
            + "Format: {\"rewritten_query\": \"...\", \"metadata_filters\": {...}}"
        )
        
        response = self.model.invoke(json_prompt)
        text = response.content.strip()
        
        # Cleanup markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()

        try:
            # First try direct JSON parse
            data = json.loads(text)
            return response_model(**data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing JSON from LLM: {e}. Raw text: {text[:50]}...")
            # Fallback: simple heuristic
            # If we failed, let's just return the raw query as rewritten and empty filters
            # Try to extract something if possible? No, safer to fallback safe.
            # Extract query from prompt is hard because prompt is mixed.
            # Let's just return a safe default using a fallback string
            # We don't have the original query easily accessible unless we parse prompt... 
            # actually we can modify signature or just return generic.
            
            # Better Fallback: try to see if it looks like a python dict string
            # But for now, returning a safe object is best to avoid crash.
            # We return an empty instance of the requested model
            print("RETURNING EMPTY FALLBACK MODEL")
            return response_model()

