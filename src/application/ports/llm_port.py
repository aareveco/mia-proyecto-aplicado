from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.domain.models import FilterSuggestion

class LLMService(ABC):
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generates a text response for the given prompt."""
        pass

    @abstractmethod
    def generate_structured(self, prompt: str, response_model: type[FilterSuggestion]) -> FilterSuggestion:
        """Generates a structured response based on a Pydantic model."""
        pass
