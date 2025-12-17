# src/application/services/query_processing.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from src.application.ports.llm_port import LLMService
from src.domain.models import FilterSuggestion

class QueryProcessingStrategy(ABC):
    @abstractmethod
    def process_query(self, query: str) -> FilterSuggestion:
        pass

class QueryRewritingStrategy(QueryProcessingStrategy):
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def process_query(self, query: str) -> FilterSuggestion:
        """
        Uses an LLM to rewrite the query and extract metadata filters.
        Returns a FilterSuggestion object with the rewritten query and filters.
        """
        # We can implement a more complex prompt here in the future.
        # For now, we assume the LLM service handles the prompt engineering or we pass a simple one.
        # But per the plan, we should instruct the LLM.
        
        # Example prompt (conceptually):
        # "Analyze the following query. Rewrite it to be more search-friendly for a semantic vector database. 
        # Extract any metadata filters (like year, author, etc.) if present.
        # Query: {query}"
        
        # Since LLMService.generate_structured expects a prompt, let's construct it.
        # Custom prompt for Metabolomics
        prompt = f"""You are an AI assistant for a Metabolomics RAG system.
        Your task is to:
        1. Rewrite the query to be search-friendly for chemical compound identification and bioactivity.
        2. Extract specific metadata filters if present, specifically: 'mz' (number), 'rt' (number in minutes).
        
        Example:
        Query: "Feature m/z 449.1 rt 8.2 in tea"
        Rewritten: "Identify compound m/z 449.1 retention time 8.2 tea bioactivity"
        Filters: {{"mz": 449.1, "rt": 8.2}}
        
        User Query: "{query}"
        
        Return the response in the specified JSON structure."""
        
        return self.llm_service.generate_structured(prompt, FilterSuggestion)

# Placeholder for HyDE or Decomposition if we want to add them later as separate strategies
# class HyDEStrategy(QueryProcessingStrategy): ...
