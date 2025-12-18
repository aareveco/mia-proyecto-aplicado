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



    def _extract_regex_filters(self, query: str) -> Dict[str, Any]:
        """
        Deterministic extraction of common metabolomics patterns using Regex.
        Decouples critical filter logic from LLM variability.
        """
        import re
        filters = {}
        
        # 1. Extract m/z (mass-to-charge)
        # Patterns: m/z 123.45, mass 123.45, mz=123.45
        mz_pattern = r"(?:m/z|mass|mz)\s*[:=]?\s*(\d+(?:\.\d+)?)"
        mz_match = re.search(mz_pattern, query, re.IGNORECASE)
        if mz_match:
            try:
                filters["mz"] = float(mz_match.group(1))
            except ValueError:
                pass

        # 2. Extract RT (Retention Time)
        # Patterns: rt 10.5, retention time 10.5, rt=10.5, in 10.5 min
        rt_pattern = r"(?:rt|retention(?:\s+time)?)\s*[:=]?\s*(\d+(?:\.\d+)?)"
        rt_match = re.search(rt_pattern, query, re.IGNORECASE)
        if rt_match:
            try:
                filters["rt"] = float(rt_match.group(1))
            except ValueError:
                pass
                
        return filters

    def process_query(self, query: str) -> FilterSuggestion:
        """
        Uses explicit Regex for Filters + LLM for Rewriting.
        """
        # 1. Deterministic Extraction (Regex)
        regex_filters = self._extract_regex_filters(query)
        
        # 2. LLM Semantic Rewriting + Fallback Extraction
        prompt = f"""You are an Expert Metabolomics Assistant.
        Your goal is to prepare a search query for a vector database and extract specific numerical filters.

        INSTRUCTIONS:
        1. **Rewrite** the user's query to focus on chemical entities, mass spectometry features, and biological context. Remove stopwords.
        2. **Extract** 'mz' (mass-to-charge ratio) and 'rt' (retention time) values if they appear in the query.
        
        OUTPUT FORMAT (JSON):
        {{
            "rewritten_query": "chemically relevant keywords...",
            "metadata_filters": {{
                "mz": <float or null>,
                "rt": <float or null>
            }}
        }}

        EXAMPLES:
        User: "Feature m/z 495.1285 rt 5.99 in cocoa powder"
        Output: {{
            "rewritten_query": "cocoa powder feature mass 495.1285 retention 5.99",
            "metadata_filters": {{"mz": 495.1285, "rt": 5.99}}
        }}

        User: "antioxidants in green tea"
        Output: {{
            "rewritten_query": "antioxidants green tea compounds activity",
            "metadata_filters": {{}}
        }}

        Current Query: "{query}"
        """
        
        suggestion = self.llm_service.generate_structured(prompt, FilterSuggestion)
        
        # 3. Merge/Override: Regex takes precedence for strict numbers
        if regex_filters:
            # We assume regex is more accurate for strictly formatted numbers
            suggestion.metadata_filters.update(regex_filters)
            
        return suggestion

