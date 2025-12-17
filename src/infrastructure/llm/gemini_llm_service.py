# src/infrastructure/llm/gemini_llm_service.py
from typing import Dict, Any
import json
import os
from src.application.ports.llm_port import LLMService
from src.domain.models import FilterSuggestion


class GeminiLLMService(LLMService):
    """
    Implementación del LLM Port usando Google Gemini.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp", api_key: str = None):
        self.model_name = model_name
        
        # Import Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY no está configurada.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=2048,
        )

    def generate_text(self, prompt: str) -> str:
        """Genera texto usando Gemini"""
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"[Gemini Error]: {e}")
            return "Error generating response."

    def generate_structured(
        self, prompt: str, response_model: type[FilterSuggestion]
    ) -> FilterSuggestion:
        """
        Genera respuesta estructurada usando Gemini.
        """
        try:
            # Gemini soporta structured output con with_structured_output
            if hasattr(self.llm, "with_structured_output"):
                structured_llm = self.llm.with_structured_output(response_model)
                result = structured_llm.invoke(prompt)
                
                # Asegurar que target_mz y target_rt estén en metadata_filters
                if result.target_mz is not None and "target_mz" not in result.metadata_filters:
                    result.metadata_filters["target_mz"] = result.target_mz
                if result.target_rt is not None and "target_rt" not in result.metadata_filters:
                    result.metadata_filters["target_rt"] = result.target_rt
                
                return result
            else:
                # Fallback: pedir JSON y parsear
                json_prompt = f"{prompt}\n\nResponde SOLO en formato JSON válido."
                response = self.llm.invoke(json_prompt)
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Limpiar markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                # Parse JSON
                result_dict = json.loads(content)

                # Extraer valores
                target_mz = result_dict.get("target_mz")
                target_rt = result_dict.get("target_rt")
                metadata_filters = result_dict.get("metadata_filters", {})

                # Asegurar que target_mz y target_rt estén en metadata_filters
                if target_mz is not None and "target_mz" not in metadata_filters:
                    metadata_filters["target_mz"] = target_mz
                if target_rt is not None and "target_rt" not in metadata_filters:
                    metadata_filters["target_rt"] = target_rt

                return FilterSuggestion(
                    rewritten_query=result_dict.get("rewritten_query", prompt),
                    metadata_filters=metadata_filters,
                    target_mz=target_mz,
                    target_rt=target_rt,
                )

        except Exception as e:
            print(f"[Gemini Error]: {e}")
            return FilterSuggestion(rewritten_query=prompt, metadata_filters={})

    def extract_chunk_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extrae metadata estructurada del chunk usando Gemini.
        """
        if len(text.strip()) < 50:
            return {
                "mz_values": None,
                "rt_values": None,
                "compound_names": None,
                "bioactivities": None,
            }

        try:
            prompt = f"""Extrae del texto científico:
- mz_values: lista de valores m/z (ej: [449.107])
- rt_values: lista de valores RT en minutos (ej: [8.2])
- compound_names: lista de nombres de compuestos (ej: ["Myricetina"])
- bioactivities: lista de bioactividades (ej: ["antioxidant"])

Responde SOLO en JSON:
{{"mz_values": [449.107], "rt_values": [8.2], "compound_names": ["X"], "bioactivities": ["Y"]}}
Si no hay datos, usa null.

Texto:
{text[:1500]}"""

            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Limpiar markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return {
                "mz_values": result.get("mz_values"),
                "rt_values": result.get("rt_values"),
                "compound_names": result.get("compound_names"),
                "bioactivities": result.get("bioactivities"),
            }

        except Exception as e:
            print(f"[Gemini Metadata Extraction Error]: {e}")
            return {
                "mz_values": None,
                "rt_values": None,
                "compound_names": None,
                "bioactivities": None,
            }