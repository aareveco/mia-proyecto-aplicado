# src/domain/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set, Any


class BenchmarkEntry(BaseModel):
    """El contrato del Golden Dataset para la evaluación (C5)."""
    query: str
    reference_answer: str
    relevant_chunk_ids: Set[str]


class ProcessedChunk(BaseModel):
    """
    Representa un chunk ya procesado del documento con metadata estructurada.
    """
    content: str
    source_file: str | None = None
    publication_year: int = 2024
    page: int | None = None
    chunk_id: str | None = None
    
    # Vectores
    dense_vector: List[float] | None = None
    sparse_vector: tuple[list[int], list[float]] | None = None
    
    # Scores de retrieval
    rerank_score: float | None = None
    
    # Metadata estructurada (metabolómica) - con valores por defecto
    mz_values: List[float] | None = Field(default=None)
    rt_values: List[float] | None = Field(default=None)
    compound_names: List[str] | None = Field(default=None)
    bioactivities: List[str] | None = Field(default=None)
    
    # Metadata general
    type: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FilterSuggestion(BaseModel):
    """
    Contrato Pydantic para el output de Query Processing (C6).
    """
    rewritten_query: str
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata extraída del query - con valores por defecto para backward compatibility
    target_mz: float | None = Field(default=None)
    target_rt: float | None = Field(default=None)