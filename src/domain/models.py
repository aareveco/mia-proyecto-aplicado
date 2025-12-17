from pydantic import BaseModel,Field

from typing import List, Dict, Optional, Set, Any

class BenchmarkEntry(BaseModel) :
    '''El contrato del Golden Dataset para la evaluación (C5). '''
    query: str
    reference_answer: str
    relevant_chunk_ids: Set[str]


class ProcessedChunk(BaseModel):
    """
    Representa un chunk ya procesado del documento.
    """
    content: str
    source_file: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    # Optional metadata fields based on your example
    type: str | None = None # e.g., 'paragraph', 'table', 'section'
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dense_vector: List[float] | None = None
    sparse_vector: tuple[List[int], List[float]] | None = None  # (indices, values)
    rerank_score: float | None = None

    # Metabolomics Metadata
    mz_values: List[float] | None = None
    rt_values: List[float] | None = None
    compound_names: List[str] | None = None
    bioactivities: List[str] | None = None


class FilterSuggestion(BaseModel):
    """
    Contrato pydantic para el output de query processing
    """
    rewritten_query: str
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)