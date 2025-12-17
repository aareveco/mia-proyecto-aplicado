from abc import ABC, abstractmethod
from typing import List
from src.domain.models import ProcessedChunk

class ChunkProcessor(ABC):
    """
    Interface for a step in the ingestion pipeline.
    Transforms a single chunk or a list of chunks.
    """
    
    @abstractmethod
    def process(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Input: List of raw chunks.
        Output: List of processed/refined chunks.
        """
        pass

class IngestionPipeline:
    """
    Orchestrates the sequence of processors.
    """
    def __init__(self, processors: List[ChunkProcessor] = None):
        self.processors = processors or []

    def add_processor(self, processor: ChunkProcessor):
        self.processors.append(processor)

    def run(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Pipes the chunks through all processors in order.
        """
        print(f"[Pipeline] Starting ingestion pipeline with {len(self.processors)} steps.")
        current_chunks = chunks
        
        for i, processor in enumerate(self.processors):
            count_in = len(current_chunks)
            current_chunks = processor.process(current_chunks)
            count_out = len(current_chunks)
            print(f"[Pipeline] Step {i+1} ({processor.__class__.__name__}): {count_in} -> {count_out} chunks.")
            
        return current_chunks
