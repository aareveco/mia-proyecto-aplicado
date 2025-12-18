import re
from typing import List
from src.domain.models import ProcessedChunk
from src.application.services.ingestion_pipeline import ChunkProcessor

class CleanerProcessor(ChunkProcessor):
    """
    Cleans text content: removes excessive whitespace, newlines, etc.
    """
    def process(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        cleaned_chunks = []
        for chunk in chunks:
            # Skip cleaning for structured JSON chunks
            if chunk.type == "table_row_json":
                cleaned_chunks.append(chunk)
                continue

            text = chunk.content
            # Remove multiple newlines
            text = re.sub(r'\n+', '\n', text)
            # Remove multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            chunk.content = text.strip()
            # If chunk is empty after cleaning, potentially skip? 
            # For now keep everything but cleaner.
            if chunk.content:
                cleaned_chunks.append(chunk)
                
        return cleaned_chunks

class MetadataExtractorProcessor(ChunkProcessor):
    """
    Extracts metadata like M/Z (Mass-to-Charge), RT (Retention Time) from text using Regex.
    """
    def process(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        # Regex patterns for Metabolomics
        mz_pattern = re.compile(r'm/z\s*[:=]?\s*(\d+\.?\d*)', re.IGNORECASE)
        rt_pattern = re.compile(r'RT\s*[:=]?\s*(\d+\.?\d*)', re.IGNORECASE)
        
        for chunk in chunks:
            # Skip extraction for structured chunks (already handled in loader)
            if chunk.type == "table_row_json":
                continue

            text = chunk.content
            
            # Init metadata if None
            if chunk.metadata is None:
                chunk.metadata = {}
                
            # Extract MZ
            mz_match = mz_pattern.search(text)
            if mz_match:
                try:
                    chunk.metadata['mz'] = float(mz_match.group(1))
                except ValueError:
                    pass

            # Extract RT
            rt_match = rt_pattern.search(text)
            if rt_match:
                try:
                    chunk.metadata['rt'] = float(rt_match.group(1))
                except ValueError:
                    pass
                    
        return chunks
