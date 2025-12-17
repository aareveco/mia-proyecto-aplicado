from typing import List, Dict, Any
from src.domain.models import ProcessedChunk
from src.application.ports.loader_port import AbstractLoader

from langchain_community.document_loaders import PyPDFLoader
from docling.document_converter import DocumentConverter
from pathlib import Path
import pandas as pd
import os

class PDFLoader(AbstractLoader):
    """Loader concreto que usa LangChain PyPDFLoader por debajo."""

    def load_and_chunk(self, path: str) -> List[ProcessedChunk]:
        print(f"[Loader] Cargando y troceando PDF con PyPDFLoader: {path}")

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 1. Cargar el PDF con LangChain
        loader = PyPDFLoader(path)

        # load_and_split divide el texto (por defecto usa RecursiveCharacterTextSplitter)
        # Usamos un chunk size más pequeño para el MVP local
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        docs = loader.load_and_split(text_splitter=splitter)

        chunks: List[ProcessedChunk] = []
        
        for i, doc in enumerate(docs):
            # Extraer metadatos comunes que provee PyPDFLoader
            # Normalmente doc.metadata tiene keys como {'source': '...', 'page': 0}
            source_metadata = doc.metadata or {}
            
            # Obtener el número de página (LangChain suele usar 0-indexed)
            page_number = source_metadata.get('page')

            chunks.append(
                ProcessedChunk(
                    content=doc.page_content,
                    source_file=path,
                    page=page_number,
                    chunk_id=f"{os.path.basename(path)}-chunk-{i}",
                    type="text", 
                    metadata=source_metadata, 
                    # dense_vector y sparse_vector se dejan en None para pasos posteriores
                )
            )
        

        print(f"[Loader] Generados {len(chunks)} text chunks desde {path}")

        file_name = Path(path).name
        # Note: Docling might be slow on large files.
        doc_result = DocumentConverter().convert(path)

        table_chunk_count = 0
        for i, table in enumerate(doc_result.document.tables):
            df = table.export_to_dataframe()
            
            # Generate Metadata
            table_meta = self._detect_scientific_metadata(df)
            
            if table_meta.get("has_scientific_data"):
                # Chunking logic for scientific tables: groups of 10 rows
                chunk_size = 10
                total_rows = len(df)
                
                # Iterate over the dataframe in chunks
                for start_row in range(0, total_rows, chunk_size):
                    end_row = min(start_row + chunk_size, total_rows)
                    df_chunk = df.iloc[start_row:end_row]
                    
                    # Convert chunk to Markdown
                    content_md = df_chunk.to_markdown(index=False, tablefmt="github")
                    
                    # Create metadata for this chunk
                    chunk_meta = table_meta.copy()
                    chunk_meta.update({
                        "table_index": i,
                        "row_start": start_row,
                        "row_end": end_row,
                        "total_table_rows": total_rows,
                        "file_name": file_name,
                        "chunk_type": "scientific_table"
                    })
                    
                    # Create ProcessedChunk
                    chunks.append(
                        ProcessedChunk(
                            content=content_md,
                            source_file=path,
                            page=None, # Docling tables might not map easily to a single page without extra logic
                            chunk_id=f"{file_name}-table-{i}-rows-{start_row}-{end_row}",
                            type="table",
                            metadata=chunk_meta
                        )
                    )
                    table_chunk_count += 1
            else:
                 # Optional: Handle non-scientific tables if needed, for now skip or add as whole
                 pass

        print(f"[Loader] Generados {table_chunk_count} table chunks desde {path}")

        return chunks

    @staticmethod
    def _detect_scientific_metadata(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Simplified heuristic to detect RT, m/z, and adduct columns 
        and extract their ranges.
        """
        headers = [str(h) for h in df.columns]
        rt_col, mz_col, adduct_col = None, None, None

        # Column Detection
        for h in headers:
            low = h.lower().strip()
            if not rt_col and ("rt" == low or "retention time" in low):
                rt_col = h
            if not mz_col and ("m/z" in low or "mz" == low):
                mz_col = h
            if not adduct_col and ("[m+" in low or "[m-" in low or "adduct" in low):
                adduct_col = h

        # Fallback: Adduct can act as m/z if m/z is missing but we want to be generous, 
        # though strictly speaking they are different. The prompt implied relying on these.
        if not mz_col and adduct_col:
            pass # Keep it strict as per original logic intent, or enable if needed.
                 # Original code had: if not mz_col and adduct_col: mz_col = adduct_col

        meta = {
            "has_scientific_data": bool(rt_col and mz_col),
            "columns": headers,
            "rt_column": rt_col,
            "mz_column": mz_col,
            "adduct_column": adduct_col,
        }
    
        return meta