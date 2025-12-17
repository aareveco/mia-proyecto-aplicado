# src/infrastructure/loaders/pdf_loader.py
from typing import List, Dict, Any
from src.domain.models import ProcessedChunk
from src.application.ports.loader_port import AbstractLoader
from src.application.ports.llm_port import LLMService

from langchain_community.document_loaders import PyPDFLoader
from docling.document_converter import DocumentConverter
from pathlib import Path
import pandas as pd
import os


class PDFLoader(AbstractLoader):
    """Loader que extrae texto, tablas y metadata estructurada con LLM."""

    def __init__(self, llm_service: LLMService | None = None):
        """
        Args:
            llm_service: Servicio LLM para extraer metadata estructurada de chunks.
                        Si es None, no se extrae metadata estructurada.
        """
        self.llm_service = llm_service

    def load_and_chunk(self, path: str) -> List[ProcessedChunk]:
        print(f"[Loader] Cargando PDF con PyPDFLoader + Docling: {path}")

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 1. Cargar texto con LangChain
        loader = PyPDFLoader(path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        docs = loader.load_and_split(text_splitter=splitter)

        chunks: List[ProcessedChunk] = []
        
        # Metadata del documento
        file_metadata = self._extract_document_metadata(path)

        # 2. Procesar chunks de texto
        total_chunks = len(docs)
        print(f"[Loader] Procesando {total_chunks} chunks de texto...")
        
        if self.llm_service:
            print(f"[Loader] 🤖 Extracción LLM de metadata ACTIVADA (esto puede tardar...)")
        else:
            print(f"[Loader] ℹ️ Extracción LLM de metadata DESACTIVADA")
        
        for i, doc in enumerate(docs):
            source_metadata = doc.metadata or {}
            page_number = source_metadata.get("page")
            
            chunk_text = doc.page_content.strip()
            if not chunk_text:
                continue

            # Crear chunk base
            chunk = ProcessedChunk(
                content=chunk_text,
                source_file=path,
                publication_year=file_metadata["publication_year"],
                page=page_number,
                chunk_id=f"{os.path.basename(path)}-chunk-{i}",
                type="text",
                metadata=source_metadata,
            )
            
            # Extraer metadata estructurada con LLM
            if self.llm_service:
                # Mostrar progreso cada 20 chunks
                if (i + 1) % 20 == 0 or (i + 1) == total_chunks:
                    print(f"   → Extrayendo metadata: {i + 1}/{total_chunks} chunks...")
                
                structured_meta = self.llm_service.extract_chunk_metadata(chunk_text)
                chunk.mz_values = structured_meta.get("mz_values")
                chunk.rt_values = structured_meta.get("rt_values")
                chunk.compound_names = structured_meta.get("compound_names")
                chunk.bioactivities = structured_meta.get("bioactivities")
            
            chunks.append(chunk)

        print(f"[Loader] ✅ Generados {len(chunks)} text chunks")

        # 3. Procesar tablas con Docling
        file_name = Path(path).name
        try:
            doc_result = DocumentConverter().convert(path)
            table_count = 0

            for i, table in enumerate(doc_result.document.tables):
                df = table.export_to_dataframe()
                table_meta = self._detect_scientific_metadata(df)

                if table_meta.get("has_scientific_data"):
                    # Chunking de tablas: grupos de 10 filas
                    chunk_size = 10
                    total_rows = len(df)

                    for start_row in range(0, total_rows, chunk_size):
                        end_row = min(start_row + chunk_size, total_rows)
                        df_chunk = df.iloc[start_row:end_row]
                        content_md = df_chunk.to_markdown(
                            index=False, tablefmt="github"
                        )

                        chunk_meta = table_meta.copy()
                        chunk_meta.update(
                            {
                                "table_index": i,
                                "row_start": start_row,
                                "row_end": end_row,
                                "total_table_rows": total_rows,
                                "file_name": file_name,
                                "chunk_type": "scientific_table",
                            }
                        )

                        chunks.append(
                            ProcessedChunk(
                                content=content_md,
                                source_file=path,
                                publication_year=file_metadata["publication_year"],
                                page=None,
                                chunk_id=f"{file_name}-table-{i}-rows-{start_row}-{end_row}",
                                type="table",
                                metadata=chunk_meta,
                            )
                        )
                        table_count += 1

            print(f"[Loader] Generados {table_count} table chunks")

        except Exception as e:
            print(f"[Loader] Error procesando tablas con Docling: {e}")

        return chunks

    def _extract_document_metadata(self, path: str) -> Dict[str, Any]:
        """Extrae metadata del documento (año, etc.)"""
        # TODO: Usar PyPDF2 o LLM para extraer año real
        return {"publication_year": 2024}

    @staticmethod
    def _detect_scientific_metadata(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta columnas de RT, m/z, adduct en tablas.
        """
        headers = [str(h) for h in df.columns]
        rt_col, mz_col, adduct_col = None, None, None

        for h in headers:
            low = h.lower().strip()
            if not rt_col and ("rt" == low or "retention time" in low):
                rt_col = h
            if not mz_col and ("m/z" in low or "mz" == low):
                mz_col = h
            if not adduct_col and ("[m+" in low or "[m-" in low or "adduct" in low):
                adduct_col = h

        meta = {
            "has_scientific_data": bool(rt_col and mz_col),
            "columns": headers,
            "rt_column": rt_col,
            "mz_column": mz_col,
            "adduct_column": adduct_col,
        }

        return meta