from typing import List
from src.domain.models import ProcessedChunk
from src.application.ports.loader_port import AbstractLoader

from langchain_community.document_loaders import PyPDFLoader
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

        print(f"[Loader] Generados {len(chunks)} chunks desde {path}")
        return chunks