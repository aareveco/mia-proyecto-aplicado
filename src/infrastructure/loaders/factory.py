# src/infrastructure/loaders/factory.py
from src.application.ports.loader_port import AbstractLoader
from src.infrastructure.loaders.pdf_loader import PDFLoader 

class DocumentLoaderFactory:
    @staticmethod
    def get_loader(path: str) -> AbstractLoader:
        p = path.lower()
        if p.endswith(".pdf"):
            return PDFLoader()
        # Add other loaders (Markdown, etc.) here
        raise ValueError(f"Format not supported: {path}")

