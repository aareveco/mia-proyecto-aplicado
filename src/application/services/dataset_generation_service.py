# src/application/services/dataset_generation_service.py

import os
import glob
from pathlib import Path
from src.application.ports.loader_port import AbstractLoader
from src.infrastructure.evaluation.ragas_generator import RagasLocalGenerator
#from src.infrastructure.llm.local_llm_factory import LocalResourcesFactory
from src.infrastructure.llm.gemini_factory import GeminiFactory
from langchain_core.documents import Document

class DatasetGenerationService:
    def __init__(self, loader: AbstractLoader):
        self.loader = loader
        
        # Inicializamos recursos locales (Ollama + HF)
        self.llm = GeminiFactory.get_generator_llm(model_name="gemini-2.0-flash")
        self.embedder = GeminiFactory.get_embeddings()
        self.generator = RagasLocalGenerator(self.llm, self.embedder)

    def run(self, input_dir: str, output_dir: str = "evals/datasets", test_size: int = 10):
        """
        Modified to scan a directory for PDFs, load all of them, and generate a 
        consolidated Golden Dataset.
        """
        print(f"[Service] Scanning directory: {input_dir}")
        
        # 1. Find all PDFs
        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {input_dir}")
            return None

        all_langchain_docs: List[Document] = []

        # 2. Iterate and Load chunks from ALL files
        for file_path in pdf_files:
            try:
                print(f"   📄 Loading: {os.path.basename(file_path)}")
                chunks = self.loader.load_and_chunk(file_path)
                
                # Adapter: ProcessedChunk -> LangChain Document
                # We accumulate them into a single list
                docs = [
                    Document(page_content=c.content, metadata=c.metadata) 
                    for c in chunks
                ]
                all_langchain_docs.extend(docs)
                
            except Exception as e:
                print(f"   ❌ Error loading {file_path}: {e}")

        print(f"[Service] Total chunks loaded: {len(all_langchain_docs)}")

        if not all_langchain_docs:
            print("❌ No documents loaded. Exiting.")
            return None

        # 3. Generate Dataset (Ragas will create a KG from the combined documents)
        print("[Service] Starting Ragas Generation...")

        df = self.generator.generate_testset(all_langchain_docs, test_size=test_size)

        # 4. Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use a generic name since it comes from a folder
        output_path = os.path.join(output_dir, "golden_dataset.csv")
        
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset saved to: {output_path}")
        return df