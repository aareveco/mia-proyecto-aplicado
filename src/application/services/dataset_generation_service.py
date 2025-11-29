# src/application/services/dataset_generation_service.py
import os
import glob
from typing import List
import pandas as pd
from pathlib import Path
from src.application.ports.loader_port import AbstractLoader
from src.infrastructure.evaluation.ragas_generator import RagasLocalGenerator
from src.infrastructure.llm.local_llm_factory import LocalResourcesFactory
from langchain_core.documents import Document

class DatasetGenerationService:
    def __init__(self, loader: AbstractLoader):
        self.loader = loader
        
        # Inicializamos recursos locales (Ollama + HF)
        self.llm = LocalResourcesFactory.get_generator_llm(model_name="qwen2.5:1.5b")
        self.embedder = LocalResourcesFactory.get_embeddings()
        self.generator = RagasLocalGenerator(self.llm, self.embedder)

    def run(self, input_dir: str, output_dir: str = "evals/datasets", test_size: int = 10):
        """
        Modified to scan a directory for PDFs, load each one individually,
        and generate a test set for each PDF separately to ensure representation.
        """
        print(f"[Service] Scanning directory: {input_dir}")
        
        # 1. Find all PDFs
        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {input_dir}")
            return None

        all_dfs: List[pd.DataFrame] = []

        # 2. Iterate and Load chunks from EACH file separately
        for file_path in pdf_files:
            try:
                print(f"   📄 Processing: {os.path.basename(file_path)}")
                chunks = self.loader.load_and_chunk(file_path)
                
                if not chunks:
                    print(f"   ⚠️ No chunks found for {file_path}. Skipping.")
                    continue

                # LIMIT CHUNKS FOR MVP (Speed up generation)
                max_chunks = 5
                if len(chunks) > max_chunks:
                    print(f"   [MVP] Limiting to first {max_chunks} chunks (of {len(chunks)})")
                    chunks = chunks[:max_chunks]

                # Adapter: ProcessedChunk -> LangChain Document
                docs = [
                    Document(page_content=c.content, metadata=c.metadata) 
                    for c in chunks
                ]

                # 3. Generate Dataset for THIS SPECIFIC PDF
                # We request 'test_size' questions PER PDF (as per user request: "distribution should be 2 per pdf")
                # The 'test_size' parameter passed to run() acts as the per-pdf limit.
                print(f"   [Service] Generating {test_size} questions for {os.path.basename(file_path)}...")
                df_doc = self.generator.generate_testset(docs, test_size=test_size)

                if df_doc is not None and not df_doc.empty:
                    all_dfs.append(df_doc)
                
            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {e}")

        if not all_dfs:
            print("❌ No datasets generated. Exiting.")
            return None

        # 4. Consolidate Results
        print("[Service] Consolidating all datasets...")
        final_df = pd.concat(all_dfs, ignore_index=True)

        # 5. Rename columns to standard 'question', 'ground_truth'
        # Ragas typically produces: 'user_input', 'reference'
        rename_map = {
            'user_input': 'question',
            'reference': 'ground_truth'
        }
        # Only rename if they exist
        final_df.rename(columns=rename_map, inplace=True)
        
        # Ensure we have the required columns for context_recall/precision
        # context_precision requires: question, ground_truth, contexts (retrieved) - contexts comes later
        # context_recall requires: ground_truth, contexts (retrieved)
        # So we just need question and ground_truth in the dataset.

        # 6. Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, "golden_dataset.csv")
        
        final_df.to_csv(output_path, index=False)
        print(f"✅ Consolidated Dataset saved to: {output_path} with {len(final_df)} total rows.")
        return final_df
