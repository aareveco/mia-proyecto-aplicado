# src/application/services/dataset_generation_service.py
import os
import glob
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import json

from src.application.ports.loader_port import AbstractLoader
from src.infrastructure.llm.llm_factory import LLMFactory
from src.domain.models import ProcessedChunk

class DatasetGenerationService:
    def __init__(self, loader: AbstractLoader):
        self.loader = loader
        
        # Initialize resources using LLMFactory (consistent with bootstrap.py)
        # Using local embedding/LLM by default for generation to be cost-effective,
        # or matching bootstrap if needed. Implicitly uses what Factory provides.
        self.llm = LLMFactory.get_app_llm(provider="local", model="qwen2.5:1.5b")
        self.embedder = LLMFactory.get_app_embeddings(provider="local")
        
        # Note: RagasLocalGenerator is deprecated for this use case 
        # as we are moving to deterministic table-based generation.

    def run(self, input_dir: str, output_dir: str = "evals/datasets", test_size: int = 10):
        """
        Scans directory for PDFs, extracts scientific tables, and generates
        formatted Q&A pairs specifically for metabolomics features (m/z, RT).
        """
        print(f"[Service] Scanning directory: {input_dir}")
        
        # 1. Find all PDFs
        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {input_dir}")
            return None

        all_entries: List[Dict[str, Any]] = []

        # 2. Iterate and Load chunks from EACH file
        for file_path in pdf_files:
            try:
                file_name = os.path.basename(file_path)
                print(f"   📄 Processing: {file_name}")
                chunks = self.loader.load_and_chunk(file_path)
                
                if not chunks:
                    print(f"   ⚠️ No chunks found for {file_name}. Skipping.")
                    continue

                # 3. Generate questions ONLY from Scientific Table chunks
                generated_count = 0
                for chunk in chunks:
                    # Check if it's a scientific table row
                    # Logic depends on how loader tags them (type='table_row_json')
                    if chunk.type != "table_row_json":
                        continue
                        
                    # Check for scientific metadata
                    if not (chunk.mz_values and chunk.rt_values):
                        continue
                        
                    # Extract Data
                    mz = chunk.mz_values[0] # Take first valid value
                    rt = chunk.rt_values[0]
                    
                    # Parse content to get detail (Name, Bioactivity)
                    try:
                        row_data = json.loads(chunk.content)
                    except:
                        row_data = {"raw": chunk.content}
                        
                    # Heuristic to find Name/Bioactivity in row keys
                    # This is best-effort. In real datasets, keys vary.
                    # We dump the whole row as ground truth.
                    
                    # Template Generation
                    question = (
                        f"Tengo una feature con m/z {mz}, RT {rt} min, "
                        f"detectada en mi muestra de '{file_name}'. "
                        f"¿Qué es y qué bioactividad tiene?"
                    )
                    
                    ground_truth = (
                        f"Information from Table Row:\n{json.dumps(row_data, indent=2)}\n"
                        f"Source File: {file_name}"
                    )
                    
                    entry = {
                        "question": question,
                        "ground_truth": ground_truth,
                        "context": chunk.content, # The chunk content itself is the context
                        "source_file": file_name,
                        "mz": mz,
                        "rt": rt
                    }
                    
                    all_entries.append(entry)
                    generated_count += 1
                    
                    # Optional: limit per file if needed, but user wants all relevant?
                    # "distribution should be 2 per pdf" was mentioned in previous context, 
                    # but "test_size" param usually controls total.
                    # Let's collect ALL candidates first, then sample.
                
                print(f"   [Service] Generated {generated_count} candidates from tables in {file_name}")

            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {e}")

        if not all_entries:
            print("❌ No table-based datasets generated. Ensure PDFs have detected scientific tables.")
            return None

        # 4. Consolidate and Sampling
        df = pd.DataFrame(all_entries)
        
        # If we have more than requested, sample
        if len(df) > test_size:
            print(f"[Service] Sampling {test_size} questions from {len(df)} candidates...")
            # Stratified sampling by file could be good, but random is fine for now
            df = df.sample(n=test_size, random_state=42)
        
        # 5. Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, "golden_dataset.csv")
        
        # Keep only required columns for Ragas/Eval
        # Ragas expects: question, ground_truth, (contexts - added during retrieval)
        final_df = df[['question', 'ground_truth', 'context', 'source_file']]
        
        final_df.to_csv(output_path, index=False)
        print(f"✅ Consolidated Dataset saved to: {output_path} with {len(final_df)} rows.")
        return final_df
