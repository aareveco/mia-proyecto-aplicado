# src/infrastructure/evaluation/ragas_generator.py
import math
import logging
import asyncio
from typing import List

import pandas as pd
import nest_asyncio  # Critical for fixing concurrency in Ragas/LangChain
from langchain_core.documents import Document as LangChainDocument

# Ragas Imports
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import (
    apply_transforms,
    HeadlinesExtractor,
    HeadlineSplitter,

)
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig

# Apply the event loop patch immediately
nest_asyncio.apply()

# Optional: Configure logging to reduce noise
logging.getLogger("ragas").setLevel(logging.INFO)

class RagasLocalGenerator:
    """
    Generador de testsets sintéticos usando Ragas a partir de documentos de LangChain.

    La idea es generar preguntas (user_input) + respuestas de referencia (reference)
    y sus contextos de referencia (reference_contexts) para luego evaluar el RAG.

    Soporta múltiples documentos usando batching para no reventar al LLM local.
    """
    def __init__(self, generator_llm, generator_embeddings):
        self.llm = generator_llm
        self.embeddings = generator_embeddings
        
        # Define a single, strict configuration for local execution
        # max_workers=1 is essential to prevent concurrency errors locally
        self.safe_run_config = RunConfig(
            timeout=300,        # 5 minutes timeout per doc/op
            max_retries=3,      # Retry if LLM fails
            max_wait=180,       # Wait time between retries
            max_workers=2,      # STRICT SERIAL EXECUTION
            log_tenacity=False
        )

    def _generate_from_docs(
        self,
        docs: List[LangChainDocument],
        test_size: int,
    ) -> pd.DataFrame:
        """
        Genera un testset para un subconjunto de documentos.
        Se usa internamente por batch.
        """
        print(f"\n⚙️ [Ragas] Processing sub-batch of {len(docs)} documents...")

        # 1. Build Knowledge Graph
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                    },
                )
            )

        
        
        transforms = [
            HeadlinesExtractor(llm=self.llm, max_num=3), # Reduced max_num for speed
            HeadlineSplitter(max_tokens=500),

        ]

        print(f"🛠️ [Ragas] Applying transforms (Sequential Mode)...")
        apply_transforms(
            kg,
            transforms=transforms,
            run_config=self.safe_run_config, # Use the safe config
        )

        # 3. Define Personas
        personas = [
            Persona(
                name="Analista Junior",
                role_description="Analista principiante que necesita identificar metabolitos básicos y entender datos m/z.",
            ),
            Persona(
                name="Químico Experto",
                role_description="Experto interesado en isómeros, estructuras complejas y rutas biosintéticas.",
            ),
        ]

        # 4. Define Synthesizer
        query_distribution = [
            (
                SingleHopSpecificQuerySynthesizer(
                    llm=self.llm,
                    property_name="headlines",
                ),
                1,
            ),
       
        ]




        # 5. Generate Testset
        print(f"🚀 [Ragas] Synthesizing {test_size} questions...")
        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embeddings,
            knowledge_graph=kg,
            persona_list=personas,
        )

        testset = generator.generate(
            testset_size=test_size,
            query_distribution=query_distribution,
            run_config=self.safe_run_config, # Use the safe config
        )

        return testset.to_pandas()

    def generate_testset(
        self,
        docs: List[LangChainDocument],
        test_size: int = 5,
        max_docs_per_batch: int = 5,
    ) -> pd.DataFrame:
        """
        Main entry point. Handles batching logic to prevent memory overflows.
        """
        if not docs:
            raise ValueError("No documents provided for testset generation.")

        # If docs are few, process directly
        if len(docs) <= max_docs_per_batch:
            return self._generate_from_docs(docs, test_size)

        # Batching Logic
        print(f"\n📚 [Ragas] Batching strategy active: {len(docs)} docs total.")
        all_dfs: List[pd.DataFrame] = []

        num_batches = math.ceil(len(docs) / max_docs_per_batch)
        base_per_batch = max(1, test_size // num_batches)
        
        # Calculate remainder to distribute among first few batches
        remaining_questions = test_size - (base_per_batch * num_batches)

        start_idx = 0
        for batch_idx in range(num_batches):
            end_idx = min(start_idx + max_docs_per_batch, len(docs))
            sub_docs = docs[start_idx:end_idx]
            start_idx = end_idx

            # Distribute remainder questions
            current_batch_size = base_per_batch + (1 if batch_idx < remaining_questions else 0)
            
            if current_batch_size > 0:
                print(f"\n📦 [Ragas] Batch {batch_idx + 1}/{num_batches}")
                try:
                    df_batch = self._generate_from_docs(sub_docs, current_batch_size)
                    all_dfs.append(df_batch)
                except Exception as e:
                    print(f"❌ [Ragas] Error in batch {batch_idx + 1}: {e}")
                    # Continue to next batch instead of crashing everything
                    continue

        if not all_dfs:
            print("❌ [Ragas] Failed to generate any questions.")
            return pd.DataFrame()

        final_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✅ [Ragas] Final testset generated: {len(final_df)} rows.")
        return final_df