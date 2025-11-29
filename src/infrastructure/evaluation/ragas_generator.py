# src/infrastructure/evaluation/ragas_generator.py
import math
from typing import List

import pandas as pd
from langchain_core.documents import Document as LangChainDocument

# Ragas Imports
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import (
    apply_transforms,
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
)
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset import TestsetGenerator


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

    def _generate_from_docs(
        self,
        docs: List[LangChainDocument],
        test_size: int,
    ) -> pd.DataFrame:
        """
        Genera un testset para un subconjunto de documentos.
        Se usa internamente por batch.
        """
        print("\n⚙️ [Ragas] Creando Knowledge Graph base...")

        # 1. Crear Grafo
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

        # 2. Configurar Transformaciones (Usando el LLM Local)
        print(
            f"\n🛠️ [Ragas] Aplicando Transforms sobre {len(docs)} docs "
            f"(esto puede tardar con CPU/LLM Local)..."
        )
        transforms = [
            # Menos headlines para no matar al modelo local
            HeadlinesExtractor(llm=self.llm, max_num=5),
            # Menos tokens por chunk para manejo de memoria
            HeadlineSplitter(max_tokens=50),
            KeyphrasesExtractor(llm=self.llm),
        ]

        apply_transforms(kg, transforms=transforms)

        # 3. Definir Personas (rol de quien hace la pregunta)
        personas = [
            Persona(
                name="Analista Junior",
                role_description=(
                    "Analista principiante que necesita identificar "
                    "metabolitos básicos y entender datos m/z."
                ),
            ),
            Persona(
                name="Químico Experto",
                role_description=(
                    "Experto interesado en isómeros, estructuras complejas "
                    "y rutas biosintéticas."
                ),
            ),
        ]

        # 4. Configurar Synthesizers (de dónde salen las preguntas)
        query_distribution = [
            (
                SingleHopSpecificQuerySynthesizer(
                    llm=self.llm,
                    property_name="headlines",
                ),
                0.5,
            ),
            (
                SingleHopSpecificQuerySynthesizer(
                    llm=self.llm,
                    property_name="keyphrases",
                ),
                0.5,
            ),
        ]

        # 5. Generar testset
        print(f"\n🚀 [Ragas] Generando {test_size} preguntas sintéticas para este batch...")
        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embeddings,
            knowledge_graph=kg,
            persona_list=personas,
        )

        testset = generator.generate(
            testset_size=test_size,
            query_distribution=query_distribution,
        )

        df = testset.to_pandas()


        return df

    def generate_testset(
        self,
        docs: List[LangChainDocument],
        test_size: int = 10,
        max_docs_per_batch: int = 5,
    ) -> pd.DataFrame:
        """
        Genera un testset sintético a partir de una lista de documentos.
        - Si hay pocos documentos, se hace en un solo batch.
        - Si hay muchos, se divide en batches para no romper el LLM local.

        :param docs: Lista de LangChainDocument.
        :param test_size: Total de preguntas a generar.
        :param max_docs_per_batch: Máximo de docs por batch (para controlar tamaño de prompt).
        :return: DataFrame con columnas de Ragas (user_input, reference, etc.).
        """
        if not docs:
            raise ValueError("No se recibieron documentos para generar el testset.")

        if len(docs) <= max_docs_per_batch:
            # Caso simple: todo en un solo grafo
            return self._generate_from_docs(docs, test_size)

        # Caso múltiple: por batches
        print(
            f"\n📚 [Ragas] Generando testset en batches: "
            f"{len(docs)} docs, max_docs_per_batch={max_docs_per_batch}"
        )
        all_dfs: List[pd.DataFrame] = []

        num_batches = math.ceil(len(docs) / max_docs_per_batch)
        # Distribuimos el test_size de forma lo más equitativa posible entre batches
        base_per_batch = max(1, test_size // num_batches)
        remaining = max(0, test_size - base_per_batch * num_batches)

        start_idx = 0
        for batch_idx in range(num_batches):
            end_idx = min(start_idx + max_docs_per_batch, len(docs))
            sub_docs = docs[start_idx:end_idx]
            start_idx = end_idx

            local_size = base_per_batch + (1 if batch_idx < remaining else 0)
            print(
                f"\n📦 [Ragas] Batch {batch_idx + 1}/{num_batches} "
                f"con {len(sub_docs)} docs → {local_size} preguntas"
            )

            df_batch = self._generate_from_docs(sub_docs, local_size)
            all_dfs.append(df_batch)

        final_df = pd.concat(all_dfs, ignore_index=True)
        print(
            f"\n✅ [Ragas] Testset final generado: {len(final_df)} filas "
            f"(solicitadas: {test_size})"
        )
        return final_df
