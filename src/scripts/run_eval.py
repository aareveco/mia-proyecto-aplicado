# src/scripts/run_eval.py
import sys
import os
import asyncio

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig 
# Añadimos el root del proyecto al path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import (
    VectorStoreService,
    run_retrieval_service,
)
from src.infrastructure.llm.local_llm_factory import LocalResourcesFactory
from src.infrastructure.llm.gemini_factory import GeminiFactory

QDRANT_PATH = "qdrant_storage"

from dotenv import load_dotenv
load_dotenv()



def get_rag_service() -> VectorStoreService:
    """
    Re-instancia el servicio de RAG (solo retrieval en este caso).
    """
    # embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    # db_impl = QdrantImpl(collection_name="rag_chunks", path=QDRANT_PATH)
       # 1. Get the Gemini Embedder Adapter (Port Implementation)
    embedder = GeminiFactory.get_app_embedder(model_name="models/text-embedding-004")
    
    # 2. Connect to Qdrant
    db_impl = QdrantImpl(collection_name="rag_chunks", path=QDRANT_PATH) 
    return VectorStoreService(embedder, db_impl)


async def main():
    DATASET_PATH = "datasets/golden_dataset.csv"

    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset not found. Run generate_dataset.py first.")
        return

    print("🔧 Load RAG Service...")
    rag_service = get_rag_service()

    print("📂 Load Golden Dataset...")
    df = pd.read_csv(DATASET_PATH)

    # El golden dataset se asume con columnas:
    #   - user_input  -> pregunta
    #   - reference   -> ground truth
    #
    # (Generadas por tu RagasLocalGenerator)
    questions = df["user_input"].tolist()
    ground_truths = df["reference"].tolist()

    answers = []
    contexts = []

    print(f"🧠 Running Retrieval on {len(questions)} questions...")
    for q in questions:
        # Solo retrieval: top_k define cuántos chunks se evaluarán
        results = run_retrieval_service(q, rag_service, top_k=3)

        # Lista de textos de contexto recuperados (en orden de ranking)
        retrieved_texts = [c.content for c in results]

        # Para métricas de retrieval (context_precision/recall),
        # la columna 'answer' NO es crítica, pero Ragas la acepta sin problema.
        generated_answer = results[0].content if results else "No answer found"
        answers.append(generated_answer)
        contexts.append(retrieved_texts)

    # 2. Construir dataset para Ragas
    #    Esquema estándar:
    #       - question      (string)
    #       - answer        (string)  → aquí usamos el top-1 del retriever
    #       - contexts      (list[str])
    #       - ground_truth  (string)
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    eval_dataset = Dataset.from_dict(eval_data)

    # 3. Configurar LLM y embeddings que usará Ragas para juzgar
    #    (LLM-as-a-judge para context_precision / context_recall)
    # llm = LocalResourcesFactory.get_generator_llm("qwen2.5:1.5b")
    # embeddings = LocalResourcesFactory.get_embeddings()

    llm = GeminiFactory.get_generator_llm("gemini-2.0-flash")
    embeddings = GeminiFactory.get_embeddings()

    ragas_run_config = RunConfig(
        timeout=120,      # segundos para cada muestra (ajusta a 180–300 si sigues con Timeouts)
        max_workers=2,    # baja concurrencia para no ahogar la CPU con Qwen local
        max_retries=2,    # opcional: menos reintentos
    )

    print("⚖️  Running Evaluation (Context Precision & Context Recall)...")
    result = evaluate(
        eval_dataset,
        metrics=[context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=ragas_run_config,
        #raise_exceptions=False, 
    )

    print("\n📊 Evaluation Results (Retriever):")
    print(result)

    # 4. Guardar resultados
    result_df = result.to_pandas()
    OUTPUT_PATH = "datasets/evaluation_results.csv"
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
