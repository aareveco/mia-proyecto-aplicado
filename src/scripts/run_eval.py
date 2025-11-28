# src/scripts/run_eval.py
import sys
import os
import asyncio

# Añadimos el root del proyecto al path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from dotenv import load_dotenv
load_dotenv()

from src.infrastructure.services.service_factory import ServiceFactory
from src.application.services.evaluation_service import EvaluationService

async def main():
    DATASET_PATH = "datasets/golden_dataset.csv"
    OUTPUT_PATH = "datasets/evaluation_results.csv"

    print("🔧 Load RAG Service...")
    rag_service = ServiceFactory.get_vector_service()
    llm, embeddings = ServiceFactory.get_evaluation_components()

    print("⚖️  Initializing Evaluation Service...")
    evaluator = EvaluationService(rag_service, llm, embeddings)

    def on_progress(idx, total, msg):
        if total > 0:
            print(f"[{idx}/{total}] {msg}")
        else:
            print(msg)

    print(f"🧠 Running Benchmark using {DATASET_PATH}...")

    df_result, result_obj = evaluator.run_benchmark(DATASET_PATH, on_progress=on_progress)

    if df_result is None:
        print(result_obj) # Error message
        return

    print("\n📊 Evaluation Results (Retriever):")
    print(result_obj)

    # 4. Guardar resultados
    df_result.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
