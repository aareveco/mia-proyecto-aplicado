# scripts/run_eval.py
import sys
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig

# Añadimos el root del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infrastructure.bootstrap import create_rag_service
from src.infrastructure.llm.gemini_factory import GeminiFactory

QDRANT_PATH = "qdrant_storage"
DATASET_PATH = "datasets/golden_dataset.csv"


def main():
    # 1. Verificar dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("Run 'python scripts/generate_test_dataset.py' first.")
        return

    print("🔧 Loading RAG Service...")
    rag_service = create_rag_service(
        qdrant_path=QDRANT_PATH,
        use_gemini_embeddings=True,
        gemini_model="gemini-2.0-flash-exp",
    )

    print("📂 Loading Golden Dataset...")
    df = pd.read_csv(DATASET_PATH)

    questions = df["question"].tolist()
    ground_truths = df["reference_contexts"].tolist()

    answers = []
    contexts = []

    print(f"🧠 Running Retrieval on {len(questions)} questions...")
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q[:60]}...")
        results = rag_service.query(q, top_k=3)

        retrieved_texts = [c.content for c in results]
        generated_answer = results[0].content if results else "No answer found"

        answers.append(generated_answer)
        contexts.append(retrieved_texts)

    # 2. Preparar dataset para Ragas
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    eval_dataset = Dataset.from_dict(eval_data)

    # 3. Configurar Gemini para evaluación
    print("⚖️  Configuring Gemini for Ragas evaluation...")
    llm = GeminiFactory.get_generator_llm("gemini-2.0-flash-exp")
    embeddings = GeminiFactory.get_embeddings()

    ragas_run_config = RunConfig(
        timeout=120,
        max_workers=4,  # Gemini soporta más concurrencia
        max_retries=2,
    )

    print("📊 Running Evaluation (Context Precision & Context Recall)...")
    result = evaluate(
        eval_dataset,
        metrics=[context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=ragas_run_config,
        raise_exceptions=False,
    )

    print("\n📊 Evaluation Results:")
    print(result)

    # 4. Guardar resultados
    result_df = result.to_pandas()
    OUTPUT_PATH = "datasets/evaluation_results.csv"
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Results saved to {OUTPUT_PATH}")

    # 5. Mostrar métricas promedio
    if "context_precision" in result_df.columns:
        print(f"\n✅ Context Precision: {result_df['context_precision'].mean():.4f}")
    if "context_recall" in result_df.columns:
        print(f"✅ Context Recall: {result_df['context_recall'].mean():.4f}")


if __name__ == "__main__":
    main()