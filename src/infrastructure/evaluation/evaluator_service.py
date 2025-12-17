import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig
from src.application.services.rag_service import VectorStoreService
from src.infrastructure.llm.llm_factory import LLMFactory
import os

class EvaluatorService:
    """
    Service responsible for running Ragas evaluations.
    Decouples evaluation logic from the UI (app.py).
    """
    def __init__(self, rag_service: VectorStoreService):
        self.rag_service = rag_service

    def run_benchmark(self, dataset_path: str, progress_callback=None):
        """
        Runs the Ragas benchmark on the provided dataset.
        
        Args:
            dataset_path: Path to the golden dataset CSV.
            progress_callback: Optional callable to report progress (e.g. st.progress).
                               Should accept (current_index, total, message).
        
        Returns:
            Tuple[pd.DataFrame | None, str |  ragas.Result]: (Result DataFrame, Error Message or Result Object)
        """
        if not os.path.exists(dataset_path):
            return None, f"❌ No se encontró '{dataset_path}'."

        df = pd.read_csv(dataset_path)
        
        # Mapping for ground truth consistency
        if "question" not in df.columns:
             return None, "❌ Falta columna 'question'."
             
        if "ground_truth" not in df.columns:
             if "reference_contexts" in df.columns:
                 df["ground_truth"] = df["reference_contexts"]
             else:
                 return None, "❌ Falta columna 'ground_truth' o 'reference_contexts'."

        questions = df["question"].tolist()
        ground_truths = df["ground_truth"].tolist() 
        
        answers = []
        contexts = []

        total = len(questions)
        
        print(f"[Evaluator] Starting evaluation for {total} questions...")

        for i, q in enumerate(questions):
            if progress_callback:
                progress_callback(i, total, f"Evaluando {i+1}/{total}: {q[:40]}...")
            
            # Execute Query via RAG Service
            # We use the default top_k=3 or configurable? Let's use 3 for benchmark consistency.
            results = self.rag_service.query(q, top_k=3)
            
            retrieved_text = [c.content for c in results]
            
            if results:
                generated_answer = results[0].content 
            else:
                generated_answer = "No information found."

            answers.append(generated_answer)
            contexts.append(retrieved_text)

        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        eval_dataset = Dataset.from_dict(eval_data)

        if progress_callback:
            progress_callback(total, total, "Calculando métricas Ragas (Local)...")
        
        # Factory for Ragas Resources
        try:
            llm = LLMFactory.get_ragas_llm(provider="gemini", model="gemini-2.0-flash-exp")
            embeddings = LLMFactory.get_ragas_embeddings(provider="gemini", model="models/text-embedding-004")
        except Exception as e:
            print(f"[Evaluator] Failed to init Gemini for Ragas, falling back to Local: {e}")
            llm = LLMFactory.get_ragas_llm(provider="local")
            embeddings = LLMFactory.get_ragas_embeddings(provider="local")

        run_config = RunConfig(timeout=120, max_workers=2, max_retries=2)

        result = evaluate(
            eval_dataset,
            metrics=[context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )

        return result.to_pandas(), result
