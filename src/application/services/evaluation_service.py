import os
import pandas as pd
from typing import List, Tuple, Optional, Callable, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig

from src.application.services.rag_service import VectorStoreService, run_retrieval_service

class EvaluationService:
    """
    Service responsible for running RAG benchmarks using Ragas.
    """

    def __init__(self, rag_service: VectorStoreService, llm: Any, embeddings: Any):
        self.rag_service = rag_service
        self.llm = llm
        self.embeddings = embeddings

    def run_benchmark(
        self,
        dataset_path: str,
        top_k: int = 3,
        on_progress: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[Optional[pd.DataFrame], Any]:
        """
        Runs the benchmark.

        Args:
            dataset_path: Path to the golden dataset CSV.
            top_k: Number of chunks to retrieve per question.
            on_progress: Optional callback function(index, total, message) for progress updates.

        Returns:
            Tuple containing the results DataFrame and the Ragas Result object.
            Returns (None, error_message_str) if validation fails.
        """
        # 1. Verify Dataset
        if not os.path.exists(dataset_path):
            return None, f"❌ Dataset not found at '{dataset_path}'."

        # 2. Load Dataset
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            return None, f"❌ Failed to load CSV: {e}"

        # Validate columns
        if "user_input" not in df.columns or "reference" not in df.columns:
            return None, "❌ CSV must have 'user_input' and 'reference' columns."

        questions = df["user_input"].tolist()
        ground_truths = df["reference"].tolist()

        answers = []
        contexts = []

        total = len(questions)

        # 3. Run Retrieval
        for i, q in enumerate(questions):
            if on_progress:
                on_progress(i, total, f"Evaluating query {i+1}/{total}: {q[:50]}...")

            # Call RAG service
            results = run_retrieval_service(q, self.rag_service, top_k=top_k)

            # Prepare context for Ragas
            retrieved_text = [c.content for c in results]

            # "Answer" = top-1 chunk (approximation for retrieval-only eval)
            generated_answer = results[0].content if results else "No information found."

            answers.append(generated_answer)
            contexts.append(retrieved_text)

        if on_progress:
            on_progress(total, total, "Calculating metrics with Ragas...")

        # 4. Prepare Ragas Dataset
        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        eval_dataset = Dataset.from_dict(eval_data)

        # 5. Run Ragas Evaluation
        run_config = RunConfig(
            timeout=120,
            max_workers=2,
            max_retries=2,
        )

        try:
            result = evaluate(
                eval_dataset,
                metrics=[context_precision, context_recall],
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=run_config,
                # raise_exceptions=False,
            )
        except Exception as e:
             return None, f"❌ Ragas evaluation failed: {e}"

        return result.to_pandas(), result
