# src/app.py
import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.infrastructure.bootstrap import create_rag_service
from src.infrastructure.helpers.ingestion_helpers import run_indexing_service
from src.infrastructure.llm.gemini_llm_service import GeminiLLMService

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig

# Cargar variables de entorno
load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

QDRANT_PATH = "qdrant_storage"
DATASET_PATH = "datasets/golden_dataset.csv"

st.set_page_config(
    page_title="Hito 1: Clean RAG Architecture with Metadata", layout="wide"
)

# ==============================================================================
# INITIALIZATION
# ==============================================================================


@st.cache_resource
def get_vector_service():
    """
    Instancia el servicio RAG con todos los componentes.
    """
    return create_rag_service(
        qdrant_path=QDRANT_PATH,
        enable_pubchem=False,  # Cambiar a True si quieres PubChem
        enable_llm_extraction=True,
    )


@st.cache_resource
def get_llm_service():
    """
    Instancia el servicio LLM para extracción de metadata.
    """
    from src.infrastructure.llm.gemini_llm_service import GeminiLLMService
    return GeminiLLMService(model_name="gemini-2.0-flash-exp")


# ==============================================================================
# EVALUATION
# ==============================================================================


class BaselineEvaluator:
    """
    Evaluador real usando Ragas y Local LLM.
    """

    def __init__(self, rag_service):
        self.rag_service = rag_service

    def run_benchmark(self):
        # 1. Verificar Dataset
        if not os.path.exists(DATASET_PATH):
            return (
                None,
                f"❌ No se encontró '{DATASET_PATH}'. Ejecuta 'scripts/generate_test_dataset.py' primero.",
            )

        # 2. Cargar Dataset
        df = pd.read_csv(DATASET_PATH)

        if "question" not in df.columns or "ground_truth" not in df.columns:
            return (
                None,
                "❌ El CSV debe tener columnas 'question' y 'ground_truth'.",
            )

        questions = df["question"].tolist()
        ground_truths = df["reference_contexts"].tolist()

        answers = []
        contexts = []

        # 3. Correr Inferencia (Retrieval)
        progress_bar = st.progress(0)
        status = st.empty()

        total = len(questions)
        for i, q in enumerate(questions):
            status.text(f"Evaluando consulta {i+1}/{total}: {q[:50]}...")

            # Llamada al servicio RAG
            results = self.rag_service.query(q, top_k=3)

            # Preparar contexto para Ragas
            retrieved_text = [c.content for c in results]

            # "Respuesta" = top-1 chunk
            generated_answer = (
                results[0].content if results else "No information found."
            )

            answers.append(generated_answer)
            contexts.append(retrieved_text)
            progress_bar.progress((i + 1) / total)

        # 4. Preparar Dataset Ragas
        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        eval_dataset = Dataset.from_dict(eval_data)

        # 5. Configurar LLM Local
        status.text("Calculando métricas con Ragas (Gemini)...")

        from src.infrastructure.llm.gemini_factory import GeminiFactory

        llm = GeminiFactory.get_generator_llm("gemini-2.0-flash-exp")
        embeddings = GeminiFactory.get_embeddings()

        run_config = RunConfig(
            timeout=120,
            max_workers=4,  # Gemini soporta más concurrencia
            max_retries=2,
        )

        # 6. Ejecutar Ragas
        result = evaluate(
            eval_dataset,
            metrics=[context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )

        progress_bar.empty()

        return result.to_pandas(), result


# ==============================================================================
# UI
# ==============================================================================


def main():
    st.title("🧪 Hito 1: Clean Architecture RAG with Structured Metadata (Gemini)")
    st.markdown(
        """
    Implementación Hexagonal con **Google Gemini**:
    - ✅ Extracción LLM de metadata (m/z, RT, compounds, bioactivities)
    - ✅ Embeddings con Gemini o HuggingFace
    - ✅ Post-filtering por valores numéricos
    - ✅ Query optimization con detección de filtros
    - ✅ Hybrid search (Dense + BM25 + PubChem opcional)
    - ✅ Reranking + Context Repacking
    """
    )

    # Obtener servicios
    rag_service = get_vector_service()
    llm_service = get_llm_service()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Gestión de Datos")

        data_folder = "data"
        st.info(f"Almacenamiento: `{QDRANT_PATH}/`")

        if st.button("🔄 Indexar (Sobreescribir)"):
            if not os.path.exists(data_folder):
                st.error(f"La carpeta '{data_folder}' no existe.")
            else:
                files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
                if not files:
                    st.warning("No hay PDFs.")
                else:
                    bar = st.progress(0)
                    for i, f in enumerate(files):
                        st.info(f"Indexando {f}...")
                        run_indexing_service(
                            os.path.join(data_folder, f),
                            rag_service,
                            llm_service=llm_service,
                            overwrite=(i == 0),  # Solo overwrite en el primero
                        )
                        bar.progress((i + 1) / len(files))
                    st.success("¡Indexado Completo con Metadata LLM!")
                    time.sleep(1)
                    st.rerun()

        st.divider()
        top_k = st.slider("Top-K Recuperados", 1, 10, 3)

    # --- TABS ---
    tab1, tab2 = st.tabs(["🔎 Consulta (Search)", "📊 Benchmark Ragas"])

    # --- TAB 1: BÚSQUEDA ---
    with tab1:
        st.subheader("Búsqueda Semántica con Metadata Filtering")

        # Ejemplos de queries
        st.markdown("**Ejemplos de queries:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Query metabolómico"):
                st.session_state.query = (
                    "Feature m/z 449.107, RT 8.2 en Té Verde. ¿Qué es?"
                )
        with col2:
            if st.button("📚 Query general"):
                st.session_state.query = "¿Qué metabolitos se encuentran en té verde?"

        query = st.text_input(
            "Escribe tu consulta:",
            value=st.session_state.get("query", ""),
            key="query_input",
        )

        if query:
            start_time = time.time()
            results = rag_service.query(query, top_k=top_k)
            end_time = time.time()

            st.markdown(
                f"**Resultados:** {len(results)} chunks en {end_time - start_time:.3f}s"
            )

            if not results:
                st.warning("No se encontraron resultados.")

            for i, chunk in enumerate(results, start=1):
                # Extraer Score
                score = 0.0
                if chunk.metadata and "score" in chunk.metadata:
                    score = chunk.metadata["score"]
                elif chunk.rerank_score is not None:
                    score = chunk.rerank_score

                score_color = (
                    "green" if score > 0.7 else "orange" if score > 0.5 else "red"
                )

                with st.expander(
                    f"Resultado #{i} | Score: :{score_color}[{score:.4f}]"
                ):
                    st.markdown(
                        f"**📄 Fuente:** `{chunk.source_file}` (Pág {chunk.page})"
                    )
                    st.markdown(f"**🆔 Chunk ID:** `{chunk.chunk_id}`")

                    # Mostrar metadata estructurada (si existe) - usar getattr seguro
                    mz_values = getattr(chunk, 'mz_values', None)
                    rt_values = getattr(chunk, 'rt_values', None)
                    compound_names = getattr(chunk, 'compound_names', None)
                    bioactivities = getattr(chunk, 'bioactivities', None)
                    
                    has_structured_meta = mz_values or rt_values or compound_names or bioactivities
                    
                    if has_structured_meta:
                        st.markdown("**🔬 Metadata Estructurada:**")
                        if mz_values:
                            st.markdown(
                                f"- ⚛️ m/z: {', '.join([f'{v:.3f}' for v in mz_values])}"
                            )
                        if rt_values:
                            st.markdown(
                                f"- ⏱️ RT: {', '.join([f'{v:.2f} min' for v in rt_values])}"
                            )
                        if compound_names:
                            st.markdown(
                                f"- 🧪 Compuestos: {', '.join(compound_names)}"
                            )
                        if bioactivities:
                            st.markdown(
                                f"- 🩺 Bioactividades: {', '.join(bioactivities)}"
                            )
                    else:
                        st.info("ℹ️ Este chunk no tiene metadata estructurada. Re-indexa para extraerla con LLM.")

                    st.info(chunk.content)
                    
                    # Mostrar metadata general de forma segura
                    if hasattr(chunk, 'metadata') and chunk.metadata:
                        st.json(chunk.metadata, expanded=False)

    # --- TAB 2: BENCHMARK ---
    with tab2:
        st.subheader("Evaluación con Ragas (Local)")
        st.markdown(f"Usando dataset: `{DATASET_PATH}` y modelo local.")

        if st.button("🚀 Ejecutar Benchmark Real"):
            evaluator = BaselineEvaluator(rag_service)

            with st.spinner("Ejecutando evaluación..."):
                df_res, metrics_obj = evaluator.run_benchmark()

            if isinstance(df_res, str):  # Error
                st.error(df_res)
            else:
                st.success("¡Evaluación Completada!")

                # Calcular promedios
                precision_score = (
                    df_res["context_precision"].mean()
                    if "context_precision" in df_res.columns
                    else 0.0
                )
                recall_score = (
                    df_res["context_recall"].mean()
                    if "context_recall" in df_res.columns
                    else 0.0
                )

                col1, col2 = st.columns(2)
                col1.metric("Context Precision", f"{precision_score:.4f}")
                col2.metric("Context Recall", f"{recall_score:.4f}")

                # Mostrar tabla
                target_cols = [
                    "question",
                    "contexts",
                    "answer",
                    "ground_truth",
                    "context_precision",
                    "context_recall",
                ]
                final_cols = [c for c in target_cols if c in df_res.columns]

                st.dataframe(df_res[final_cols], use_container_width=True)

                # Descarga
                csv = df_res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Descargar Resultados CSV",
                    csv,
                    "ragas_results.csv",
                    "text/csv",
                )


if __name__ == "__main__":
    main()