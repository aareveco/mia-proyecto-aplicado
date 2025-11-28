# app.py
import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.infrastructure.services.service_factory import ServiceFactory
from src.application.services.rag_service import VectorStoreService, run_retrieval_service
from src.application.services.evaluation_service import EvaluationService
from src.application.services.indexing_service import BatchIndexingService

# Cargar variables de entorno
load_dotenv()

# ==============================================================================

QDRANT_PATH = "qdrant_storage" 
DATASET_PATH = "datasets/golden_dataset.csv"

st.set_page_config(page_title="Hito 1: Clean RAG Architecture", layout="wide")

# ==============================================================================
# 1. CAPA DE APLICACIÓN (Inicialización)
# ==============================================================================


@st.cache_resource
def get_vector_service() -> VectorStoreService:
    """
    Instancia los adaptadores y el servicio de aplicación usando la Factory.
    """
    return ServiceFactory.get_vector_service(QDRANT_PATH)

# ==============================================================================
# 2. INTERFAZ GRÁFICA
# ==============================================================================

def main():
    st.title("🧪 Hito 1: Clean Architecture RAG")
    st.markdown("Implementación Hexagonal con Evaluación (Gemini + Ragas).")

    # Obtener el servicio (Singleton)
    rag_service = get_vector_service()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Gestión de Datos")
        
        data_folder = "data"
        st.info(f"Usando almacenamiento en: `{QDRANT_PATH}/`")
        
        if st.button("🔄 Indexar (Sobreescribir)"):
            if not os.path.exists(data_folder):
                st.error(f"La carpeta '{data_folder}' no existe.")
            else:
                # Use IndexingService
                indexing_service = BatchIndexingService(rag_service) # No quarantine in UI for now, or maybe? App didn't have it before.

                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(idx, total, fname):
                     progress_bar.progress((idx) / total)
                     status_text.text(f"Indexing: {fname}")

                indexing_service.index_directory(data_folder, on_progress=update_progress)

                progress_bar.progress(1.0)
                status_text.text("Done!")
                st.success("¡Indexado Completo!")
                time.sleep(1)
                st.rerun()

        st.divider()
        top_k = st.slider("Top-K Recuperados", 1, 10, 3)

    # --- TABS ---
    tab1, tab2 = st.tabs(["🔎 Consulta (Search)", "📊 Benchmark Ragas"])

    # --- TAB 1: BÚSQUEDA ---
    with tab1:
        st.subheader("Búsqueda Semántica")
        query = st.text_input("Escribe tu consulta:")

        if query:
            start_time = time.time()
            results = run_retrieval_service(query, rag_service, top_k=top_k)
            end_time = time.time()

            st.markdown(f"**Resultados:** {len(results)} chunks en {end_time - start_time:.3f}s")

            if not results:
                st.warning("No se encontraron resultados.")

            for i, chunk in enumerate(results, start=1):
                # Extraer Score de metadata
                score = 0.0
                if chunk.metadata and "score" in chunk.metadata:
                    score = chunk.metadata["score"]
                
                # Color del score
                score_color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"

                with st.expander(f"Resultado #{i} | Score: :{score_color}[{score:.4f}]"):
                    st.markdown(f"**📄 Fuente:** `{chunk.source_file}` (Pág {chunk.page})")
                    st.markdown(f"**🆔 Chunk ID:** `{chunk.chunk_id}`")
                    st.info(chunk.content)
                    st.json(chunk.metadata, expanded=False)

    # --- TAB 2: BENCHMARK ---
    with tab2:
        st.subheader("Evaluación con Ragas")
        st.markdown(f"Usando dataset: `{DATASET_PATH}` y modelo **Gemini 2.0 Flash**.")

        if st.button("🚀 Ejecutar Benchmark Real"):
            
            llm, embeddings = ServiceFactory.get_evaluation_components()
            evaluator = EvaluationService(rag_service, llm, embeddings)

            # Progress UI
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_eval_progress(idx, total, msg):
                if total > 0:
                    progress_bar.progress(idx / total)
                status_text.text(msg)

            with st.spinner("Ejecutando evaluación..."):
                try:
                    df_res, metrics_obj = evaluator.run_benchmark(DATASET_PATH, on_progress=update_eval_progress)
                    
                    if df_res is None:
                         # metrics_obj holds the error message in this case per my implementation of run_benchmark
                         st.error(metrics_obj)
                    else:
                        st.success("¡Evaluación Completada!")
                        progress_bar.empty()
                        status_text.empty()
                        
                        # 1. Calcular promedios de las métricas de retrieval
                        precision_score = df_res["context_precision"].mean()
                        recall_score = df_res["context_recall"].mean()

                        col1, col2 = st.columns(2)
                        col1.metric("Context Precision", f"{precision_score:.4f}")
                        col2.metric("Context Recall", f"{recall_score:.4f}")

                        # 2. Mostrar tabla con columnas relevantes
                        target_cols = [
                            "user_input",
                            "retrieved_contexts",
                            "context_recall",
                            "context_precision",
                            'response', 'reference'
                        ]

                        # Note: 'response' might not be in df_res if Ragas names it 'answer'. 'reference' is 'ground_truth'.
                        # Let's adjust target_cols based on Ragas standard output or what we injected.
                        # In EvaluationService we mapped: "question", "answer", "contexts", "ground_truth".
                        # Ragas adds metrics columns.

                        # Let's see what columns are actually there.
                        # Ragas typically returns 'question', 'contexts', 'answer', 'ground_truth', 'context_precision', 'context_recall'.

                        # Adjust target cols to match what we likely have
                        possible_cols = [
                            "question", "user_input",
                            "contexts", "retrieved_contexts",
                            "answer", "response",
                            "ground_truth", "reference",
                            "context_recall", "context_precision"
                        ]

                        final_cols = [c for c in possible_cols if c in df_res.columns]

                        st.dataframe(df_res[final_cols], use_container_width=True)
                        
                        # 3. Botón de descarga
                        csv = df_res.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "💾 Descargar Resultados CSV",
                            csv,
                            "ragas_results.csv",
                            "text/csv",
                        )
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
