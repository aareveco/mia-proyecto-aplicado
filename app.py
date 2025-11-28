# app.py
import os
import time
import pandas as pd
import asyncio
import streamlit as st
from dotenv import load_dotenv

# --- CHANGED: Removed HuggingFaceEmbedder, Added GeminiFactory ---
# from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.llm.gemini_factory import GeminiFactory
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService, run_indexing_service, run_retrieval_service

# --- REMOVED: LocalResourcesFactory is no longer needed if using Gemini ---
# from src.infrastructure.llm.local_llm_factory import LocalResourcesFactory

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig

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
    Instancia los adaptadores y el servicio de aplicación.
    Usa persistencia en disco para compartir datos con los scripts.
    """
    # --- CHANGED: Use Gemini Embedder to match run_eval.py ---
    # embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    embedder = GeminiFactory.get_app_embedder(model_name="models/text-embedding-004")
    
    # 2. Adaptador de Base de Datos Vectorial (CON PERSISTENCIA)
    db_impl = QdrantImpl(collection_name="rag_chunks", path=QDRANT_PATH)
    
    # 3. Inyección de dependencias
    service = VectorStoreService(embedder=embedder, db_impl=db_impl)
    return service

# ==============================================================================
# 2. SISTEMA DE BENCHMARK REAL
# ==============================================================================

class BaselineEvaluator:
    """
    Evaluador real usando Ragas y Gemini (Mirroring run_eval.py).
    """
    def __init__(self, rag_service: VectorStoreService):
        self.rag_service = rag_service

    def run_benchmark(self):
        # 1. Verificar Dataset
        if not os.path.exists(DATASET_PATH):
            return None, "❌ No se encontró 'datasets/golden_dataset.csv'. Ejecuta primero 'scripts/generate_dataset.py'."

        # 2. Cargar Dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Validar columnas necesarias
        if "user_input" not in df.columns or "reference" not in df.columns:
            return None, "❌ El CSV debe tener columnas 'user_input' y 'reference'."

        questions = df["user_input"].tolist()
        ground_truths = df["reference"].tolist()
        
        answers = []
        contexts = []

        # 3. Correr Inferencia (Retrieval)
        progress_bar = st.progress(0)
        status = st.empty()
        
        total = len(questions)
        for i, q in enumerate(questions):
            status.text(f"Evaluando consulta {i+1}/{total}: {q[:50]}...")
            
            # Llamada al servicio RAG
            results = run_retrieval_service(q, self.rag_service, top_k=3)
            
            # Preparar contexto para Ragas (lista de strings)
            retrieved_text = [c.content for c in results]
            
            # "Respuesta" = top-1 chunk, solo para cumplir schema
            generated_answer = results[0].content if results else "No information found."
            
            answers.append(generated_answer)
            contexts.append(retrieved_text)
            progress_bar.progress((i + 1) / total)

        # 4. Preparar Dataset Ragas (igual que run_eval.py)
        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,

            
        }
        eval_dataset = Dataset.from_dict(eval_data)

        # 5. Configurar LLM para Evaluación (Judge)
        status.text("Calculando métricas con Ragas (usando Gemini)...")
        
        # --- CHANGED: Use Gemini for Evaluation Judge instead of LocalResourcesFactory ---
        # llm = LocalResourcesFactory.get_generator_llm("qwen2.5:1.5b")
        # embeddings = LocalResourcesFactory.get_embeddings()
        
        llm = GeminiFactory.get_generator_llm("gemini-2.0-flash")
        embeddings = GeminiFactory.get_embeddings()

        # RunConfig para evitar TimeoutError
        run_config = RunConfig(
            timeout=120,     
            max_workers=2,   
            max_retries=2,
        )

        # 6. Ejecutar Ragas con métricas de retrieval
        result = evaluate(
            eval_dataset,
            metrics=[context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            # raise_exceptions=False, 
        )

        progress_bar.empty()
        
        # Retornar DataFrame de resultados y objeto Ragas
        return result.to_pandas(), result


# ==============================================================================
# 3. INTERFAZ GRÁFICA
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
                files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
                if not files:
                    st.warning("No hay PDFs.")
                else:
                    bar = st.progress(0)
                    for i, f in enumerate(files):
                        run_indexing_service(os.path.join(data_folder, f), rag_service)
                        bar.progress((i+1)/len(files))
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
            evaluator = BaselineEvaluator(rag_service)
            
            with st.spinner("Ejecutando evaluación..."):
                try:
                    df_res, metrics_obj = evaluator.run_benchmark()
                    
                    if isinstance(df_res, str): # Manejo de errores
                        st.error(df_res)
                    else:
                        st.success("¡Evaluación Completada!")
                        
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
                        print(df_res.columns)
                        final_cols = [c for c in target_cols if c in df_res.columns]

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