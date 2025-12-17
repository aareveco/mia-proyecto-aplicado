import os
import time
import pandas as pd
import asyncio
import streamlit as st
from dotenv import load_dotenv

from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService, run_retrieval_service
from src.infrastructure.llm.local_llm_factory import LocalResourcesFactory
# Infrastructure Imports for Injection
from src.infrastructure.llm.local_llm_service import LocalLLMService
from src.infrastructure.reranker.cross_encoder_reranker import CrossEncoderRerankerService
from src.infrastructure.retrieval.bm25_service import BM25RetrieverImpl
from src.infrastructure.loaders.factory import DocumentLoaderFactory

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig

# Cargar variables de entorno
load_dotenv()

# ==============================================================================

QDRANT_PATH = "qdrant_storage" # La misma ruta  que run_eval.py
DATASET_PATH = "datasets/golden_dataset.csv"

st.set_page_config(page_title="Hito 1: Clean RAG Architecture", layout="wide")

# ==============================================================================
# Helper Function (Moved from rag_service.py to keep Application layer clean)
# ==============================================================================
from src.application.services.ingestion_pipeline import IngestionPipeline
from src.infrastructure.processors.processors import CleanerProcessor, MetadataExtractorProcessor

def run_indexing_service(
    file_path: str,
    vector_store: VectorStoreService,
    overwrite: bool = False,
) -> None:
    loader = DocumentLoaderFactory.get_loader(file_path)
    chunks = loader.load_and_chunk(file_path)
    
    # --- PIPELINE STEP ---
    pipeline = IngestionPipeline([
        CleanerProcessor(),
        MetadataExtractorProcessor()
    ])
    
    print("[Index] Ejecutando Pipeline de Ingesta (Limpieza + Extracción)...")
    refined_chunks = pipeline.run(chunks)
    # ---------------------

    print("[Index] Generando embeddings e indexando en Qdrant...")
    vector_store.index_chunks(refined_chunks, overwrite=overwrite)
    print("[Index] Listo.")

# ==============================================================================
# 1. CAPA DE APLICACIÓN (Inicialización)
# ==============================================================================


@st.cache_resource
def get_vector_service() -> VectorStoreService:
    """
    Instancia los adaptadores y el servicio de aplicación.
    Usa persistencia en disco para compartir datos con los scripts.
    """
    # 1. Adaptador de Embeddings
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    
    # 2. Adaptador de Base de Datos Vectorial (CON PERSISTENCIA)
    db_impl = QdrantImpl(collection_name="rag_chunks", path=QDRANT_PATH)
    
    # 3. New Infrastructure Services
    llm_service = LocalLLMService()
    reranker_service = CrossEncoderRerankerService()
    bm25_service = BM25RetrieverImpl(storage_path="data/bm25_index.pkl")

    # 4. Inyección de dependencias
    # Force cache invalidation for overwrite flag update
    service = VectorStoreService(
        embedder=embedder, 
        db_impl=db_impl,
        llm_service=llm_service,
        reranker_service=reranker_service,
        sparse_retriever=bm25_service
    )
    return service

# ==============================================================================
# 2. SISTEMA DE BENCHMARK REAL
# ==============================================================================

class BaselineEvaluator:
    """
    Evaluador real usando Ragas y Local LLM.
    """
    def __init__(self, rag_service: VectorStoreService):
        self.rag_service = rag_service

    def run_benchmark(self):
        # 1. Verificar Dataset
        if not os.path.exists(DATASET_PATH):
            return None, "❌ No se encontró 'datasets/golden_dataset.csv'. Ejecuta primero 'scripts/generate_dataset.py'."

        # 2. Cargar Dataset
        df = pd.read_csv(DATASET_PATH)
        

        if "question" not in df.columns or "ground_truth" not in df.columns:
            return None, "❌ El CSV debe tener columnas 'question' y 'ground_truth'."

        questions = df["question"].tolist()
        ground_truths = df["reference_contexts"].tolist() #por ahora usamos esto dado que no tenemos la patrte de Generación
        
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

        # 5. Configurar LLM Local para Evaluación
        status.text("Calculando métricas con Ragas (esto puede tardar)...")
        
        llm = LocalResourcesFactory.get_generator_llm("qwen2.5:1.5b")
        embeddings = LocalResourcesFactory.get_embeddings()

        # RunConfig para evitar TimeoutError con LLM local
        run_config = RunConfig(
            timeout=120,     # súbelo si aún ves timeouts (180-300)
            max_workers=2,   # menos concurrencia para no ahogar la máquina
            max_retries=2,
        )

        # 6. Ejecutar Ragas con métricas de retrieval
        result = evaluate(
            eval_dataset,
            metrics=[context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )

        progress_bar.empty()
        
        # Retornar DataFrame de resultados y objeto Ragas
        return result.to_pandas(), result


# ==============================================================================
# 3. INTERFAZ GRÁFICA
# ==============================================================================

def main():
    st.title("🧪 Hito 1: Clean Architecture RAG")
    st.markdown("Implementación Hexagonal con Evaluación Local (Ollama + Ragas).")

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
                        run_indexing_service(os.path.join(data_folder, f), rag_service, overwrite=True)
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
        st.subheader("Evaluación con Ragas (Local)")
        st.markdown(f"Usando dataset: `{DATASET_PATH}` y modelo local.")

        if st.button("🚀 Ejecutar Benchmark Real"):
            evaluator = BaselineEvaluator(rag_service)
            
            with st.spinner("Ejecutando evaluación..."):
                df_res, metrics_obj = evaluator.run_benchmark()
            
            if isinstance(df_res, str): # Manejo de errores
                st.error(df_res)
            else:
                st.success("¡Evaluación Completada!")
                
                # 1. Calcular promedios de las métricas de retrieval
                if "context_precision" in df_res.columns:
                    precision_score = df_res["context_precision"].mean()
                else:
                    precision_score = 0.0
                    st.warning("⚠️ 'context_precision' no se calculó correctamente.")

                if "context_recall" in df_res.columns:
                    recall_score = df_res["context_recall"].mean()
                else:
                    recall_score = 0.0
                    st.warning("⚠️ 'context_recall' no se calculó correctamente.")

                col1, col2 = st.columns(2)
                col1.metric("Context Precision", f"{precision_score:.4f}")
                col2.metric("Context Recall", f"{recall_score:.4f}")

                # 2. Mostrar tabla con columnas relevantes
                target_cols = [
                    'question', 'contexts', 'answer', 'ground_truth',
       'context_precision', 'context_recall'
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

if __name__ == "__main__":
    main()