#!/usr/bin/env python3
"""
3_streamlit_app.py - Interfaz Web RAG con Streamlit
Visualización completa del proceso RAG
"""

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from rag_core import (
    OpenAILLM,
    CrossEncoderReranker,
    SentenceTransformerAPI,
    QdrantVectorStore,
    BM25API,
    HybridSearchStrategy,
    QueryRewritingStrategy,
    QueryOptimizerRetriever,
    RerankingDecorator,
    ContextRepackerDecorator,
)

# Configuración de página
st.set_page_config(
    page_title="RAG Bio-Actives",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chunk-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stage-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Inicializa el sistema RAG (se ejecuta una sola vez)"""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "metabolomics_agent_db")
    
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY no configurada en .env")
        st.stop()
    
    with st.spinner("🔄 Cargando modelos..."):
        # LLM
        llm = OpenAILLM(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
        
        # Reranker
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Embeddings
        embedding_model = SentenceTransformerAPI(model_name="all-MiniLM-L6-v2")
        
        # BM25
        bm25_api = BM25API()
        
        # Qdrant
        db = QdrantVectorStore(
            collection_name=COLLECTION_NAME, 
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY,
            bm25_encoder=bm25_api
        )
        
        # Pipeline completo
        base_strategy = HybridSearchStrategy(db, embedding_model)
        retriever_with_c6 = QueryOptimizerRetriever(
            query_processor=QueryRewritingStrategy(llm),
            retrieval_strategy=base_strategy,
        )
        reranked_retriever = RerankingDecorator(
            wrapped_strategy=retriever_with_c6,
            reranker=reranker,
        )
        final_pipeline = ContextRepackerDecorator(reranked_retriever)
    
    return final_pipeline, COLLECTION_NAME, QDRANT_URL


def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 RAG Bio-Actives</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema de Anotación de Compuestos Metabolómicos</p>', unsafe_allow_html=True)
    
    # Inicializar sistema
    try:
        final_pipeline, collection_name, qdrant_url = initialize_rag_system()
        st.success("✅ Sistema RAG inicializado correctamente")
    except Exception as e:
        st.error(f"❌ Error al inicializar sistema: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        st.metric("Colección", collection_name)
        st.metric("Qdrant URL", qdrant_url)
        
        st.divider()
        
        st.subheader("🔧 Parámetros")
        k_results = st.slider("Número de chunks", 1, 10, 3)
        show_process = st.checkbox("Mostrar proceso detallado", value=True)
        
        st.divider()
        
        st.subheader("📚 Pipeline")
        st.markdown("""
        **C2:** PDF Loader  
        **C3:** Embeddings (Dense + Sparse)  
        **C4:** Búsqueda Híbrida (Qdrant)  
        **C6:** Query Rewriting (OpenAI)  
        **C7:** Reranking (Cross-Encoder)  
        **C7:** Context Repacking  
        """)
        
        st.divider()
        
        st.subheader("💡 Ejemplos")
        if st.button("Ejemplo: Feature m/z"):
            st.session_state.example_query = "Tengo una feature con m/z 449.107, RT 8.2 min, detectada en mi muestra de Té Verde. ¿Qué es y qué bioactividad tiene?"
        
        if st.button("Ejemplo: Bioactividad"):
            st.session_state.example_query = "¿Qué compuestos tienen actividad antioxidante?"
        
        if st.button("Ejemplo: Método"):
            st.session_state.example_query = "Buscar features detectadas por LC-MS"
    
    # Main content
    st.divider()
    
    # Query input
    query_input = st.text_area(
        "🔍 Tu consulta metabolómica:",
        value=st.session_state.get('example_query', ''),
        height=100,
        placeholder="Ejemplo: Tengo una feature con m/z 449.107, RT 8.2 min en Té Verde. ¿Qué es?"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("🚀 Buscar", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Limpiar", use_container_width=True)
    
    if clear_button:
        st.session_state.example_query = ""
        st.rerun()
    
    if search_button and query_input:
        
        # Contenedor para el proceso
        process_container = st.container()
        
        with process_container:
            if show_process:
                st.header("📊 Proceso de Recuperación")
                
                # Stage 1: Query Original
                with st.expander("1️⃣ Query Original", expanded=True):
                    st.markdown(f"**Query del usuario:**")
                    st.info(query_input)
            
            # Ejecutar búsqueda
            with st.spinner("🔄 Procesando query..."):
                try:
                    result = final_pipeline.retrieve_context(
                        query=query_input, 
                        filters={}, 
                        k=k_results
                    )
                    
                    # Desempaquetar resultado
                    if isinstance(result, tuple):
                        context_list, filter_suggestion = result
                    else:
                        context_list = result
                        filter_suggestion = None
                    
                    if show_process and filter_suggestion:
                        # Stage 2: Query Rewriting (C6)
                        with st.expander("2️⃣ Query Rewriting (C6 - OpenAI)", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Query Reescrita:**")
                                st.success(filter_suggestion.rewritten_query)
                            
                            with col2:
                                st.markdown("**Filtros Sugeridos:**")
                                if filter_suggestion.metadata_filters:
                                    for key, value in filter_suggestion.metadata_filters.items():
                                        st.markdown(f"- `{key}`: **{value}**")
                                else:
                                    st.markdown("*Sin filtros sugeridos*")
                        
                        # Stage 3: Búsqueda Híbrida (C4)
                        with st.expander("3️⃣ Búsqueda Híbrida (C4 - Qdrant)", expanded=True):
                            st.markdown("**Técnica:** Dense Embeddings + Sparse BM25 + RRF Fusion")
                            st.markdown(f"**Candidatos recuperados:** {len(context_list)} chunks")
                            
                            if context_list:
                                st.markdown("**Preview de candidatos:**")
                                for i, chunk in enumerate(context_list[:3]):
                                    st.markdown(f"- Chunk {i+1}: `{chunk.chunk_id}` (Método: {chunk.experimental_method})")
                        
                        # Stage 4: Reranking (C7)
                        with st.expander("4️⃣ Reranking (C7 - Cross-Encoder)", expanded=True):
                            st.markdown("**Modelo:** ms-marco-MiniLM-L-6-v2")
                            
                            if context_list and context_list[0].rerank_score:
                                scores_data = {
                                    "Chunk": [f"Chunk {i+1}" for i in range(len(context_list))],
                                    "Score": [c.rerank_score for c in context_list]
                                }
                                st.bar_chart(scores_data, x="Chunk", y="Score")
                                
                                st.markdown(f"**Top Score:** {context_list[0].rerank_score:.4f}")
                        
                        # Stage 5: Context Repacking (C7)
                        with st.expander("5️⃣ Context Repacking (C7)", expanded=True):
                            st.markdown("**Técnica:** Sides Repacking - El chunk más relevante primero")
                            st.markdown(f"**Orden final:** Top-1 + Resto ordenado por score")
                    
                    # Resultados finales
                    st.divider()
                    st.header("✨ Resultados Finales")
                    
                    if not context_list:
                        st.warning("No se encontraron resultados para esta consulta.")
                    else:
                        # Métricas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Chunks Recuperados", len(context_list))
                        with col2:
                            avg_score = sum(c.rerank_score for c in context_list if c.rerank_score) / len(context_list)
                            st.metric("Score Promedio", f"{avg_score:.4f}")
                        with col3:
                            methods = set(c.experimental_method for c in context_list)
                            st.metric("Métodos Detectados", len(methods))
                        
                        st.divider()
                        
                        # Mostrar chunks
                        for i, chunk in enumerate(context_list):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"### 📄 Chunk {i+1}: `{chunk.chunk_id}`")
                                
                                with col2:
                                    score_str = f"{chunk.rerank_score:.4f}" if chunk.rerank_score else "N/A"
                                    st.metric("Score", score_str)
                                
                                # Metadata
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown(f"**Método:** {chunk.experimental_method}")
                                with col2:
                                    st.markdown(f"**Año:** {chunk.publication_year}")
                                with col3:
                                    st.markdown(f"**Fuente:** {chunk.source_file}")
                                
                                # Contenido
                                st.markdown("**Contenido:**")
                                st.markdown(f'<div class="chunk-card">{chunk.content}</div>', unsafe_allow_html=True)
                                
                                st.divider()
                    
                except Exception as e:
                    st.error(f"❌ Error al procesar consulta: {e}")
                    st.exception(e)


if __name__ == "__main__":
    main()
