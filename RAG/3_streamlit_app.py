#!/usr/bin/env python3
"""
3_streamlit_app.py - Interfaz Web RAG con Streamlit
Visualización completa del proceso RAG con UI/UX mejorada
"""

import streamlit as st
import os
from dotenv import load_dotenv
import time
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
    page_title="RAG Bio-Actives 🧬",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.markdown("""
<style>
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Fuentes y colores principales */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.8s ease-in;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Animaciones */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Cards de chunks mejoradas */
    .chunk-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-out;
    }
    
    .chunk-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-left-color: #764ba2;
    }
    
    /* Badges y etiquetas */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .badge-info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    /* Stage boxes mejoradas */
    .stage-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #fbbf24;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        animation: slideIn 0.5s ease-out;
    }
    
    .stage-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Métricas mejoradas */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Progress bar personalizada */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Botones mejorados */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Sidebar mejorada */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Input mejorado */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Divider personalizado */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    }
    
    /* Score bar */
    .score-bar {
        height: 8px;
        background: #e5e7eb;
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        border-radius: 9999px;
        transition: width 0.5s ease;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Pipeline icons */
    .pipeline-step {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: white;
        border-radius: 8px;
        margin: 0.25rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e5e7eb 0%, #d1d5db 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    """Inicializa el sistema RAG (se ejecuta una sola vez)"""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "metabolomics_agent_db")
    
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY no configurada en .env")
        st.stop()
    
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


def render_metric_card(label, value, icon):
    """Renderiza una tarjeta de métrica con estilo"""
    return f"""
    <div class="metric-container">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_chunk_card_native(chunk, index):
    """Renderiza una tarjeta de chunk usando componentes nativos de Streamlit"""
    score_str = f"{chunk.rerank_score:.4f}" if chunk.rerank_score else "N/A"
    
    # Crear card con Streamlit nativo
    with st.container():
        # Header del chunk
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 📄 Chunk {index + 1}")
            st.code(chunk.chunk_id, language=None)
        
        with col2:
            st.metric("Score", score_str)
        
        # Barra de progreso del score
        if chunk.rerank_score:
            # Normalizar score a 0-1
            normalized_score = max(0, min(1, (chunk.rerank_score + 5) / 20))
            st.progress(normalized_score)
        
        # Metadata
        st.markdown(f"📅 **Año:** {chunk.publication_year}")
        
        # Metadata estructurada (si existe)
        if chunk.mz_values or chunk.rt_values or chunk.compound_names or chunk.bioactivities:
            with st.expander("🔍 Metadata Estructurada", expanded=False):
                if chunk.mz_values:
                    st.markdown(f"**⚛️ m/z values:** {', '.join([f'{v:.3f}' for v in chunk.mz_values])}")
                if chunk.rt_values:
                    st.markdown(f"**⏱️ RT values:** {', '.join([f'{v:.2f} min' for v in chunk.rt_values])}")
                if chunk.compound_names:
                    st.markdown(f"**🧪 Compuestos:** {', '.join(chunk.compound_names)}")
                if chunk.bioactivities:
                    st.markdown(f"**🎯 Bioactividades:** {', '.join(chunk.bioactivities)}")
        
        # Contenido
        st.markdown("**📝 Contenido:**")
        st.info(chunk.content)
        
        # Fuente
        st.caption(f"📂 Fuente: `{chunk.source_file}`")
        
        st.divider()


def render_chunk_card(chunk, index):
    """Renderiza una tarjeta de chunk con diseño mejorado"""
    score_str = f"{chunk.rerank_score:.4f}" if chunk.rerank_score else "N/A"
    
    # Normalizar score: típicamente van de -5 a +15, normalizamos a 0-100%
    if chunk.rerank_score:
        # Escala: scores negativos = 0%, score 0 = 25%, score 5 = 50%, score 10+ = 100%
        normalized_score = max(0, min(100, (chunk.rerank_score + 5) * 6.67))
        score_percentage = normalized_score
    else:
        score_percentage = 0
    
    return f"""
    <div class="chunk-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
            <div>
                <h3 style="margin: 0; color: #1f2937; font-size: 1.25rem;">
                    📄 Chunk {index + 1}
                </h3>
                <p style="margin: 0.25rem 0 0 0; color: #6b7280; font-size: 0.875rem;">
                    <code>{chunk.chunk_id}</code>
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #667eea;">
                    {score_str}
                </div>
                <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">
                    Relevance Score
                </div>
            </div>
        </div>
        
        <div class="score-bar">
            <div class="score-fill" style="width: {score_percentage}%;"></div>
        </div>
        
        <div style="display: flex; gap: 0.5rem; margin: 1rem 0;">
            <span class="badge badge-success">
                📅 {chunk.publication_year}
            </span>
        </div>
        
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <div style="font-size: 0.875rem; color: #4b5563; margin-bottom: 0.5rem; font-weight: 600;">
                📝 Contenido:
            </div>
            <div style="color: #1f2937; line-height: 1.6;">
                {chunk.content}
            </div>
        </div>
        
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
            <div style="font-size: 0.875rem; color: #6b7280;">
                📂 Fuente: <code>{chunk.source_file}</code>
            </div>
        </div>
    </div>
    """


def main():
    # Header con animación
    st.markdown('<h1 class="main-header">🧬 RAG Bio-Actives</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema Inteligente de Anotación de Compuestos Metabolómicos</p>', unsafe_allow_html=True)
    
    # Inicializar sistema con spinner bonito
    with st.spinner("🔄 Inicializando sistema RAG..."):
        try:
            final_pipeline, collection_name, qdrant_url = initialize_rag_system()
            st.markdown("""
            <div class="success-box">
                ✅ <strong>Sistema RAG inicializado correctamente</strong>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error al inicializar sistema: {e}")
            st.stop()
    
    # Sidebar mejorada
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        
        # Métricas en el sidebar
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 8px;">
                <div style="font-size: 1.5rem;">🗄️</div>
                <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">Colección</div>
                <div style="font-size: 0.875rem; font-weight: 600; color: #1f2937; margin-top: 0.25rem;">{collection_name[:15]}...</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 8px;">
                <div style="font-size: 1.5rem;">🌐</div>
                <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">Estado</div>
                <div style="font-size: 0.875rem; font-weight: 600; color: #059669; margin-top: 0.25rem;">Conectado</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔧 Parámetros")
        k_results = st.slider("📊 Número de chunks", 1, 10, 3, help="Cantidad de chunks más relevantes a recuperar")
        show_process = st.toggle("🔍 Mostrar proceso detallado", value=True, help="Visualizar cada etapa del pipeline RAG")
        show_scores = st.toggle("📈 Mostrar gráfico de scores", value=True, help="Visualizar scores de reranking")
        
        st.markdown("---")
        
        st.markdown("### 📚 Pipeline RAG")
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; font-size: 0.875rem;">
            <div class="pipeline-step">
                <span style="font-size: 1.25rem;">📄</span>
                <strong>C2:</strong> PDF Loader
            </div>
            <div class="pipeline-step">
                <span style="font-size: 1.25rem;">🔢</span>
                <strong>C3:</strong> Embeddings
            </div>
            <div class="pipeline-step">
                <span style="font-size: 1.25rem;">🔍</span>
                <strong>C4:</strong> Hybrid Search
            </div>
            <div class="pipeline-step">
                <span style="font-size: 1.25rem;">✏️</span>
                <strong>C6:</strong> Query Rewriting
            </div>
            <div class="pipeline-step">
                <span style="font-size: 1.25rem;">🎯</span>
                <strong>C7:</strong> Reranking
            </div>
            <div class="pipeline-step">
                <span style="font-size: 1.25rem;">📦</span>
                <strong>C7:</strong> Repacking
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 💡 Consultas de Ejemplo")
        
        example_queries = {
            "🔬 Feature m/z + RT": "Tengo una feature con m/z 449.107, RT 8.2 min, detectada en mi muestra de Té Verde. ¿Qué es y qué bioactividad tiene?",
            "💊 Bioactividad Antioxidante": "¿Qué compuestos tienen actividad antioxidante reportada?",
            "🧪 Método LC-MS": "Buscar features detectadas por LC-MS con propiedades antidiabéticas",
            "🍇 Compuesto Arándano": "¿Qué compuestos bioactivos se han encontrado en arándano?",
        }
        
        for label, query in example_queries.items():
            if st.button(label, use_container_width=True, key=f"example_{label}"):
                st.session_state.example_query = query
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 8px; font-size: 0.875rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚡</div>
            <strong>Tip:</strong> Sé específico con m/z, RT y matriz para mejores resultados
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Query input mejorado
    st.markdown("### 🔍 Tu Consulta Metabolómica")
    query_input = st.text_area(
        "Query input",
        value=st.session_state.get('example_query', ''),
        height=120,
        placeholder="💬 Ejemplo: Tengo una feature con m/z 449.107, RT 8.2 min en Té Verde. ¿Qué es y qué bioactividad tiene?",
        label_visibility="collapsed"
    )
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 3])
    with col1:
        search_button = st.button("🚀 Buscar Compuestos", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Limpiar", use_container_width=True)
    
    if clear_button:
        st.session_state.example_query = ""
        st.rerun()
    
    if search_button and query_input:
        
        # Progress bar animado
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Contenedor para el proceso
        process_container = st.container()
        
        with process_container:
            if show_process:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("## 📊 Proceso de Recuperación")
                st.markdown("---")
                
                # Stage 1: Query Original
                with st.expander("1️⃣ Query Original del Usuario", expanded=True):
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>📝 Consulta recibida:</strong><br>
                        <div style="margin-top: 0.5rem; padding: 1rem; background: white; border-radius: 6px; font-size: 1.05rem;">
                            {query_input}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Ejecutar búsqueda con animación
            progress_bar.progress(20)
            status_text.text("🔄 Procesando query con OpenAI...")
            
            try:
                time.sleep(0.3)  # Pausa para efecto visual
                result = final_pipeline.retrieve_context(
                    query=query_input, 
                    filters={}, 
                    k=k_results
                )
                
                progress_bar.progress(60)
                status_text.text("🔍 Buscando en base de datos vectorial...")
                time.sleep(0.3)
                
                # Desempaquetar resultado
                if isinstance(result, tuple):
                    context_list, filter_suggestion = result
                else:
                    context_list = result
                    filter_suggestion = None
                
                progress_bar.progress(80)
                status_text.text("🎯 Aplicando reranking...")
                time.sleep(0.3)
                
                progress_bar.progress(100)
                status_text.text("✅ ¡Búsqueda completada!")
                time.sleep(0.5)
                
                # Limpiar progress
                progress_bar.empty()
                status_text.empty()
                
                if show_process and filter_suggestion:
                    # Stage 2: Query Rewriting (C6)
                    with st.expander("2️⃣ Query Rewriting (C6 - OpenAI GPT-4)", expanded=True):
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown("**✏️ Query Optimizada:**")
                            st.markdown(f"""
                            <div class="success-box">
                                {filter_suggestion.rewritten_query}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**🏷️ Filtros Detectados:**")
                            if filter_suggestion.metadata_filters:
                                filters_html = ""
                                for key, value in filter_suggestion.metadata_filters.items():
                                    # Formatear según el tipo de valor
                                    if key == "target_mz":
                                        display_key = "⚛️ m/z"
                                        display_value = f"{value:.3f} Da"
                                    elif key == "target_rt":
                                        display_key = "⏱️ RT"
                                        display_value = f"{value:.2f} min"
                                    elif key == "publication_year":
                                        display_key = "📅 Año"
                                        display_value = str(value)
                                    else:
                                        display_key = key
                                        display_value = str(value)
                                    
                                    filters_html += f"""
                                    <div style="background: white; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 3px solid #3b82f6;">
                                        <div style="font-size: 0.875rem; color: #6b7280;">{display_key}</div>
                                        <div style="font-size: 1.125rem; font-weight: 600; color: #1f2937;">{display_value}</div>
                                    </div>
                                    """
                                st.markdown(filters_html, unsafe_allow_html=True)
                                
                                # Mostrar info de filtrado
                                st.info(f"ℹ️ Búsqueda dirigida: {len(filter_suggestion.metadata_filters)} filtros aplicados para encontrar chunks relevantes.")
                            else:
                                st.info("Sin filtros específicos - Búsqueda general")
                    
                    # Stage 3: Búsqueda Híbrida (C4)
                    with st.expander("3️⃣ Búsqueda Híbrida (C4 - Qdrant + RRF)", expanded=True):
                        st.markdown("""
                        <div class="info-box">
                            <strong>🔬 Técnica Utilizada:</strong> Dense Embeddings + Sparse BM25 + Reciprocal Rank Fusion (RRF)
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🎯 Candidatos", len(context_list), help="Chunks recuperados antes de reranking")
                        with col2:
                            st.metric("🔢 Vector Dim", "384", help="Dimensión de embeddings densos")
                        
                        # Mostrar metadata de los chunks recuperados
                        if context_list:
                            with st.expander("📋 Metadata de chunks recuperados", expanded=False):
                                for i, chunk in enumerate(context_list[:3]):  # Mostrar primeros 3
                                    st.markdown(f"""
                                    **Chunk {i+1}**: `{chunk.chunk_id}`  
                                    - Año: **{chunk.publication_year}** {"✅" if not filter_suggestion or not filter_suggestion.metadata_filters.get('publication_year') or chunk.publication_year == filter_suggestion.metadata_filters.get('publication_year') else "❌"}
                                    """)
                                if len(context_list) > 3:
                                    st.caption(f"... y {len(context_list) - 3} chunks más")
                        with col3:
                            st.metric("⚡ Método", "Híbrido", help="Dense + Sparse fusion")
                        
                        if context_list:
                            st.markdown("**📋 Preview de Candidatos:**")
                            preview_html = ""
                            for i, chunk in enumerate(context_list[:5]):
                                preview_html += f"""
                                <div style="background: white; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 3px solid #3b82f6;">
                                    <strong>Chunk {i+1}:</strong> <code>{chunk.chunk_id}</code>
                                </div>
                                """
                            st.markdown(preview_html, unsafe_allow_html=True)
                    
                    # Stage 4: Reranking (C7)
                    with st.expander("4️⃣ Reranking Semántico (C7 - Cross-Encoder)", expanded=True):
                        st.markdown("""
                        <div class="info-box">
                            <strong>🤖 Modelo:</strong> ms-marco-MiniLM-L-6-v2 (Microsoft)<br>
                            <strong>📊 Técnica:</strong> Cross-attention entre query y documentos
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if context_list and context_list[0].rerank_score and show_scores:
                            import pandas as pd
                            
                            scores_df = pd.DataFrame({
                                'Chunk': [f"Chunk {i+1}" for i in range(len(context_list))],
                                'Score': [c.rerank_score for c in context_list]
                            })
                            
                            st.bar_chart(scores_df.set_index('Chunk'), height=300)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🏆 Top Score", f"{context_list[0].rerank_score:.4f}")
                            with col2:
                                avg_score = sum(c.rerank_score for c in context_list) / len(context_list)
                                st.metric("📊 Score Promedio", f"{avg_score:.4f}")
                            with col3:
                                score_range = context_list[0].rerank_score - context_list[-1].rerank_score
                                st.metric("📏 Rango", f"{score_range:.4f}")
                    
                    # Stage 5: Context Repacking (C7)
                    with st.expander("5️⃣ Context Repacking (C7 - Optimización)", expanded=True):
                        st.markdown("""
                        <div class="success-box">
                            <strong>📦 Técnica:</strong> Sides Repacking<br>
                            <strong>🎯 Objetivo:</strong> Posicionar el chunk más relevante al inicio<br>
                            <strong>✨ Beneficio:</strong> Mejora la atención del LLM en generación final
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**🔄 Orden Final de Chunks:**")
                        order_html = ""
                        for i, chunk in enumerate(context_list):
                            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📄"
                            order_html += f"""
                            <div style="display: flex; align-items: center; gap: 1rem; background: white; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.5rem;">{icon}</span>
                                <div style="flex: 1;">
                                    <strong>{chunk.chunk_id}</strong><br>
                                    <span style="font-size: 0.875rem; color: #6b7280;">Score: {chunk.rerank_score:.4f}</span>
                                </div>
                            </div>
                            """
                        st.markdown(order_html, unsafe_allow_html=True)
                
                # Resultados finales
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("## ✨ Resultados Finales")
                st.markdown("---")
                
                if not context_list:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ No se encontraron resultados</strong><br>
                        Intenta reformular tu consulta o usar términos más específicos.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Métricas superiores con diseño mejorado
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(render_metric_card(
                            "Chunks Recuperados",
                            len(context_list),
                            "📄"
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        avg_score = sum(c.rerank_score for c in context_list if c.rerank_score) / len(context_list)
                        st.markdown(render_metric_card(
                            "Score Promedio",
                            f"{avg_score:.4f}",
                            "📊"
                        ), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(render_metric_card(
                            "Chunks Recuperados",
                            len(context_list),
                            "📄"
                        ), unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Mostrar chunks con diseño mejorado (componentes nativos)
                    for i, chunk in enumerate(context_list):
                        render_chunk_card_native(chunk, i)
                    
                    # ============== GENERACIÓN DE RESPUESTA FINAL CON LLM ==============
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("## 🤖 Respuesta Generada por IA")
                    st.markdown("---")
                    
                    with st.spinner("🧠 Generando respuesta basada en los chunks recuperados..."):
                        try:
                            # Construir contexto para el LLM
                            context_text = "\n\n".join([
                                f"[Chunk {i+1} - Score: {chunk.rerank_score:.4f}]\n{chunk.content}"
                                for i, chunk in enumerate(context_list)
                            ])
                            
                            # Prompt para generación
                            generation_prompt = f"""Eres un experto en metabolómica y anotación de compuestos bioactivos.

Basándote EXCLUSIVAMENTE en el siguiente contexto recuperado de documentos científicos, responde a la consulta del usuario de forma clara, precisa y estructurada.

CONSULTA DEL USUARIO:
{query_input}

CONTEXTO CIENTÍFICO RECUPERADO:
{context_text}

INSTRUCCIONES:
1. Responde SOLO con información presente en el contexto
2. Si el contexto contiene información sobre m/z y RT, identifica el compuesto
3. Menciona las bioactividades encontradas con sus valores (IC50, EC50, etc.)
4. Cita la fuente cuando sea relevante
5. Si la información es insuficiente, indícalo claramente
6. Usa formato markdown para mejor legibilidad

RESPUESTA:"""

                            # Llamar a OpenAI para generar respuesta
                            from openai import OpenAI
                            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                            
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "Eres un experto en metabolómica que genera respuestas precisas basadas en evidencia científica."},
                                    {"role": "user", "content": generation_prompt}
                                ],
                                temperature=0.3,
                                max_tokens=1000
                            )
                            
                            generated_answer = response.choices[0].message.content
                            
                            # Mostrar respuesta generada con markdown nativo de Streamlit
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                                        padding: 2rem; 
                                        border-radius: 12px; 
                                        border-left: 4px solid #10b981;
                                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                                        margin-bottom: 1rem;">
                                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                                    <span style="font-size: 2rem;">🤖</span>
                                    <span style="font-size: 1.25rem; font-weight: 600; color: #059669;">
                                        Respuesta del Asistente IA
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Renderizar el markdown generado por el LLM
                            st.markdown(generated_answer)
                            
                            # Sección de chunks usados
                            with st.expander("📚 Ver chunks utilizados para generar esta respuesta", expanded=False):
                                st.markdown(f"**Total de chunks:** {len(context_list)}")
                                for i, chunk in enumerate(context_list):
                                    st.markdown(f"""
                                    **Chunk {i+1}** (Score: {chunk.rerank_score:.4f})  
                                    `{chunk.chunk_id}`
                                    """)
                                    with st.container():
                                        st.text(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)
                            
                            # Botón para copiar respuesta
                            st.markdown("<br>", unsafe_allow_html=True)
                            col1, col2, col3 = st.columns([1, 1, 3])
                            with col1:
                                if st.button("📋 Copiar Respuesta", use_container_width=True):
                                    st.code(generated_answer, language="markdown")
                            with col2:
                                if st.button("🔄 Regenerar", use_container_width=True):
                                    st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ Error al generar respuesta: {e}")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.markdown(f"""
                <div class="warning-box">
                    <strong>❌ Error al procesar consulta:</strong><br>
                    {str(e)}
                </div>
                """, unsafe_allow_html=True)
                st.exception(e)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
        <strong>🧬 RAG Bio-Actives</strong> | Desarrollado con ❤️ usando Streamlit<br>
        Arquitectura del profesor 100% preservada | Patrones: Factory, Adapter, Strategy, Decorator
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()