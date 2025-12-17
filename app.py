import os
import time
import pandas as pd
import asyncio
import streamlit as st
from dotenv import load_dotenv
import textwrap

from src.infrastructure.bootstrap import create_rag_service
from src.application.services.rag_service import VectorStoreService
from src.application.services.ingestion_service import IngestionService
from src.domain.models import ProcessedChunk
from src.infrastructure.evaluation.evaluator_service import EvaluatorService

# Imports for Benchmark
from datasets import Dataset


# Load env vars
load_dotenv()

# ==============================================================================
# CONFIGURATION & CSS
# ==============================================================================

# Constants
QDRANT_PATH = "qdrant_storage"
DATASET_PATH = "datasets/golden_dataset.csv"

# Page Config
st.set_page_config(
    page_title="RAG Bio-Actives 🧬",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-success { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }
    
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

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e2e8f0;
        color: #4c1d95;
    }

</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. INITIALIZATION & SERVICES
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_rag_service() -> VectorStoreService:
    """
    Instancia los adaptadores y el servicio de aplicación usando la nueva arquitectura src.
    """
    return create_rag_service(
        qdrant_path=QDRANT_PATH,
        qdrant_collection="metabolomics_agent_db",
        enable_pubchem=True
    )

def run_indexing_service(
    file_path: str,
    rag_service: VectorStoreService,
    overwrite: bool = False,
) -> None:
    """Delgates to the Application Service."""
    ingestion_service = IngestionService(rag_service)
    ingestion_service.run_ingestion(file_path, overwrite=overwrite)

# ==============================================================================
# 3. UI HELPERS
# ==============================================================================

def render_chunk_card_html(chunk: ProcessedChunk, index: int):
    """Renderiza una tarjeta de chunk con diseño CSS personalizado"""
    # Manejo seguro de score
    raw_score = 0.0
    if chunk.metadata and "score" in chunk.metadata:
        raw_score = chunk.metadata["score"]
    elif chunk.rerank_score:
        raw_score = chunk.rerank_score
        
    score_str = f"{raw_score:.4f}"
    
    # Normalizar para visualización (heurística simple)
    # scores typical range: -10 to +10 (reranker) or 0 to 1 (cosine)
    # let's assume reranker scores here
    normalized_score = max(0, min(100, (raw_score + 5) * 6.67)) if raw_score > -10 else raw_score * 100
    
    pub_year = chunk.metadata.get("publication_year", "N/A") if chunk.metadata else "N/A"
    src_file = chunk.source_file or "Desconocido"
    
    # Process Content (JSON handling)
    import json
    content_display = chunk.content
    
    # Check if it is a JSON chunk (either by type or content heuristic)
    is_json = False
    try:
        # Explicit type check from PDFLoader
        if chunk.type == "table_row_json" or chunk.content.strip().startswith("{"):
            data = json.loads(chunk.content)
            if isinstance(data, dict):
                is_json = True
                # Convert dict to simple HTML table
                # Ensure keys and values are strings and escaped if necessary (basic HTML safety)
                rows = ""
                for k, v in data.items():
                    rows += f"<tr><td style='font-weight:600; padding:4px; border-bottom:1px solid #eee;'>{k}</td><td style='padding:4px; border-bottom:1px solid #eee;'>{v}</td></tr>"
                
                content_display = f"<table style='width:100%; border-collapse:collapse; font-size:0.9rem;'>{rows}</table>"
            elif isinstance(data, list):
                 # Handle list of dicts if that ever happens (e.g. chunks of length > 1)
                 is_json = True
                 content_display = f"<pre style='white-space: pre-wrap; font-family: monospace;'>{json.dumps(data, indent=2)}</pre>"
    except json.JSONDecodeError:
        # If it was supposed to be JSON but failed, show raw with warning
        if chunk.type == "table_row_json":
             content_display = f"<div style='color:red;'>⚠️ Invalid JSON Content</div><pre>{chunk.content}</pre>"
        pass
    except Exception as e:
        pass
    
    return textwrap.dedent(f"""
    <div class="chunk-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
            <div>
                <h3 style="margin: 0; color: #1f2937; font-size: 1.25rem;">📄 Chunk {index + 1}</h3>
                <p style="margin: 0.25rem 0 0 0; color: #6b7280; font-size: 0.875rem;"><code>{chunk.chunk_id or 'ID'}</code></p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #667eea;">{score_str}</div>
                <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">Score</div>
            </div>
        </div>
        <div class="score-bar">
            <div class="score-fill" style="width: {normalized_score}%;"></div>
        </div>
        <div style="display: flex; gap: 0.5rem; margin: 1rem 0;">
            <span class="badge badge-success">📅 {pub_year}</span>
            <span class="badge" style="background:#e0f2fe; color:#0369a1;">{chunk.type or 'text'}</span>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <div style="font-size: 0.875rem; color: #4b5563; margin-bottom: 0.5rem; font-weight: 600;">📝 Contenido:</div>
            <div style="color: #1f2937; line-height: 1.6; max-height: 400px; overflow-y: auto; overflow-x: auto;">
                {content_display}
            </div>
        </div>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
            <div style="font-size: 0.875rem; color: #6b7280;">📂 Fuente: <code>{src_file}</code></div>
        </div>
    </div>
    """)




# ==============================================================================
# 4. MAIN APP LOGIC
# ==============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 RAG Bio-Actives</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema Inteligente de Anotación de Compuestos Metabolómicos (v2.0 Clean Arch)</p>', unsafe_allow_html=True)

    # Initialize Service
    with st.spinner("🔄 Inicializando sistema RAG..."):
        try:
            rag_service = get_rag_service()
        except Exception as e:
            st.error(f"❌ Error al inicializar: {e}")
            st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Gestión & Config")
        
        # Pipeline Configuration (New)
        with st.expander("🛠️ Estrategia RAG", expanded=True):
            st.markdown("**1. Pre-Processing**")
            use_rewriting = st.checkbox("Enable Query Rewriting", value=True)
            
            st.markdown("**2. Retrieval**")
            retrieval_mode = st.selectbox(
                "Base Strategy", 
                ["Hybrid (Dense+Sparse)", "Semantic Only (Dense)"],
                index=0
            )
            use_pubchem = st.checkbox("Include PubChem", value=True)
            
            st.markdown("**3. Post-Processing**")
            use_reranking = st.checkbox("Enable Reranking", value=True)
            use_repacking = st.checkbox("Enable Repacking", value=True)

        st.divider()
        k_results = st.slider("📊 Top-K Chunks", 1, 15, 5)
        
        # Indexer
        data_folder = "data"
        if st.toggle("📂 Mostrar Indexador"):
            st.info(f"Storage: `{QDRANT_PATH}/`")
            if st.button("🔄 Indexar Data (Sobreescribir)"):
                if not os.path.exists(data_folder):
                    st.error(f"Falta carpeta {data_folder}")
                else:
                    files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
                    bar = st.progress(0)
                    for i, f in enumerate(files):
                        run_indexing_service(os.path.join(data_folder, f), rag_service, overwrite=(i==0)) 
                        bar.progress((i+1)/len(files))
                    st.success("¡Indexado Completo!")
                    time.sleep(1)
                    st.rerun()

        st.markdown("### 💡 Ejemplos")
        example_queries = {
            "🔬 Feature m/z + RT": "Feature m/z 495.1285 rt 5.99 in cocoa powder",
            "💊 Antioxidante": "¿Qué compuestos tienen actividad antioxidante?",
            "🧪 Método LC-MS": "Features LC-MS antidiabéticas",
        }
        for label, q in example_queries.items():
            if st.button(label, use_container_width=True):
                st.session_state.example_query = q
                st.rerun()

    # Tabs
    tab_search, tab_eval = st.tabs(["🔎 Búsqueda Semántica", "📊 Benchmark"])

    # --- TAB SEARCH ---
    with tab_search:
        st.markdown("### 🔍 Tu Consulta")
        query_input = st.text_area(
            "Query",
            value=st.session_state.get('example_query', ''),
            height=100,
            placeholder="Escribe tu consulta aquí...",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_btn = st.button("🚀 Buscar", type="primary", use_container_width=True)
            
        if search_btn and query_input:
            # Container for process steps
            process_container = st.container()
            results_container = st.container()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Internal context holder
            current_query = query_input
            current_filters = {}
            current_candidates = []
            
            from src.domain.models import FilterSuggestion
            
            with process_container:
                st.markdown("#### ⚙️ Pipeline Execution Log")
                
                # --- STEP 1: QUERY REWRITING ---
                if use_rewriting:
                    status_text.text("1️⃣ Rewriting Query & Extracting Filters...")
                    progress_bar.progress(10)
                    
                    with st.expander("1️⃣ Query Optimization", expanded=True):
                        start_time = time.time()
                        suggestion = rag_service.query_processor.process_query(query_input)
                        
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.info(f"**Rewritten Query:** {suggestion.rewritten_query}")
                        with c2:
                            st.json(suggestion.metadata_filters, expanded=True)
                        st.caption(f"⏱️ Time: {time.time() - start_time:.2f}s")
                        
                        current_query = suggestion.rewritten_query
                        current_filters = suggestion.metadata_filters
                else:
                    st.info("ℹ️ Rewriting skipped (using raw query).")
                    current_query = query_input
                    current_filters = {}

                # --- STEP 2: RETRIEVAL ---
                mode_key = "dense" if "Semantic Only" in retrieval_mode else "hybrid"
                strat_name = f"Federated ({mode_key} + PubChem)" if use_pubchem else f"{mode_key.capitalize()} Search"
                
                status_text.text(f"2️⃣ Executing Retrieval: {strat_name}...")
                progress_bar.progress(30)
                
                with st.expander(f"2️⃣ Retrieval ({strat_name})", expanded=True):
                    start_time = time.time()
                    
                    # Factory Strategy
                    retriever = rag_service.get_retrieval_strategy(mode=mode_key, use_pubchem=use_pubchem)
                    
                    # Candidates to fetch (fetch more if reranking is enabled)
                    fetch_k = k_results * 3 if use_reranking else k_results
                    
                    candidates = retriever.retrieve_context(
                        current_query, 
                        filters=current_filters, 
                        top_k=fetch_k
                    )
                    
                    st.write(f"Found **{len(candidates)}** candidates.")
                    if candidates:
                        for i, c in enumerate(candidates[:3]):
                            source_icon = "🧪" if c.metadata.get("source") == "PubChem" else "📄"
                            src_name = c.source_file or c.metadata.get('source')
                            st.text(f"[{i+1}] {source_icon} {c.chunk_id or 'No ID'} | {src_name}")
                    
                    st.caption(f"⏱️ Time: {time.time() - start_time:.2f}s")
                    current_candidates = candidates
                
                if not current_candidates:
                    st.warning("⚠️ No candidates found. Stopping pipeline.")
                    progress_bar.progress(100)
                    st.stop()

                # --- STEP 3: RERANKING ---
                if use_reranking:
                    status_text.text("3️⃣ Semantic Reranking...")
                    progress_bar.progress(60)
                    
                    with st.expander("3️⃣ Cross-Encoder Reranking", expanded=True):
                        start_time = time.time()
                        
                        # Debug: Show sources BEFORE reranking
                        sources_before = [c.metadata.get("source", "internal") if c.metadata else "internal" for c in current_candidates]
                        pubchem_count_before = sum(1 for s in sources_before if s == "PubChem")
                        print(f"[APP] BEFORE Reranking: {len(current_candidates)} candidates, {pubchem_count_before} from PubChem")
                        
                        texts = [c.content for c in current_candidates]
                        scores = rag_service.reranker_service.rerank(current_query, texts)
                        
                        scored_candidates = []
                        for chunk, score in zip(current_candidates, scores):
                            chunk.rerank_score = float(score)
                            chunk.metadata["score"] = float(score)
                            scored_candidates.append(chunk)
                        
                        scored_candidates.sort(key=lambda x: x.rerank_score, reverse=True)
                        
                        # Force-include PubChem chunks if present (they were specifically requested via m/z filter)
                        pubchem_chunks = [c for c in scored_candidates if c.metadata and c.metadata.get("source") == "PubChem"]
                        non_pubchem_chunks = [c for c in scored_candidates if not (c.metadata and c.metadata.get("source") == "PubChem")]
                        
                        print(f"[APP] Force-include check: Found {len(pubchem_chunks)} PubChem chunks in scored_candidates")
                        if pubchem_chunks:
                            print(f"[APP] PubChem chunk metadata: {pubchem_chunks[0].metadata}")
                        
                        if pubchem_chunks:
                            # Reserve 1 slot for PubChem, fill rest with top non-PubChem chunks
                            final_selection = pubchem_chunks[:1] + non_pubchem_chunks[:k_results-1]
                            st.info(f"ℹ️ Force-included {len(pubchem_chunks[:1])} PubChem chunk(s) (m/z filter active)")
                        else:
                            # No PubChem chunks, just take top k
                            final_selection = scored_candidates[:k_results]
                        
                        # Debug: Show sources AFTER reranking
                        sources_after = [(c.metadata.get("source", "internal") if c.metadata else "internal", c.rerank_score) for c in final_selection]
                        pubchem_in_final = sum(1 for s, _ in sources_after if s == "PubChem")
                        print(f"[APP] AFTER Reranking (top {k_results}): {pubchem_in_final} from PubChem")
                        print(f"[APP] Top {k_results} sources: {sources_after}")
                        
                        # Viz
                        score_data = {"Chunk": [c.chunk_id or "ext" for c in final_selection], "Score": [c.rerank_score for c in final_selection]}
                        st.bar_chart(pd.DataFrame(score_data).set_index("Chunk"), height=200)
                        
                        st.caption(f"⏱️ Time: {time.time() - start_time:.2f}s")
                        current_candidates = final_selection
                else:
                    st.info("ℹ️ Reranking skipped.")
                    # If no rerank, take top k (already sorted by vector db score usually)
                    current_candidates = current_candidates[:k_results]

                # --- STEP 4: REPACKING ---
                if use_repacking:
                    status_text.text("4️⃣ Context Repacking...")
                    progress_bar.progress(80)
                    
                    with st.expander("4️⃣ Context Repacking", expanded=False):
                        chunks = current_candidates
                        repacked = [None] * len(chunks)
                        left, right = 0, len(chunks) - 1
                        for i, chunk in enumerate(chunks):
                            if i % 2 == 0:
                                repacked[left] = chunk
                                left += 1
                            else:
                                repacked[right] = chunk
                                right -= 1
                        final_context = repacked
                        st.success("Repacking applied.")
                else:
                    st.info("ℹ️ Repacking skipped.")
                    final_context = current_candidates

                status_text.text("✅ Pipeline Complete")
                progress_bar.progress(100)
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

            # --- DISPLAY RESULTS ---
            with results_container:
                st.divider()
                st.subheader("🏁 Final Results")
                
                # Show naturally sorted by relevance (score if available)
                # If reranking was used, they have rerank_score. 
                display_chunks = sorted(final_context, key=lambda x: x.rerank_score if x.rerank_score else 0, reverse=True)
                
                # Metrics
                if display_chunks:
                    top_score = display_chunks[0].rerank_score if display_chunks[0].rerank_score else 0
                    c1, c2 = st.columns(2)
                    c1.metric("Chunks", len(display_chunks))
                    c2.metric("Top Score", f"{top_score:.4f}" if top_score else "N/A")
                
                for i, chunk in enumerate(display_chunks):
                    html = render_chunk_card_html(chunk, i)
                    st.markdown(html, unsafe_allow_html=True)
                
                st.divider()
                
                # --- AUTOMATIC GENERATION ---
                st.markdown("### 🤖 Synthesizing Answer...")
                progress_bar_gen = st.progress(0)
                
                with st.spinner("Generating answer with LLM..."):
                     start_gen = time.time()
                     answer = rag_service.generator.generate_answer(current_query, final_context)
                     
                     st.markdown("#### 📝 Final Answer")
                     st.success(answer)
                     
                     st.caption(f"⏱️ Generation Time: {time.time() - start_gen:.2f}s")
                     progress_bar_gen.progress(100)
                     time.sleep(0.5)
                     progress_bar_gen.empty()

    # --- TAB EVAL ---
    with tab_eval:
        st.subheader("Evaluación Ragas (Dataset Golden)")
        if st.button("📉 Ejecutar Benchmark"):

            evaluator = EvaluatorService(rag_service)
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            def update_progress(current, total, msg):
                status_text.text(msg)
                progress_bar.progress(current / total)

            with st.spinner("Evaluando... (Esto toma tiempo)"):
                df_res, metrics = evaluator.run_benchmark(DATASET_PATH, progress_callback=update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            if isinstance(df_res, str):
                st.error(df_res)
            else:
                st.success("Evaluación finalizada.")
                # Show metrics
                c1, c2, c3, c4 = st.columns(4)
                p = df_res.get("context_precision", pd.Series([0])).mean()
                r = df_res.get("context_recall", pd.Series([0])).mean()
                ac = df_res.get("answer_correctness", pd.Series([0])).mean()
                f = df_res.get("faithfulness", pd.Series([0])).mean()
                
                c1.metric("Retrieval Precision", f"{p:.4f}")
                c2.metric("Retrieval Recall", f"{r:.4f}")
                c3.metric("Answer Correctness", f"{ac:.4f}")
                c4.metric("Faithfulness", f"{f:.4f}")
                
                st.dataframe(df_res)


if __name__ == "__main__":
    main()