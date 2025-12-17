#!/usr/bin/env python3
"""
2_query_rag.py - Consultas al RAG (Terminal)
Ejecutar MUCHAS VECES para hacer consultas
"""

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

def main():
    print("=" * 70)
    print("RAG QUERY - Sistema de Consultas")
    print("=" * 70)
    
    # ============== CONFIGURACIÓN ==============
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "metabolomics_agent_db")
    
    if not OPENAI_API_KEY:
        print("\n⚠️  ERROR: OPENAI_API_KEY no configurada")
        print("   Configura en .env: OPENAI_API_KEY=tu-api-key\n")
        exit(1)
    
    print(f"🗄️  Colección: {COLLECTION_NAME}")
    print(f"🌐 Qdrant: {QDRANT_URL}")
    
    # ============== INICIALIZACIÓN (sin indexar) ==============
    print("\n1️⃣ Cargando modelos...")
    
    # LLM
    llm = OpenAILLM(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    
    # Reranker
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Embeddings
    embedding_model = SentenceTransformerAPI(model_name="all-MiniLM-L6-v2")
    
    # BM25 (necesario para queries)
    bm25_api = BM25API()
    
    # Qdrant
    db = QdrantVectorStore(
        collection_name=COLLECTION_NAME, 
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        bm25_encoder=bm25_api
    )
    
    print("   ✓ Modelos cargados")
    
    # ============== CONSTRUCCIÓN DEL PIPELINE ==============
    print("\n2️⃣ Construyendo pipeline...")
    print("   - C4: Búsqueda Híbrida")
    print("   - C6: Query Rewriting (OpenAI)")
    print("   - C7: Reranking + Context Repacking")
    
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
    
    print("   ✓ Pipeline listo")
    
    # ============== MODO INTERACTIVO ==============
    print("\n" + "=" * 70)
    print("💬 MODO INTERACTIVO")
    print("=" * 70)
    print("Escribe 'salir' para terminar\n")
    
    while True:
        try:
            query = input("\n🔍 Tu consulta: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['salir', 'exit', 'quit']:
                print("\n👋 ¡Hasta luego!")
                break
            
            print("\n" + "-" * 70)
            print("🔄 Procesando...")
            print("-" * 70)
            
            # Ejecutar query
            result = final_pipeline.retrieve_context(query=query, filters={}, k=3)
            
            # Desempaquetar resultado
            if isinstance(result, tuple):
                context_list, filter_suggestion = result
            else:
                context_list = result
                filter_suggestion = None
            
            # Mostrar resultados
            print(f"\n📊 RESULTADOS ({len(context_list)} chunks recuperados)")
            print("=" * 70)
            
            # Verificar si algún chunk tiene metadata estructurada
            has_metadata = any(c.mz_values or c.rt_values or c.compound_names or c.bioactivities for c in context_list)
            
            if not has_metadata and (filter_suggestion and (filter_suggestion.target_mz or filter_suggestion.target_rt)):
                print("\n⚠️  Nota: Los chunks recuperados no contienen metadata estructurada (m/z, RT)")
                print("   Esto puede ocurrir si:")
                print("   - Los chunks son secciones de métodos/referencias")
                print("   - El LLM no extrajo valores de esos chunks específicos")
                print("   - El documento no contiene la feature buscada\n")
            
            for i, chunk in enumerate(context_list):
                score_str = f"{chunk.rerank_score:.4f}" if chunk.rerank_score else "N/A"
                print(f"\n[{i + 1}] {chunk.chunk_id}")
                print(f"    Score: {score_str}")
                print(f"    Año: {chunk.publication_year}")
                print(f"    Fuente: {chunk.source_file}")
                
                # Mostrar metadata estructurada si existe
                if chunk.mz_values:
                    print(f"    m/z: {chunk.mz_values}")
                if chunk.rt_values:
                    print(f"    RT: {chunk.rt_values}")
                if chunk.compound_names:
                    print(f"    Compuestos: {chunk.compound_names}")
                if chunk.bioactivities:
                    print(f"    Bioactividades: {chunk.bioactivities}")
                
                print(f"    Contenido: {chunk.content[:200]}...")
            
            print("\n" + "=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrumpido por usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()