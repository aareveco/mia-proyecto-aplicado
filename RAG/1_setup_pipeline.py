#!/usr/bin/env python3
"""
1_setup_pipeline.py - Indexación de Documentos
Ejecutar UNA VEZ o cuando se agregan nuevos documentos
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from rag_core import (
    OpenAILLM,
    CrossEncoderReranker,
    SentenceTransformerAPI,
    QdrantVectorStore,
    BioBERTAdapter,
    BM25Adapter,
    BM25API,
    run_indexing_service
)

def main():
    print("=" * 70)
    print("SETUP PIPELINE - Indexación de Documentos")
    print("=" * 70)
    
    # ============== CONFIGURACIÓN ==============
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "metabolomics_agent_db")
    PDF_PATH = os.getenv("PDF_PATH", "bioactives_sample.pdf")
    
    # Verificar PDF
    if not Path(PDF_PATH).exists():
        print(f"\n⚠️  ERROR: El archivo '{PDF_PATH}' no existe.")
        print("   1. Ejecuta: python create_sample_pdf.py")
        print("   2. O configura: export PDF_PATH='ruta/a/tu/pdf.pdf'\n")
        exit(1)
    
    print(f"\n📄 Documento a indexar: {PDF_PATH}")
    print(f"🗄️  Colección Qdrant: {COLLECTION_NAME}")
    print(f"🌐 URL Qdrant: {QDRANT_URL}")
    
    # ============== INICIALIZACIÓN ==============
    print("\n1️⃣ Inicializando componentes...")
    
    # Embeddings
    embedding_model = SentenceTransformerAPI(model_name="all-MiniLM-L6-v2")
    
    # BM25 compartido
    bm25_api = BM25API()
    
    # Adapters
    dense_adapter = BioBERTAdapter(embedding_model)
    sparse_adapter = BM25Adapter(bm25_api)
    
    # Qdrant
    db = QdrantVectorStore(
        collection_name=COLLECTION_NAME, 
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        bm25_encoder=bm25_api
    )
    
    print("   ✓ Componentes inicializados")
    
    # ============== INDEXACIÓN ==============
    print(f"\n2️⃣ Indexando documentos...")
    
    run_indexing_service(
        PDF_PATH,
        dense_adapter,
        sparse_adapter,
        db,
        chunk_size=500,
        chunk_overlap=50
    )
    
    print("\n" + "=" * 70)
    print("✅ INDEXACIÓN COMPLETADA")
    print("=" * 70)
    print(f"\n📊 Colección '{COLLECTION_NAME}' lista para consultas")
    print(f"🚀 Ahora puedes ejecutar:")
    print(f"   - python 2_query_rag.py (terminal)")
    print(f"   - streamlit run 3_streamlit_app.py (interfaz web)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
