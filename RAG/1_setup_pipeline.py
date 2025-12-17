#!/usr/bin/env python3
"""
1_setup_pipeline.py - Indexación de Documentos
Soporta archivo único o carpeta con múltiples PDFs
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from rag_core import (
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
    PDF_FOLDER = os.getenv("PDF_FOLDER")  # Nuevo: carpeta con múltiples PDFs
    
    # Argumentos de línea de comandos
    import sys
    recreate = "--recreate" in sys.argv
    delete = "--delete" in sys.argv
    list_docs = "--list" in sys.argv
    clean = "--clean" in sys.argv  # Nuevo: borrar colección completa
    
    # Determinar si indexar archivo único o carpeta
    files_to_index = []
    
    if PDF_FOLDER and Path(PDF_FOLDER).is_dir():
        # Indexar todos los PDFs en la carpeta
        pdf_files = sorted(Path(PDF_FOLDER).glob("*.pdf"))
        if pdf_files:
            files_to_index = [str(f) for f in pdf_files]
            print(f"\n📁 Carpeta detectada: {PDF_FOLDER}")
            print(f"📄 PDFs encontrados: {len(files_to_index)}")
        else:
            print(f"\n⚠️  ERROR: No se encontraron PDFs en '{PDF_FOLDER}'")
            exit(1)
    elif not list_docs and not delete:
        # Indexar archivo único
        if not Path(PDF_PATH).exists():
            print(f"\n⚠️  ERROR: El archivo '{PDF_PATH}' no existe.")
            print("   Opciones:")
            print("   1. Ejecuta: python create_sample_pdf.py")
            print("   2. Archivo único: export PDF_PATH='ruta/archivo.pdf'")
            print("   3. Carpeta: export PDF_FOLDER='ruta/carpeta/'\n")
            exit(1)
        files_to_index = [PDF_PATH]
    
    print(f"\n🗄️  Colección Qdrant: {COLLECTION_NAME}")
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
    
    # ============== OPERACIONES ==============
    
    # Limpiar colección completa
    if clean:
        if db.collection_exists():
            print(f"\n⚠️  Limpiando colección '{COLLECTION_NAME}'...")
            info = db.get_collection_info()
            if info:
                print(f"   Se eliminarán {info['points_count']} chunks de {len(db.list_documents())} documentos")
            
            confirm = input("\n¿Estás seguro? (sí/no): ").strip().lower()
            if confirm in ['si', 'sí', 'yes', 'y', 's']:
                if db.delete_collection():
                    print(f"✅ Colección '{COLLECTION_NAME}' eliminada completamente")
                else:
                    print(f"❌ Error al eliminar la colección")
            else:
                print("❌ Operación cancelada")
        else:
            print(f"\n⚠️  La colección '{COLLECTION_NAME}' no existe")
        return
    
    # Listar documentos
    if list_docs:
        print(f"\n📋 Documentos en la colección '{COLLECTION_NAME}':")
        if db.collection_exists():
            info = db.get_collection_info()
            if info:
                print(f"   Total de chunks: {info['points_count']}")
            print("\n   Documentos:")
            docs = db.list_documents()
            if docs:
                for i, doc in enumerate(docs, 1):
                    print(f"   {i}. {doc}")
            else:
                print("   (vacío)")
        else:
            print("   ⚠️  La colección no existe aún")
        return
    
    # Eliminar documento específico
    if delete:
        # Mostrar documentos disponibles
        if db.collection_exists():
            docs = db.list_documents()
            if docs:
                print("\n📚 Documentos disponibles:")
                for i, doc in enumerate(docs, 1):
                    print(f"   {i}. {doc}")
        
        doc_to_delete = input("\n🗑️  Nombre del documento a eliminar (o número): ").strip()
        
        # Permitir eliminar por número
        if doc_to_delete.isdigit() and db.collection_exists():
            idx = int(doc_to_delete) - 1
            docs = db.list_documents()
            if 0 <= idx < len(docs):
                doc_to_delete = docs[idx]
        
        if doc_to_delete:
            deleted = db.delete_document(doc_to_delete)
            if deleted > 0:
                print(f"✅ Eliminados {deleted} chunks del documento '{doc_to_delete}'")
            else:
                print(f"⚠️  No se encontraron chunks del documento '{doc_to_delete}'")
        return
    
    # Verificar si colección existe
    if db.collection_exists():
        info = db.get_collection_info()
        if info:
            print(f"\n📊 Colección existente detectada:")
            print(f"   - Chunks actuales: {info['points_count']}")
            print(f"   - Documentos: {len(db.list_documents())}")
        
        if recreate:
            print("\n⚠️  Modo --recreate: Se eliminará toda la colección")
        else:
            print("\n📝 Modo incremental: Se agregarán nuevos chunks")
    else:
        print(f"\n✨ Creando nueva colección '{COLLECTION_NAME}'")
    
    # ============== INDEXACIÓN ==============
    if len(files_to_index) == 1:
        print(f"\n2️⃣ Indexando documento: {Path(files_to_index[0]).name}")
    else:
        print(f"\n2️⃣ Indexando {len(files_to_index)} documentos:")
        for i, f in enumerate(files_to_index, 1):
            print(f"   {i}. {Path(f).name}")
    
    # Indexar cada archivo
    successful_files = []
    failed_files = []
    
    for idx, file_path in enumerate(files_to_index):
        try:
            if len(files_to_index) > 1:
                print(f"\n{'='*70}")
                print(f"📄 Procesando [{idx+1}/{len(files_to_index)}]: {Path(file_path).name}")
                print(f"{'='*70}")
            
            run_indexing_service(
                file_path,
                dense_adapter,
                sparse_adapter,
                db,
                recreate_collection=(recreate and idx == 0),  # Solo recrear en el primer archivo
                chunk_size=500,
                chunk_overlap=50
            )
            
            successful_files.append(Path(file_path).name)
            if len(files_to_index) > 1:
                print(f"✅ Completado: {Path(file_path).name}")
            
        except Exception as e:
            print(f"❌ Error al procesar {Path(file_path).name}: {e}")
            failed_files.append((Path(file_path).name, str(e)))
            continue
    
    # Mostrar estadísticas finales
    info = db.get_collection_info()
    docs = db.list_documents()
    
    print("\n" + "=" * 70)
    print("✅ INDEXACIÓN COMPLETADA")
    print("=" * 70)
    
    # Resumen de procesamiento
    if len(files_to_index) > 1:
        print(f"\n📊 Resumen del batch:")
        print(f"   ✅ Exitosos: {len(successful_files)}")
        if failed_files:
            print(f"   ❌ Fallidos: {len(failed_files)}")
    
    # Estadísticas de la colección
    if info:
        print(f"\n📈 Estadísticas de la colección:")
        print(f"   - Total chunks: {info['points_count']}")
        print(f"   - Total documentos: {len(docs)}")
    
    # Documentos indexados
    print(f"\n📚 Documentos en la colección:")
    if docs:
        for i, doc in enumerate(docs, 1):
            status = "✨" if Path(doc).name in successful_files else ""
            print(f"   {i}. {doc} {status}")
    
    # Mostrar archivos fallidos
    if failed_files:
        print(f"\n❌ Archivos con errores:")
        for file, error in failed_files:
            print(f"   - {file}")
            print(f"     Error: {error[:100]}...")
    
    print(f"\n🚀 Siguiente paso:")
    print(f"   - python 2_query_rag.py (consultas en terminal)")
    print(f"   - streamlit run 3_streamlit_app.py (interfaz web)")
    
    print(f"\n💡 Comandos útiles:")
    print(f"   - python 1_setup_pipeline.py --list         (listar documentos)")
    print(f"   - python 1_setup_pipeline.py --delete       (eliminar documento)")
    print(f"   - python 1_setup_pipeline.py --clean        (borrar colección completa)")
    print(f"   - python 1_setup_pipeline.py --recreate     (recrear colección)")
    
    if len(files_to_index) > 1:
        print(f"\n📁 Para indexar otra carpeta:")
        print(f"   export PDF_FOLDER='ruta/otra/carpeta'")
        print(f"   python 1_setup_pipeline.py")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()