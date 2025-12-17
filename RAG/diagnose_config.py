#!/usr/bin/env python3
"""
diagnose_config.py - Diagnosticar configuración del sistema
"""

import os
from pathlib import Path
from dotenv import load_dotenv

print("=" * 70)
print("🔍 DIAGNÓSTICO DE CONFIGURACIÓN")
print("=" * 70)

# Cargar .env
load_dotenv()

print("\n1️⃣ Variables de Entorno:")
print("-" * 70)

# OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✅ OPENAI_API_KEY: {'*' * 20}{openai_key[-4:]}")
else:
    print(f"❌ OPENAI_API_KEY: No configurado")

# Qdrant
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
print(f"✅ QDRANT_URL: {qdrant_url}")

# Colección
collection = os.getenv("COLLECTION_NAME", "metabolomics_agent_db")
print(f"✅ COLLECTION_NAME: {collection}")

print("\n2️⃣ Configuración de PDFs:")
print("-" * 70)

# PDF_PATH
pdf_path = os.getenv("PDF_PATH", "bioactives_sample.pdf")
print(f"📄 PDF_PATH: {pdf_path}")
if Path(pdf_path).exists():
    print(f"   ✅ Archivo existe")
    print(f"   📊 Tamaño: {Path(pdf_path).stat().st_size / 1024:.2f} KB")
else:
    print(f"   ❌ Archivo NO existe")

# PDF_FOLDER
pdf_folder = os.getenv("PDF_FOLDER")
print(f"\n📁 PDF_FOLDER: {pdf_folder if pdf_folder else '(no configurado)'}")

if pdf_folder:
    folder_path = Path(pdf_folder)
    if folder_path.exists():
        if folder_path.is_dir():
            print(f"   ✅ Directorio existe")
            pdf_files = list(folder_path.glob("*.pdf"))
            print(f"   📄 PDFs encontrados: {len(pdf_files)}")
            
            if pdf_files:
                print(f"\n   Archivos:")
                for i, pdf in enumerate(pdf_files[:10], 1):
                    size_kb = pdf.stat().st_size / 1024
                    print(f"   {i}. {pdf.name} ({size_kb:.2f} KB)")
                
                if len(pdf_files) > 10:
                    print(f"   ... y {len(pdf_files) - 10} más")
            else:
                print(f"   ⚠️  No se encontraron archivos PDF")
        else:
            print(f"   ❌ La ruta existe pero NO es un directorio")
    else:
        print(f"   ❌ Directorio NO existe")
        print(f"   💡 Ruta absoluta: {folder_path.absolute()}")
else:
    print(f"   ℹ️  No configurado (se usará PDF_PATH)")

print("\n3️⃣ Prioridad de Indexación:")
print("-" * 70)

if pdf_folder and Path(pdf_folder).is_dir():
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    if pdf_files:
        print(f"🎯 Se indexará: CARPETA ({pdf_folder})")
        print(f"   📄 {len(pdf_files)} archivos PDF")
    else:
        print(f"⚠️  Carpeta configurada pero sin PDFs")
        print(f"   Se intentará usar: {pdf_path}")
elif Path(pdf_path).exists():
    print(f"🎯 Se indexará: ARCHIVO ({pdf_path})")
else:
    print(f"❌ ERROR: No hay archivos válidos para indexar")

print("\n4️⃣ Verificación de Qdrant:")
print("-" * 70)

try:
    import requests
    response = requests.get(f"{qdrant_url}/collections")
    if response.status_code == 200:
        print(f"✅ Qdrant está accesible en {qdrant_url}")
        collections = response.json().get('result', {}).get('collections', [])
        print(f"   📦 Colecciones existentes: {len(collections)}")
        
        # Verificar si existe nuestra colección
        our_collection = next((c for c in collections if c['name'] == collection), None)
        if our_collection:
            print(f"   ✅ Colección '{collection}' existe")
        else:
            print(f"   ℹ️  Colección '{collection}' no existe (se creará al indexar)")
    else:
        print(f"❌ Qdrant responde pero con error: {response.status_code}")
except Exception as e:
    print(f"❌ No se puede conectar a Qdrant: {e}")
    print(f"   💡 ¿Está corriendo Docker? Ejecuta: docker-compose up -d")

print("\n5️⃣ Recomendaciones:")
print("-" * 70)

# Verificar .env
if not Path(".env").exists():
    print("⚠️  Archivo .env no existe")
    print("   💡 Ejecuta: cp .env.example .env")
else:
    print("✅ Archivo .env existe")

# Verificar si PDF_FOLDER está mal configurado
if pdf_folder and not Path(pdf_folder).exists():
    print(f"\n⚠️  PDF_FOLDER='{pdf_folder}' no existe")
    print("   Opciones:")
    print(f"   1. Crear carpeta: mkdir -p {pdf_folder}")
    print(f"   2. Cambiar ruta en .env a una carpeta existente")
    print(f"   3. Comentar PDF_FOLDER y usar PDF_PATH")
    print("\n   Ejemplo de configuración correcta:")
    print("   # En .env:")
    print("   PDF_FOLDER=./mis_documentos")
    print("   # Luego:")
    print("   mkdir -p ./mis_documentos")
    print("   cp *.pdf ./mis_documentos/")

print("\n" + "=" * 70)
print("✅ DIAGNÓSTICO COMPLETADO")
print("=" * 70)