#!/usr/bin/env python3
"""
Script de verificación del setup para RAG Production
Verifica que todos los componentes estén correctamente instalados y configurados
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_check(item, status, message=""):
    symbols = {"ok": "✓", "error": "✗", "warning": "⚠"}
    symbol = symbols.get(status, "?")
    status_text = status.upper()
    
    color_codes = {
        "ok": "\033[92m",      # Green
        "error": "\033[91m",   # Red
        "warning": "\033[93m"  # Yellow
    }
    reset_code = "\033[0m"
    
    color = color_codes.get(status, "")
    print(f"{color}{symbol}{reset_code} {item:.<50} {status_text}")
    if message:
        print(f"   └─> {message}")

def check_python_version():
    """Verifica versión de Python"""
    version = sys.version_info
    if version >= (3, 10):
        print_check("Python 3.10+", "ok", f"Versión: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_check("Python 3.10+", "error", f"Versión actual: {version.major}.{version.minor}. Actualiza Python")
        return False

def check_package(package_name, import_name=None):
    """Verifica si un paquete está instalado"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print_check(package_name, "ok")
        return True
    except ImportError:
        print_check(package_name, "error", f"Instala con: pip install {package_name}")
        return False

def check_env_var(var_name, required=True):
    """Verifica variable de entorno"""
    value = os.getenv(var_name)
    if value:
        # Ocultar parte de la API key por seguridad
        if "KEY" in var_name or "TOKEN" in var_name:
            display_value = value[:10] + "..." + value[-4:] if len(value) > 14 else "***"
        else:
            display_value = value
        print_check(var_name, "ok", display_value)
        return True
    else:
        status = "error" if required else "warning"
        message = "REQUERIDA" if required else "OPCIONAL (usando default)"
        print_check(var_name, status, message)
        return not required

def check_qdrant_connection():
    """Verifica conexión a Qdrant Local"""
    try:
        from qdrant_client import QdrantClient
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if api_key:
            client = QdrantClient(url=url, api_key=api_key)
        else:
            client = QdrantClient(url=url)
        
        # Intenta hacer una operación simple
        client.get_collections()
        print_check("Qdrant Local Connection", "ok", f"Conectado a {url}")
        return True
    except Exception as e:
        print_check("Qdrant Local Connection", "error", f"No se pudo conectar: {str(e)[:50]}")
        print("   Asegúrate de tener Qdrant corriendo: docker-compose up -d")
        return False

def check_file_exists(filepath, required=True):
    """Verifica si un archivo existe"""
    if Path(filepath).exists():
        print_check(filepath, "ok")
        return True
    else:
        status = "error" if required else "warning"
        message = "Archivo no encontrado"
        print_check(filepath, status, message)
        return not required

def check_openai_api():
    """Verifica API de OpenAI"""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print_check("OpenAI API", "error", "API key no configurada")
            return False
        
        client = OpenAI(api_key=api_key)
        # Intenta hacer una llamada muy simple
        # (comentado para no gastar créditos en verificación)
        # response = client.models.list()
        print_check("OpenAI API", "ok", "API key configurada (no verificada)")
        return True
    except Exception as e:
        print_check("OpenAI API", "error", str(e)[:50])
        return False

def main():
    print_header("RAG PRODUCTION - Verificación de Setup")
    
    all_checks = []
    
    # 1. Python version
    print("\n📦 PYTHON")
    all_checks.append(check_python_version())
    
    # 2. Dependencias
    print("\n📚 DEPENDENCIAS")
    packages = [
        ("PyPDF2", "PyPDF2"),
        ("pydantic", "pydantic"),
        ("numpy", "numpy"),
        ("openai", "openai"),
        ("sentence-transformers", "sentence_transformers"),
        ("qdrant-client", "qdrant_client"),
        ("rank-bm25", "rank_bm25"),
        ("langchain-text-splitters", "langchain_text_splitters"),
    ]
    
    for package, import_name in packages:
        all_checks.append(check_package(package, import_name))
    
    # 3. Variables de entorno
    print("\n🔐 VARIABLES DE ENTORNO")
    all_checks.append(check_env_var("OPENAI_API_KEY", required=True))
    all_checks.append(check_env_var("QDRANT_URL", required=False))
    all_checks.append(check_env_var("QDRANT_API_KEY", required=False))
    all_checks.append(check_env_var("PDF_PATH", required=False))
    
    # 4. Conexiones
    print("\n🌐 CONEXIONES")
    all_checks.append(check_qdrant_connection())
    all_checks.append(check_openai_api())
    
    # 5. Archivos
    print("\n📄 ARCHIVOS")
    pdf_path = os.getenv("PDF_PATH", "sample_document.pdf")
    all_checks.append(check_file_exists(pdf_path, required=False))
    all_checks.append(check_file_exists("rag_production.py", required=True))
    all_checks.append(check_file_exists("requirements_production.txt", required=True))
    
    # Resumen
    print_header("RESUMEN")
    
    total = len(all_checks)
    passed = sum(all_checks)
    failed = total - passed
    
    print(f"\nTotal de verificaciones: {total}")
    print(f"✓ Pasadas: {passed}")
    print(f"✗ Fallidas: {failed}")
    
    if failed == 0:
        print("\n🎉 ¡Todo está configurado correctamente!")
        print("   Puedes ejecutar: python rag_production.py")
        return 0
    else:
        print("\n⚠️  Hay problemas de configuración.")
        print("   Revisa los errores arriba y corrígelos antes de ejecutar.")
        print("\n📖 Consulta PRODUCTION_GUIDE.md para ayuda")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
