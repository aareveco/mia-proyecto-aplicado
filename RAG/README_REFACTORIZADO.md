# 🧬 RAG Bio-Actives - Estructura Modular

Sistema RAG para anotación de compuestos metabolómicos con arquitectura separada en 3 archivos.

## 📁 Estructura del Proyecto

```
RAG/
├── rag_core.py              # ⚙️ Clases y funciones compartidas
├── 1_setup_pipeline.py      # 🔧 Indexación (ejecutar UNA VEZ)
├── 2_query_rag.py           # 💬 Consultas en terminal (ejecutar MUCHAS VECES)
├── 3_streamlit_app.py       # 🌐 Interfaz web interactiva
├── bioactives_sample.pdf    # 📄 PDF de ejemplo
├── requirements_production.txt
└── .env
```

---

## 🚀 Quick Start

### 1️⃣ Instalar Dependencias

```bash
pip install -r requirements_production.txt
```

### 2️⃣ Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env y agregar:
# OPENAI_API_KEY=sk-proj-xxxxx...
# QDRANT_URL=https://tu-qdrant-cloud.io:6333  (o http://localhost:6333)
# QDRANT_API_KEY=tu-api-key  (si usas Qdrant Cloud)
```

### 3️⃣ Generar PDF de Ejemplo

```bash
python create_sample_pdf.py
# Genera: bioactives_sample.pdf
```

### 4️⃣ Indexar Documentos (UNA VEZ)

```bash
python 1_setup_pipeline.py
```

**Salida esperada:**
```
======================================================================
SETUP PIPELINE - Indexación de Documentos
======================================================================
📄 Documento a indexar: bioactives_sample.pdf
🗄️  Colección Qdrant: metabolomics_agent_db

1️⃣ Inicializando componentes...
   ✓ Componentes inicializados

2️⃣ Indexando documentos...
-> [C2/PDFLoader]: Aplicando Layout-Aware Chunking a bioactives_sample.pdf
   -> Extraídos 5 chunks de bioactives_sample.pdf
[C4/Qdrant]: Indexados 5 chunks en colección 'metabolomics_agent_db'

======================================================================
✅ INDEXACIÓN COMPLETADA
======================================================================
```

**Tiempo:** ~30 segundos (primera vez, descarga modelos ~2GB)

---

## 💬 Opción A: Consultas en Terminal

```bash
python 2_query_rag.py
```

**Modo interactivo:**
```
🔍 Tu consulta: Tengo una feature con m/z 449.107, RT 8.2 min en Té Verde. ¿Qué es?

🔄 Procesando...
----------------------------------------------------------------------
-> [C6/OpenAI]: Query reescrita...
-> [C4/Strategy]: Búsqueda HÍBRIDA...
-> [C7/Cross-Encoder]: Rerankendo 5 chunks...

📊 RESULTADOS (3 chunks recuperados)
======================================================================

[1] bioactives_sample-chunk-3
    Score: 5.4535
    Método: LC-MS
    Contenido: Myricetina-derivado detectado en Arándano 004...

[2] bioactives_sample-chunk-1
    Score: 3.9025
    Método: LC-MS
    Contenido: Feature m/z 449.107, RT 8.2 min...
```

**Tiempo por consulta:** ~2 segundos

---

## 🌐 Opción B: Interfaz Web (Streamlit)

```bash
streamlit run 3_streamlit_app.py
```

**Se abre en el navegador:** `http://localhost:8501`

### 🎨 Características de la Interfaz:

1. **Query Input**
   - Campo de texto para consultas
   - Botones de ejemplo pre-cargados

2. **Visualización del Proceso** (expandible):
   - ✅ Query Original
   - ✅ Query Rewriting (C6 - OpenAI)
   - ✅ Búsqueda Híbrida (C4 - Qdrant)
   - ✅ Reranking (C7 - Cross-Encoder) con gráfico de scores
   - ✅ Context Repacking (C7)

3. **Resultados**:
   - Métricas: Chunks recuperados, Score promedio, Métodos detectados
   - Cards por cada chunk con:
     - Chunk ID
     - Score de reranking
     - Metadata (método, año, fuente)
     - Contenido completo

4. **Sidebar**:
   - Configuración de parámetros (k results)
   - Toggle para mostrar/ocultar proceso
   - Información del pipeline (C2-C7)
   - Ejemplos de consultas

---

## 📊 Arquitectura del Código

### ⚙️ `rag_core.py` - Funciones Compartidas

**Clases principales:**
- `OpenAILLM` - LLM para query rewriting
- `CrossEncoderReranker` - Reranking semántico
- `SentenceTransformerAPI` - Embeddings densos
- `BM25API` - Sparse vectors
- `QdrantVectorStore` - Base de datos vectorial
- `PDFLoader` - Carga y chunking de PDFs
- Strategies: `HybridSearchStrategy`, `QueryRewritingStrategy`
- Decorators: `RerankingDecorator`, `ContextRepackerDecorator`

**Funciones de utilidad:**
- `run_embedding_pipeline()` - Genera embeddings
- `run_indexing_service()` - Indexa documentos

### 🔧 `1_setup_pipeline.py` - Indexación

**Ejecutar:** Una vez o cuando agregas nuevos PDFs

**Flujo:**
1. Carga modelos (embeddings, BM25)
2. Conecta a Qdrant
3. Procesa PDF → chunks
4. Genera embeddings (dense + sparse)
5. Indexa en Qdrant con índices

### 💬 `2_query_rag.py` - Consultas Terminal

**Ejecutar:** Muchas veces para consultas rápidas

**Flujo:**
1. Carga modelos (ya cacheados, rápido)
2. Construye pipeline (C4 + C6 + C7)
3. Modo interactivo: lee queries del usuario
4. Muestra resultados en consola

### 🌐 `3_streamlit_app.py` - Interfaz Web

**Ejecutar:** Para interfaz visual interactiva

**Flujo:**
1. Carga modelos con `@st.cache_resource` (una sola vez)
2. Construye pipeline
3. Interfaz visual con:
   - Input de query
   - Visualización del proceso paso a paso
   - Resultados con cards y métricas

---

## 🔄 Workflow Típico

### Día 1: Setup Inicial
```bash
# 1. Instalar
pip install -r requirements_production.txt

# 2. Configurar .env
cp .env.example .env
# Agregar OPENAI_API_KEY

# 3. Generar PDF
python create_sample_pdf.py

# 4. Indexar (UNA VEZ)
python 1_setup_pipeline.py
```

### Día 2+: Solo Consultas
```bash
# Opción A: Terminal
python 2_query_rag.py

# Opción B: Web
streamlit run 3_streamlit_app.py
```

### Agregar Más PDFs:
```bash
# Editar PDF_PATH en .env o pasar como argumento
export PDF_PATH="nuevo_documento.pdf"
python 1_setup_pipeline.py
```

---

## 🎯 Ventajas de Esta Estructura

### ✅ Separación de Responsabilidades
- **Setup:** Se ejecuta una vez (lento, ~30s)
- **Query:** Se ejecuta muchas veces (rápido, ~2s)

### ✅ Reutilización de Código
- `rag_core.py` compartido por todos los scripts
- Sin duplicación de código

### ✅ Flexibilidad
- Terminal para testing rápido
- Web para demos y producción

### ✅ Mantenibilidad
- Cambios en `rag_core.py` se reflejan en todos
- Fácil agregar nuevas interfaces

### ✅ Performance
- Modelos cargados una sola vez
- Sin re-indexar en cada query
- Streamlit cachea recursos

---

## 📝 Casos de Uso

### Desarrollo / Testing
```bash
python 2_query_rag.py
# Rápido, sin GUI, perfecto para debugging
```

### Demos / Presentaciones
```bash
streamlit run 3_streamlit_app.py
# Visual, interactivo, muestra el proceso
```

### Producción / API
```python
from rag_core import *

# Usar las clases directamente en tu aplicación
pipeline = build_pipeline()
results = pipeline.retrieve_context(query, filters, k)
```

---

## 🧪 Ejemplos de Consultas

### Feature Identification
```
Tengo una feature con m/z 449.107, RT 8.2 min, detectada en mi muestra de Té Verde. ¿Qué es y qué bioactividad tiene?
```

### Bioactivity Search
```
¿Qué compuestos tienen actividad antioxidante reportada?
```

### Method Filtering
```
Buscar features detectadas por LC-MS con propiedades antidiabéticas
```

---

## 🐛 Troubleshooting

### Error: "Collection not found"
```bash
# Re-indexar
python 1_setup_pipeline.py
```

### Error: "OPENAI_API_KEY not found"
```bash
# Verificar .env
cat .env | grep OPENAI_API_KEY
```

### Streamlit muy lento
```bash
# Limpiar cache
streamlit cache clear
```

### Modelos no se cargan
```bash
# Verificar instalación
pip install --upgrade sentence-transformers torch
```

---

## 📊 Performance

| Operación | Tiempo | Frecuencia |
|-----------|--------|------------|
| Setup inicial (primera vez) | ~30s | Una vez |
| Setup (con cache) | ~10s | Una vez |
| Query (terminal) | ~2s | Muchas veces |
| Query (streamlit primera) | ~3s | Primera vez |
| Query (streamlit cache) | ~2s | Siguientes |

---

## 🎓 Arquitectura del Profesor (100% Preservada)

- ✅ **C2:** Factory Pattern (PDFLoader)
- ✅ **C3:** Adapter Pattern (Embeddings)
- ✅ **C4:** Strategy Pattern (Retrieval)
- ✅ **C6:** Query Processing (OpenAI)
- ✅ **C7:** Decorator Pattern (Reranking + Repacking)

**Contratos Pydantic intactos:**
- `ProcessedChunk`
- `FilterSuggestion`
- `BenchmarkEntry`

---

## 📚 Documentación Adicional

- `START_HERE.md` - Punto de entrada general
- `CASO_USO_BIOACTIVES.md` - Caso de uso metabolómico
- `CAMBIOS_BIOACTIVES.md` - Cambios vs. código original
- `PRODUCTION_GUIDE.md` - Guía completa de producción

---

## ✨ Next Steps

1. **Agregar más PDFs:** Modifica `1_setup_pipeline.py` para procesar múltiples archivos
2. **API REST:** Usa `rag_core.py` con FastAPI
3. **Fine-tuning:** Ajusta modelos para tu dominio específico
4. **Monitoring:** Agrega logging y métricas

---

**Estado:** ✅ **PRODUCCIÓN READY**

Sistema modular, escalable y listo para Bio-Actives 🧬
