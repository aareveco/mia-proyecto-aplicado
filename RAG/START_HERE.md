# 🎯 EMPIEZA AQUÍ - RAG Bio-Actives

## ¿Qué es este proyecto?

Sistema RAG para **automatizar la anotación funcional de features metabolómicas** (compuestos bioactivos), manteniendo la arquitectura exacta de patrones de diseño del profesor.

### Problema que Resuelve

Vincular una **feature cromatográfica** (m/z + RT) → **compuesto químico** → **bioactividad** es actualmente un proceso manual y lento.

Este RAG automatiza todo el proceso integrando:
- ✅ Bases de datos públicas (PubChem, PubMed)
- ✅ Literatura científica
- ✅ Datos internos de laboratorio

---

## 🚀 Quick Start (5 minutos)

```bash
# 1. Instalar dependencias
pip install -r requirements_production.txt

# 2. Iniciar Qdrant Local con Docker
docker-compose up -d

# 3. Configurar OpenAI
cp .env.example .env
# Editar .env y agregar: OPENAI_API_KEY=sk-proj-xxxxx...

# 4. Crear PDF de Bio-Actives
python create_sample_pdf.py
# Genera: bioactives_sample.pdf con datos de Myricetina

# 5. ¡Ejecutar!
python rag_production.py
```

---

## 📚 ¿Por dónde empezar?

### 🧬 Entender el Caso de Uso (Bio-Actives)

1. Lee **[CASO_USO_BIOACTIVES.md](CASO_USO_BIOACTIVES.md)** - Explicación del problema metabolómico

### 👤 Soy Usuario / Primera Vez

1. Lee **[QUICKSTART.md](QUICKSTART.md)** (5 min)
2. Ejecuta **verify_setup.py** para verificar instalación
3. Ejecuta **rag_production.py**

### 👨‍💻 Soy Desarrollador

1. Lee **[CAMBIOS_DETALLADOS.md](CAMBIOS_DETALLADOS.md)** - Comparación mocks vs. producción
2. Revisa **[rag_production.py](rag_production.py)** - Código fuente
3. Lee **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)** - Guía completa

### 👔 Soy Profesor / Manager

1. Lee **[RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md)** - Vista de alto nivel
2. Lee **[CASO_USO_BIOACTIVES.md](CASO_USO_BIOACTIVES.md)** - Aplicación a metabolómica
3. Revisa **[INDEX.md](INDEX.md)** - Índice completo del proyecto

---

## 🎯 Lo que hace este RAG

**Input:** Una pregunta como:

```
"Tengo una feature con m/z 449.107, RT 8.2 min, 
detectada en mi muestra de Té Verde. 
¿Qué es y qué bioactividad tiene?"
```

**Output:** Los 3 chunks más relevantes con scores de reranking:

```
[1] Chunk ID: bioactives_sample-chunk-5
    Score: 9.2341
    Método: LC-MS
    Contenido: Feature m/z 449.107, RT 8.2 min
               Anotación: Myricetina 3-galactósido (C21H20O12)...

[2] Chunk ID: bioactives_sample-chunk-8
    Score: 8.7654
    Método: LC-MS
    Contenido: Myricetina - Actividad Antioxidante EC50: 12.5 μM
               Actividad Antidiabética: Inhibición α-glucosidasa...

[3] Chunk ID: bioactives_sample-chunk-11
    Score: 7.9876
    Método: LC-MS
    Contenido: Contexto Interno: Feature similar detectada en
               Muestra Arándano 004...
```

---

## ✨ Stack Tecnológico

- **LLM:** OpenAI GPT-4o-mini (query rewriting)
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Reranking:** Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- **Vector DB:** Qdrant con búsqueda COSINE
- **Sparse Vectors:** BM25Okapi
- **PDF Processing:** PyPDF2 + LangChain splitters

---

## 📊 Comparación con código del Profesor

| Componente | Código Profesor | Versión Producción |
|------------|----------------|-------------------|
| **LLM** | Mock (hardcoded) | ✅ OpenAI real |
| **Embeddings** | Random vectors | ✅ Sentence Transformers |
| **Reranking** | Random scores | ✅ Cross-Encoder |
| **Vector DB** | Dict en memoria | ✅ Qdrant |
| **PDF Loading** | 3 chunks fijos | ✅ Dinámico con PyPDF2 |

**Arquitectura:** ✅ **100% PRESERVADA**
- Todos los patrones de diseño intactos
- Mismas interfaces (ABC)
- Mismos contratos Pydantic

---

## 🎓 Arquitectura (Patrones del Profesor)

```
C2: FACTORY → PDFLoader (PyPDF2)
    ↓
C3: ADAPTER → Embeddings (Transformers) + BM25
    ↓
C4: STRATEGY → Búsqueda Híbrida (Qdrant)
    ↓
C6: STRATEGY → Query Rewriting (OpenAI)
    ↓
C7: DECORATOR → Reranking (Cross-Encoder) + Repacking
```

---

## 💰 Costos

- **Por query:** ~$0.001-0.002 (OpenAI)
- **100 queries:** ~$0.10-0.20
- **Qdrant local:** Gratis (Docker)
- **Modelos locales:** Gratis

---

## 📂 Archivos Importantes

| Archivo | Descripción |
|---------|-------------|
| **rag_production.py** | Código principal (752 líneas) |
| **QUICKSTART.md** | Inicio rápido |
| **PRODUCTION_GUIDE.md** | Guía completa |
| **verify_setup.py** | Verificar instalación |
| **docker-compose.yml** | Setup de Qdrant |

---

## ✅ Checklist de Setup

```
□ Python 3.10+ instalado
□ Docker corriendo
□ Dependencies: pip install -r requirements_production.txt
□ Qdrant: docker-compose up -d
□ Variables: cp .env.example .env (agregar OPENAI_API_KEY)
□ Verificar: python verify_setup.py
□ PDF de prueba: python create_sample_pdf.py
□ Ejecutar: python rag_production.py
```

---

## 🐛 Problemas Comunes

### Qdrant no conecta
```bash
docker ps | grep qdrant
docker-compose up -d
```

### OpenAI API key inválida
```bash
cat .env  # Verificar OPENAI_API_KEY
```

### Primera ejecución lenta
Normal - descarga modelos (~1-2 GB). Caché: `~/.cache/huggingface/`

---

## 📖 Documentación Completa

- **[INDEX.md](INDEX.md)** - Índice completo
- **[QUICKSTART.md](QUICKSTART.md)** - 5 pasos rápidos
- **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)** - Guía detallada
- **[CAMBIOS_DETALLADOS.md](CAMBIOS_DETALLADOS.md)** - Comparación línea por línea
- **[RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md)** - Vista ejecutiva

---

## 🎯 Próximos Pasos

1. ✅ Ejecuta con PDF de prueba
2. ✅ Usa tus propios PDFs
3. ✅ Personaliza la query
4. ✅ Ajusta parámetros de chunking
5. ✅ Cambia modelos de embeddings/reranking

---

## 🤝 Conclusión

Este proyecto demuestra cómo un código educativo bien diseñado puede convertirse en producción **sin cambiar la arquitectura**, solo reemplazando implementaciones.

**Resultado:**
- ✅ RAG 100% funcional
- ✅ Arquitectura del profesor preservada
- ✅ Sin mocks - Todo real
- ✅ Producción ready

---

**¿Listo?** → [QUICKSTART.md](QUICKSTART.md)

**¿Dudas?** → [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)

**¿Detalles técnicos?** → [CAMBIOS_DETALLADOS.md](CAMBIOS_DETALLADOS.md)
