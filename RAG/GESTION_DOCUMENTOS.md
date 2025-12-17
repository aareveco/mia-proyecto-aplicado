# 📚 Guía de Gestión de Documentos - RAG Bio-Actives

## 🎯 Resumen

Los datos procesados se **persisten automáticamente en Qdrant**. No necesitas reindexar cada vez.

---

## 📖 Comandos Disponibles

### 1️⃣ Indexar Nuevo Documento (Modo Incremental)

```bash
# Agregar documento SIN eliminar los existentes
python 1_setup_pipeline.py

# O especificar otro PDF
export PDF_PATH="nuevo_documento.pdf"
python 1_setup_pipeline.py
```

**Resultado:** Los nuevos chunks se **agregan** a la colección existente.

---

### 2️⃣ Limpiar Colección Completa (NUEVO) ⭐

```bash
# Borrar TODOS los documentos de la colección
python 1_setup_pipeline.py --clean
```

**Prompt interactivo:**
```
⚠️  Limpiando colección 'metabolomics_agent_db'...
   Se eliminarán 127 chunks de 10 documentos

¿Estás seguro? (sí/no): sí
✅ Colección 'metabolomics_agent_db' eliminada completamente
```

**Uso típico:** Limpiar antes de re-indexar todo desde cero.

```bash
# Limpiar colección
python 1_setup_pipeline.py --clean

# Re-indexar carpeta completa
export PDF_FOLDER="./mis_documentos"
python 1_setup_pipeline.py
```

---

### 3️⃣ Recrear Colección desde Cero

```bash
# Elimina TODA la colección y crea una nueva
python 1_setup_pipeline.py --recreate
```

**⚠️ ADVERTENCIA:** Esto elimina **todos** los documentos previamente indexados.

**Diferencia con --clean:**
- `--clean`: Borra la colección y **termina** (debes re-indexar manualmente)
- `--recreate`: Borra la colección y **re-indexa automáticamente** el archivo/carpeta especificado

---

### 4️⃣ Listar Documentos Indexados

```bash
python 1_setup_pipeline.py --list
```

**Salida esperada:**
```
📋 Documentos en la colección 'metabolomics_agent_db':
   Total de chunks: 15
   
   Documentos:
   1. bioactives_sample.pdf
   2. blueberry_analysis.pdf
   3. green_tea_metabolites.pdf
```

---

### 5️⃣ Eliminar Documento Específico

```bash
python 1_setup_pipeline.py --delete
```

**Prompt interactivo:**
```
🗑️  Nombre del documento a eliminar: bioactives_sample.pdf
✅ Eliminados 5 chunks del documento 'bioactives_sample.pdf'
```

---

## 🔄 Workflows Típicos

### Caso A: Agregar Múltiples Documentos

```bash
# Documento 1
export PDF_PATH="doc1.pdf"
python 1_setup_pipeline.py

# Documento 2 (modo incremental automático)
export PDF_PATH="doc2.pdf"
python 1_setup_pipeline.py

# Documento 3
export PDF_PATH="doc3.pdf"
python 1_setup_pipeline.py

# Verificar todos los documentos
python 1_setup_pipeline.py --list
```

**Resultado:** Colección con chunks de los 3 documentos.

---

### Caso B: Reemplazar un Documento

```bash
# 1. Eliminar documento viejo
python 1_setup_pipeline.py --delete
# Escribir: "old_document.pdf"

# 2. Indexar documento nuevo
export PDF_PATH="new_document.pdf"
python 1_setup_pipeline.py
```

---

### Caso C: Limpiar y Re-indexar Todo

```bash
# 1. Limpiar colección completa
python 1_setup_pipeline.py --clean
# Confirmar: sí

# 2. Re-indexar carpeta desde cero
export PDF_FOLDER="./todos_mis_documentos"
python 1_setup_pipeline.py

# Resultado: Colección limpia con solo los nuevos documentos
```

---

### Caso D: Limpiar Todo y Empezar de Nuevo

```bash
# Eliminar colección completa
python 1_setup_pipeline.py --recreate

# Los queries seguirán funcionando, pero la colección estará vacía
```

---

## 📊 Información de la Colección

### Ver Estadísticas

```python
from rag_core import QdrantVectorStore

db = QdrantVectorStore(collection_name="metabolomics_agent_db")

# Ver info
info = db.get_collection_info()
print(f"Total chunks: {info['points_count']}")

# Listar documentos
docs = db.list_documents()
print(f"Documentos: {docs}")
```

---

## 🗄️ Persistencia de Datos

### Dónde se Guardan los Datos

#### Qdrant Cloud
```
☁️ Los datos se guardan en Qdrant Cloud
✅ Persistencia automática
✅ Disponible desde cualquier máquina
✅ Backups gestionados por Qdrant
```

#### Qdrant Local (Docker)
```
📁 Los datos se guardan en:
   ./qdrant_storage/

✅ Persiste entre reinicios de Docker
✅ Backup manual: tar -czf backup.tar.gz qdrant_storage/
⚠️  Si eliminas el volumen Docker, pierdes los datos
```

---

## 🔍 Verificar Persistencia

### Test Rápido

```bash
# 1. Indexar documento
python 1_setup_pipeline.py

# 2. Hacer consulta
python 2_query_rag.py
# Escribir query y ver resultados

# 3. Cerrar todo y reiniciar Docker (si usas local)
docker-compose restart

# 4. Hacer consulta de nuevo (sin re-indexar)
python 2_query_rag.py
# ✅ Los datos siguen ahí!
```

---

## 🧹 Limpieza de Datos

### Eliminar Documentos Viejos

```bash
# Ver qué documentos tienes
python 1_setup_pipeline.py --list

# Eliminar los que no necesites
python 1_setup_pipeline.py --delete
```

### Eliminar Duplicados

Si indexaste el mismo documento dos veces:

```bash
# Eliminar documento
python 1_setup_pipeline.py --delete
# Escribir nombre exacto

# Re-indexar
export PDF_PATH="documento.pdf"
python 1_setup_pipeline.py
```

---

## 💡 Tips y Mejores Prácticas

### ✅ DO's

1. **Usa modo incremental por defecto**
   ```bash
   python 1_setup_pipeline.py  # Agrega, no elimina
   ```

2. **Lista documentos antes de eliminar**
   ```bash
   python 1_setup_pipeline.py --list
   ```

3. **Nombres de archivo descriptivos**
   ```
   green_tea_2024.pdf  ✅
   documento.pdf       ❌
   ```

4. **Verifica después de indexar**
   ```bash
   python 1_setup_pipeline.py --list
   ```

### ❌ DON'Ts

1. **No uses --recreate sin querer**
   ```bash
   python 1_setup_pipeline.py --recreate  # Elimina TODO!
   ```

2. **No indexes el mismo archivo dos veces sin eliminar primero**
   ```bash
   # Malo: Duplicados
   python 1_setup_pipeline.py  # Primera vez
   python 1_setup_pipeline.py  # ¡Duplicado!
   
   # Bueno: Eliminar primero
   python 1_setup_pipeline.py --delete
   python 1_setup_pipeline.py
   ```

---

## 🐛 Troubleshooting

### "Collection not found"

```bash
# La colección no existe, créala:
python 1_setup_pipeline.py
```

### "Duplicated chunks"

```bash
# Eliminar documento y re-indexar
python 1_setup_pipeline.py --delete
export PDF_PATH="documento.pdf"
python 1_setup_pipeline.py
```

### "No results in queries"

```bash
# Verificar que hay documentos
python 1_setup_pipeline.py --list

# Si está vacío, indexar
python 1_setup_pipeline.py
```

### Backup de Qdrant Local

```bash
# Detener Docker
docker-compose down

# Backup
tar -czf qdrant_backup_$(date +%Y%m%d).tar.gz qdrant_storage/

# Reiniciar
docker-compose up -d
```

---

## 📈 Escalabilidad

### Límites Prácticos

| Métrica | Qdrant Local | Qdrant Cloud |
|---------|--------------|--------------|
| Documentos | ~1000 | ~1M+ |
| Chunks | ~100K | ~10M+ |
| Tamaño total | ~10GB | ~100GB+ |
| Performance | Buena | Excelente |

### Si tienes muchos documentos:

1. **Usar Qdrant Cloud** (más escalable)
2. **Chunking más pequeño** (menos chunks por doc)
3. **Limpieza periódica** (eliminar docs viejos)

---

## ✨ Resumen de Comandos

```bash
# Agregar documento
python 1_setup_pipeline.py

# Listar documentos
python 1_setup_pipeline.py --list

# Eliminar documento específico
python 1_setup_pipeline.py --delete

# Limpiar colección completa (requiere confirmación)
python 1_setup_pipeline.py --clean

# Recrear colección y re-indexar
python 1_setup_pipeline.py --recreate

# Especificar PDF
export PDF_PATH="mi_documento.pdf"
python 1_setup_pipeline.py
```

---

**Comparación rápida:**
| Comando | Borra datos | Re-indexa | Confirmación |
|---------|-------------|-----------|--------------|
| (normal) | ❌ No | ✅ Sí | No |
| --clean | ✅ Todo | ❌ No | ✅ Sí |
| --recreate | ✅ Todo | ✅ Sí | No |
| --delete | ⚠️ Uno | ❌ No | No |

---

**Estado:** ✅ Datos persistentes y gestionables 🗄️