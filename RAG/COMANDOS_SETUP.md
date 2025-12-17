# 🎯 Guía Completa de Comandos - Setup Pipeline

## 📋 Resumen Rápido

| Comando | Descripción | Borra Datos | Re-indexa | Confirmación |
|---------|-------------|-------------|-----------|--------------|
| `python 1_setup_pipeline.py` | Indexar (incremental) | ❌ | ✅ | ❌ |
| `python 1_setup_pipeline.py --list` | Ver documentos | ❌ | ❌ | ❌ |
| `python 1_setup_pipeline.py --delete` | Eliminar uno | ⚠️ Uno | ❌ | ❌ |
| `python 1_setup_pipeline.py --clean` | Limpiar todo | ✅ Todo | ❌ | ✅ |
| `python 1_setup_pipeline.py --recreate` | Recrear colección | ✅ Todo | ✅ | ❌ |

---

## 1️⃣ Indexación Normal (Incremental)

### Comando
```bash
python 1_setup_pipeline.py
```

### ¿Qué hace?
- **Agrega** nuevos documentos sin borrar los existentes
- Crea la colección si no existe
- Preserva documentos previamente indexados

### Cuándo usar
- ✅ Primera vez que usas el sistema
- ✅ Agregar nuevos documentos a la colección
- ✅ Actualizar documentos (eliminar viejo primero con --delete)

### Ejemplo
```bash
# Primera vez
export PDF_PATH="documento1.pdf"
python 1_setup_pipeline.py
# Resultado: 1 documento

# Segunda vez (incremental)
export PDF_PATH="documento2.pdf"
python 1_setup_pipeline.py
# Resultado: 2 documentos (1 + 1)
```

---

## 2️⃣ Listar Documentos

### Comando
```bash
python 1_setup_pipeline.py --list
```

### ¿Qué hace?
- Muestra todos los documentos indexados
- Muestra estadísticas (total de chunks)
- **NO modifica** nada

### Cuándo usar
- ✅ Ver qué hay indexado
- ✅ Verificar después de indexar
- ✅ Encontrar nombre exacto para eliminar

### Ejemplo
```bash
python 1_setup_pipeline.py --list
```

**Salida:**
```
📋 Documentos en la colección 'metabolomics_agent_db':
   Total de chunks: 127
   
   Documentos:
   1. articulo1.pdf
   2. articulo2.pdf
   3. estudio_metabolomico.pdf
```

---

## 3️⃣ Eliminar Documento Específico

### Comando
```bash
python 1_setup_pipeline.py --delete
```

### ¿Qué hace?
- Muestra lista de documentos disponibles
- Permite eliminar **uno** por nombre o número
- Elimina todos los chunks de ese documento
- **NO afecta** otros documentos

### Cuándo usar
- ✅ Eliminar documento duplicado
- ✅ Quitar documento obsoleto
- ✅ Reemplazar un documento (eliminar + re-indexar)

### Ejemplo
```bash
python 1_setup_pipeline.py --delete
```

**Interacción:**
```
📚 Documentos disponibles:
   1. articulo1.pdf
   2. articulo2.pdf
   3. estudio_metabolomico.pdf

🗑️  Nombre del documento a eliminar (o número): 2

✅ Eliminados 15 chunks del documento 'articulo2.pdf'
```

**Dos formas de eliminar:**
```bash
# Opción A: Por número
🗑️  Nombre: 2

# Opción B: Por nombre
🗑️  Nombre: articulo2.pdf
```

---

## 4️⃣ Limpiar Colección Completa (NUEVO) ⭐

### Comando
```bash
python 1_setup_pipeline.py --clean
```

### ¿Qué hace?
- Elimina **TODA** la colección
- Requiere **confirmación** (sí/no)
- **NO re-indexa** automáticamente
- Permite empezar de cero manualmente

### Cuándo usar
- ✅ Quieres re-indexar todo desde cero
- ✅ Cambiar estructura de chunks (chunk_size, overlap)
- ✅ Limpiar antes de indexar carpeta nueva
- ✅ Solucionar problemas de duplicados

### Ejemplo
```bash
python 1_setup_pipeline.py --clean
```

**Interacción:**
```
⚠️  Limpiando colección 'metabolomics_agent_db'...
   Se eliminarán 127 chunks de 10 documentos

¿Estás seguro? (sí/no): sí
✅ Colección 'metabolomics_agent_db' eliminada completamente
```

**Workflow típico:**
```bash
# Paso 1: Limpiar
python 1_setup_pipeline.py --clean

# Paso 2: Re-indexar
export PDF_FOLDER="./todos_mis_documentos"
python 1_setup_pipeline.py
```

---

## 5️⃣ Recrear Colección

### Comando
```bash
python 1_setup_pipeline.py --recreate
```

### ¿Qué hace?
- Elimina **TODA** la colección
- **Re-indexa automáticamente** el archivo/carpeta especificado
- **NO requiere** confirmación (¡cuidado!)
- Todo en un solo paso

### Cuándo usar
- ✅ Reemplazar completamente la colección
- ✅ Workflow automatizado (scripts)
- ❌ NO usar si quieres revisar antes de borrar

### Ejemplo
```bash
export PDF_FOLDER="./nueva_carpeta"
python 1_setup_pipeline.py --recreate
```

**Resultado:**
```
⚠️  Modo --recreate: Se eliminará toda la colección

[Elimina colección]
[Re-indexa documentos de ./nueva_carpeta]

✅ Colección recreada con 5 documentos
```

---

## 🔄 Comparación: --clean vs --recreate

| Aspecto | --clean | --recreate |
|---------|---------|------------|
| **Borra colección** | ✅ Sí | ✅ Sí |
| **Re-indexa automático** | ❌ No | ✅ Sí |
| **Confirmación** | ✅ Requiere | ❌ No requiere |
| **Control manual** | ✅ Más control | ⚠️ Menos control |
| **Uso típico** | Limpieza manual | Scripts automáticos |

### Flujo de trabajo

**Con --clean (más seguro):**
```bash
# 1. Limpiar
python 1_setup_pipeline.py --clean
# Confirmar: sí

# 2. Revisar (opcional)
python 1_setup_pipeline.py --list
# (vacío)

# 3. Re-indexar manualmente
export PDF_FOLDER="./docs"
python 1_setup_pipeline.py
```

**Con --recreate (más rápido):**
```bash
# Todo en uno
export PDF_FOLDER="./docs"
python 1_setup_pipeline.py --recreate
```

---

## 💡 Casos de Uso Reales

### Caso 1: Primera Instalación
```bash
# Indexar documento de ejemplo
python 1_setup_pipeline.py
```

---

### Caso 2: Agregar Documentos Gradualmente
```bash
# Día 1
export PDF_PATH="doc1.pdf"
python 1_setup_pipeline.py

# Día 2 (incremental)
export PDF_PATH="doc2.pdf"
python 1_setup_pipeline.py

# Día 3 (incremental)
export PDF_PATH="doc3.pdf"
python 1_setup_pipeline.py

# Verificar
python 1_setup_pipeline.py --list
# Resultado: 3 documentos
```

---

### Caso 3: Actualizar Documento Existente
```bash
# 1. Eliminar versión vieja
python 1_setup_pipeline.py --delete
# Escribir: documento_viejo.pdf

# 2. Indexar versión nueva
export PDF_PATH="documento_nuevo.pdf"
python 1_setup_pipeline.py
```

---

### Caso 4: Cambiar Completamente la Colección
```bash
# Opción A: Con confirmación (recomendado)
python 1_setup_pipeline.py --clean
# Confirmar: sí
export PDF_FOLDER="./nueva_carpeta"
python 1_setup_pipeline.py

# Opción B: Automático
export PDF_FOLDER="./nueva_carpeta"
python 1_setup_pipeline.py --recreate
```

---

### Caso 5: Solucionar Duplicados
```bash
# Si indexaste el mismo documento dos veces

# Opción A: Eliminar duplicado específico
python 1_setup_pipeline.py --delete
# Eliminar una copia

# Opción B: Limpiar todo y re-indexar
python 1_setup_pipeline.py --clean
export PDF_FOLDER="./docs_limpios"
python 1_setup_pipeline.py
```

---

### Caso 6: Cambiar Chunk Size
```bash
# Limpiar colección
python 1_setup_pipeline.py --clean

# Modificar chunk_size en rag_core.py o 1_setup_pipeline.py
# Luego re-indexar
export PDF_FOLDER="./documentos"
python 1_setup_pipeline.py
```

---

## 🛡️ Seguridad y Confirmaciones

### Sin confirmación (automático)
```bash
python 1_setup_pipeline.py          # ✅ Agrega
python 1_setup_pipeline.py --list   # ✅ Solo lee
python 1_setup_pipeline.py --delete # ✅ Elimina uno
python 1_setup_pipeline.py --recreate # ⚠️ BORRA TODO
```

### Con confirmación
```bash
python 1_setup_pipeline.py --clean  # ✅ Pide confirmación
```

**Recomendación:** Usa `--clean` en lugar de `--recreate` si quieres más control.

---

## 🔍 Verificación y Debugging

### Verificar antes de modificar
```bash
# 1. Ver estado actual
python 1_setup_pipeline.py --list

# 2. Decidir qué hacer
# - Si hay duplicados: --delete
# - Si quieres empezar de cero: --clean
# - Si quieres agregar: (normal)
```

### Verificar después de modificar
```bash
# Después de indexar
python 1_setup_pipeline.py --list

# Hacer query de prueba
python 2_query_rag.py
```

---

## 📊 Matriz de Decisión

| Quiero... | Comando |
|-----------|---------|
| Indexar primer documento | `python 1_setup_pipeline.py` |
| Agregar más documentos | `python 1_setup_pipeline.py` |
| Ver qué tengo indexado | `python 1_setup_pipeline.py --list` |
| Quitar un documento | `python 1_setup_pipeline.py --delete` |
| Empezar de cero (manual) | `python 1_setup_pipeline.py --clean` |
| Empezar de cero (auto) | `python 1_setup_pipeline.py --recreate` |
| Reemplazar documento | `--delete` + indexar de nuevo |
| Cambiar chunk_size | `--clean` + modificar código + indexar |

---

## ⚠️ Advertencias Importantes

### 🚨 --recreate NO pide confirmación
```bash
# Esto BORRA TODO inmediatamente
python 1_setup_pipeline.py --recreate
```

**Solución:** Usa `--clean` si quieres más control.

### 🚨 No hay "undo"
Una vez que borras con `--clean` o `--recreate`, **no se puede recuperar**.

**Solución:** Haz backup de `qdrant_storage/` antes de limpiar.

```bash
# Backup antes de limpiar
tar -czf qdrant_backup.tar.gz qdrant_storage/

# Limpiar
python 1_setup_pipeline.py --clean

# Si te arrepientes, restaurar
rm -rf qdrant_storage/
tar -xzf qdrant_backup.tar.gz
```

---

## ✅ Resumen Final

```bash
# Comandos seguros (no borran)
python 1_setup_pipeline.py          # Agrega
python 1_setup_pipeline.py --list   # Lista

# Comandos que borran (con control)
python 1_setup_pipeline.py --delete # Borra uno
python 1_setup_pipeline.py --clean  # Borra todo (CON confirmación)

# Comando que borra (sin control)
python 1_setup_pipeline.py --recreate # Borra todo (SIN confirmación)
```

**Recomendación:** Usa `--clean` en lugar de `--recreate` para tener más control. ✅
