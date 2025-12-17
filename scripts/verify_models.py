#!/usr/bin/env python3
"""
Script para verificar y corregir la definición de modelos Pydantic.
Asegura que ProcessedChunk y FilterSuggestion tengan los campos correctos.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_models():
    """Verifica que los modelos tengan los campos requeridos"""
    
    print("🔍 Verificando modelos Pydantic...")
    
    try:
        from src.domain.models import ProcessedChunk, FilterSuggestion
        
        # Verificar ProcessedChunk
        print("\n📦 Verificando ProcessedChunk...")
        chunk_fields = ProcessedChunk.model_fields.keys()
        
        required_chunk_fields = [
            'content', 'source_file', 'publication_year', 'chunk_id',
            'mz_values', 'rt_values', 'compound_names', 'bioactivities'
        ]
        
        missing_chunk = [f for f in required_chunk_fields if f not in chunk_fields]
        
        if missing_chunk:
            print(f"❌ Campos faltantes en ProcessedChunk: {missing_chunk}")
            return False
        else:
            print(f"✅ ProcessedChunk tiene todos los campos requeridos")
            print(f"   Campos: {list(chunk_fields)}")
        
        # Verificar FilterSuggestion
        print("\n🔍 Verificando FilterSuggestion...")
        filter_fields = FilterSuggestion.model_fields.keys()
        
        required_filter_fields = [
            'rewritten_query', 'metadata_filters', 'target_mz', 'target_rt'
        ]
        
        missing_filter = [f for f in required_filter_fields if f not in filter_fields]
        
        if missing_filter:
            print(f"❌ Campos faltantes en FilterSuggestion: {missing_filter}")
            return False
        else:
            print(f"✅ FilterSuggestion tiene todos los campos requeridos")
            print(f"   Campos: {list(filter_fields)}")
        
        # Test de creación
        print("\n🧪 Probando creación de objetos...")
        
        try:
            chunk = ProcessedChunk(
                content="Test content",
                source_file="test.pdf",
                chunk_id="test-1"
            )
            print(f"✅ ProcessedChunk creado correctamente")
            print(f"   mz_values: {chunk.mz_values}")
            print(f"   rt_values: {chunk.rt_values}")
        except Exception as e:
            print(f"❌ Error creando ProcessedChunk: {e}")
            return False
        
        try:
            filter_sug = FilterSuggestion(
                rewritten_query="test query",
                metadata_filters={"target_mz": 449.107}
            )
            print(f"✅ FilterSuggestion creado correctamente")
            print(f"   target_mz: {filter_sug.target_mz}")
            print(f"   target_rt: {filter_sug.target_rt}")
        except Exception as e:
            print(f"❌ Error creando FilterSuggestion: {e}")
            return False
        
        print("\n✅ Todos los modelos están correctos!")
        return True
        
    except ImportError as e:
        print(f"❌ Error importando modelos: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


def show_fix_instructions():
    """Muestra instrucciones para corregir los modelos"""
    print("\n" + "="*70)
    print("📋 INSTRUCCIONES PARA CORREGIR")
    print("="*70)
    print("""
1. Abre el archivo: src/domain/models.py

2. Asegúrate que ProcessedChunk tenga estos campos:

   class ProcessedChunk(BaseModel):
       content: str
       source_file: str | None = None
       publication_year: int = 2024
       chunk_id: str | None = None
       
       # ESTOS CAMPOS SON CRÍTICOS:
       mz_values: List[float] | None = Field(default=None)
       rt_values: List[float] | None = Field(default=None)
       compound_names: List[str] | None = Field(default=None)
       bioactivities: List[str] | None = Field(default=None)
       
       metadata: Dict[str, Any] = Field(default_factory=dict)

3. Asegúrate que FilterSuggestion tenga estos campos:

   class FilterSuggestion(BaseModel):
       rewritten_query: str
       metadata_filters: Dict[str, Any] = Field(default_factory=dict)
       
       # ESTOS CAMPOS SON CRÍTICOS:
       target_mz: float | None = Field(default=None)
       target_rt: float | None = Field(default=None)

4. Los Field(default=None) son importantes para backward compatibility

5. Después de corregir, reinicia Streamlit:
   streamlit run src/app.py
""")


if __name__ == "__main__":
    print("="*70)
    print("🔧 VERIFICADOR DE MODELOS PYDANTIC")
    print("="*70)
    
    success = check_models()
    
    if not success:
        show_fix_instructions()
        sys.exit(1)
    else:
        print("\n🎉 Todo está correcto. Puedes ejecutar el sistema sin problemas.")
        sys.exit(0)