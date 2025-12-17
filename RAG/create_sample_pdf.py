"""
Script para crear un PDF de prueba si no tienes uno disponible.
Útil para probar el código sin necesidad de un PDF real.
"""

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Para crear PDFs de prueba, instala: pip install reportlab")


def create_sample_pdf(filename="bioactives_sample.pdf"):
    """Crea un PDF de prueba con contenido de metabolómica y bio-actives"""
    
    if not REPORTLAB_AVAILABLE:
        print("\n⚠️  ReportLab no está instalado.")
        print("Instala con: pip install reportlab")
        return
    
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Página 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, "Identificación de Compuestos Bioactivos en Té Verde")
    
    c.setFont("Helvetica", 12)
    y_position = height - 1.5*inch
    
    content_page1 = [
        "Resumen:",
        "Este estudio identifica compuestos bioactivos en extractos de Té Verde",
        "utilizando LC-MS de alta resolución. Se detectaron flavonoides con",
        "propiedades antioxidantes y antidiabéticas.",
        "",
        "Metodología:",
        "Análisis por Cromatografía Líquida acoplada a Espectrometría de Masas (LC-MS).",
        "Columna C18, fase móvil: agua/acetonitrilo con 0.1% ácido fórmico.",
        "Detección en modo negativo ESI, rango m/z 100-1000.",
        "",
        "Feature 1: m/z 449.107, RT 8.2 min",
        "Anotación putativa: Myricetina 3-galactósido (C21H20O12)",
        "Fuente: Base de datos PubChem, fórmula exacta match.",
    ]
    
    for line in content_page1:
        c.drawString(1*inch, y_position, line)
        y_position -= 0.3*inch
    
    c.showPage()
    
    # Página 2
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, height - 1*inch, "Bioactividades Reportadas")
    
    c.setFont("Helvetica", 12)
    y_position = height - 1.5*inch
    
    content_page2 = [
        "Myricetina - Propiedades Biológicas:",
        "",
        "1. Actividad Antioxidante:",
        "   - Fuerte capacidad de captación de radicales libres (DPPH).",
        "   - EC50: 12.5 μM (Ref: PubMed ID 23456789).",
        "",
        "2. Actividad Antidiabética:",
        "   - Inhibición de α-glucosidasa: IC50 25.3 μM.",
        "   - Mejora la sensibilidad a la insulina en modelos in vitro.",
        "   - Fuente: PubChem BioAssay 5678.",
        "",
        "3. Actividad Antiinflamatoria:",
        "   - Reducción de citoquinas pro-inflamatorias (IL-6, TNF-α).",
        "   - Mecanismo: Inhibición de NF-κB.",
        "",
        "4. Actividad Antiplaquetaria:",
        "   - Previene la agregación plaquetaria inducida por ADP.",
        "   - Dosis efectiva: 50 μM (estudio in vitro).",
    ]
    
    for line in content_page2:
        c.drawString(1*inch, y_position, line)
        y_position -= 0.3*inch
    
    c.showPage()
    
    # Página 3
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, height - 1*inch, "Datos Internos y Referencias")
    
    c.setFont("Helvetica", 12)
    y_position = height - 1.5*inch
    
    content_page3 = [
        "Contexto Interno:",
        "",
        "Feature similar detectada en análisis previos:",
        "- Muestra: Arándano 004",
        "- m/z 449.1, RT 8.1 min",
        "- Anotación: Myricetina-derivado",
        "- Método: LC-MS (Agilent 6550 iFunnel Q-TOF)",
        "",
        "Otras Features de Interés en Té Verde:",
        "",
        "Feature 2: m/z 609.146, RT 6.8 min",
        "Anotación: Rutina (Quercetina-3-rutinósido)",
        "Bioactividad: Antioxidante, cardioprotector",
        "",
        "Feature 3: m/z 289.071, RT 9.5 min",
        "Anotación: Catequina",
        "Bioactividad: Neuroprotector, antiinflamatorio",
        "",
        "Referencias:",
        "[1] Zhang et al. (2023). Flavonoid profiling in green tea. J. Agric. Food Chem.",
        "[2] Smith et al. (2024). Bioactive compounds from botanical sources. Food Res. Int.",
        "[3] PubChem BioAssay Database. https://pubchem.ncbi.nlm.nih.gov/",
    ]
    
    for line in content_page3:
        c.drawString(1*inch, y_position, line)
        y_position -= 0.3*inch
    
    c.save()
    print(f"\n✅ PDF de Bio-Actives creado: {filename}")
    print(f"   El archivo contiene 3 páginas con datos metabolómicos y bioactividades.")


if __name__ == "__main__":
    print("=" * 60)
    print("Generador de PDF de Bio-Actives para RAG Metabolómico")
    print("=" * 60)
    
    create_sample_pdf("bioactives_sample.pdf")
    
    print("\n📖 Contenido del PDF:")
    print("   - Página 1: Identificación de Myricetina en Té Verde")
    print("   - Página 2: Bioactividades reportadas (antioxidante, antidiabético)")
    print("   - Página 3: Datos internos y referencias")
    
    print("\n💡 Uso:")
    print("   1. Ejecuta este script: python create_sample_pdf.py")
    print("   2. Se creará 'bioactives_sample.pdf'")
    print("   3. Cambia la ruta en rag_production.py:")
    print('      pdf_path = "bioactives_sample.pdf"')
    print("   4. Ejecuta el RAG: python rag_production.py")
    
    print("\n" + "=" * 60)
