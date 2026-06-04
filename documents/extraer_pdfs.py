import pdfplumber
import os

# Rutas de los PDFs
pdf_files = [
    'MachineLearning_for_CombinatorialOptimizacion.pdf',
    'Roadmap_Catalina.pdf'
]

# Directorio actual
base_dir = os.path.dirname(os.path.abspath(__file__))

for pdf_file in pdf_files:
    pdf_path = os.path.join(base_dir, pdf_file)
    txt_path = os.path.join(base_dir, pdf_file.replace('.pdf', '.txt'))
    
    print(f"Procesando: {pdf_file}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    full_text.append(f"--- Página {page_num} ---\n{text}\n")
            
            # Guardar en archivo de texto
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(full_text))
            
            print(f"✓ Extraído: {txt_path}")
            print(f"  Páginas procesadas: {len(pdf.pages)}")
            
    except Exception as e:
        print(f"✗ Error procesando {pdf_file}: {e}")

print("\nProceso completado.")
