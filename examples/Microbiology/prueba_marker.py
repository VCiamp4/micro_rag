from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
import time

def formatear_tiempo(segundos):
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segundos_restantes = int(segundos % 60)

    partes = []
    if horas > 0:
        partes.append(f"{horas} hora{'s' if horas != 1 else ''}")
    if minutos > 0:
        partes.append(f"{minutos} minuto{'s' if minutos != 1 else ''}")
    if segundos_restantes > 0 or not partes:
        partes.append(f"{segundos_restantes} segundo{'s' if segundos_restantes != 1 else ''}")

    return " ".join(partes)

# Inicialización del conversor
marker = PdfConverter(artifact_dict=create_model_dict())

# Definición de rutas
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "cuento"
OUTPUT_DIR = BASE_DIR / "data_prueba_marker"

# Crear la carpeta de salida si no existe
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Buscando archivos en: {DATA_DIR}")
print(f"Guardando resultados en: {OUTPUT_DIR}")

# Búsqueda recursiva de archivos
files = list(DATA_DIR.glob("**/*.pdf")) + list(DATA_DIR.glob("**/*.txt"))

for f in files:
    inicio = time.time()

    print(f"Procesando: {f.name}...")
    
    try:
        result = marker(str(f))
        markdown = getattr(result, "markdown", None) or getattr(result, "text", "")

        fin = time.time()
        tiempo_str = formatear_tiempo(fin - inicio)

        # 1. Calculamos la ruta relativa respecto a la carpeta origen
        # Esto permite que si el PDF estaba en 'pdf_para_prueba/sub/doc.pdf',
        # se guarde en 'data_prueba_marker/sub/doc.md'
        ruta_relativa = f.relative_to(DATA_DIR)
        output_path = (OUTPUT_DIR / ruta_relativa).with_suffix(".md")

        # 2. Crear las subcarpetas dentro de la carpeta de salida si es necesario
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as out:
            out.write(f"Procesado en: {tiempo_str}\n\n")
            out.write(markdown)
            
        print(f"Finalizado: {output_path.name} ({tiempo_str})")

    except Exception as e:
        print(f"Error procesando {f.name}: {e}")

print("\nListo: Todos los archivos han sido procesados.")