from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

marker = PdfConverter(artifact_dict=create_model_dict())
DATA_DIR = Path(__file__).parent / "data"
files = list(DATA_DIR.glob("**/*.pdf")) + list(DATA_DIR.glob("**/*.txt"))

for f in files:
    result = marker(str(f))
    markdown = getattr(result, "markdown", None) or getattr(result, "text", "")
    output_path = f.with_suffix(".md")

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(markdown)

print("listo")