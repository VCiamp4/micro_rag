"""
title: RAG Microbiologia Haystack
author: Grupo 7
description: Pipeline de Haystack para microbiología médica.
version: 1.0
requirements: haystack-ai, ollama-haystack, pypdf
"""
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from providers import get_document_store, get_doc_embedder, file_hash
from pathlib import Path


# --- CONFIGURACION ---
#EMBEDDING_MODEL = "bge-m3"
EMBEDDING_MODEL = "qwen-3-embedding:0.6b"
OLLAMA_BASE_URL = "http://localhost:11434"
DATA_DIR = Path(__file__).parent / "data"



# --- DOCUMENT STORE ---
document_store = get_document_store()

# --- INDEXING PIPELINE ---
# Lee archivos PDF y TXT desde data/, los parte en chunks y los embeddea

file_router = FileTypeRouter(mime_types=["application/pdf", "text/plain"])
pdf_converter = PyPDFToDocument()
txt_converter = TextFileToDocument()
joiner = DocumentJoiner()

# Cada chunk: ~10 oraciones con overlap de 2 para no perder contexto entre chunks
splitter = DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2)


doc_embedder = get_doc_embedder()

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("router", file_router)
indexing_pipeline.add_component("pdf_converter", pdf_converter)
indexing_pipeline.add_component("txt_converter", txt_converter)
indexing_pipeline.add_component("joiner", joiner)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("embedder", doc_embedder)

indexing_pipeline.connect("router.application/pdf", "pdf_converter.sources")
indexing_pipeline.connect("router.text/plain", "txt_converter.sources")
indexing_pipeline.connect("pdf_converter", "joiner")
indexing_pipeline.connect("txt_converter", "joiner")
indexing_pipeline.connect("joiner", "splitter")
indexing_pipeline.connect("splitter", "embedder")


# Buscar todos los archivos en data/
# files = list(DATA_DIR.glob("**/*.pdf")) + list(DATA_DIR.glob("**/*.txt"))


files = [f.resolve() for f in DATA_DIR.glob("**/*.pdf")] + \
        [f.resolve() for f in DATA_DIR.glob("**/*.txt")]

if not files:
    print(f"No se encontraron archivos en {DATA_DIR}/")
    print("Agregá PDFs o archivos .txt con contenido de microbiología y volvé a correr.")
    exit(1)

# Aca se van a guardar los archivos ya cargados
existing_docs = document_store.filter_documents()

# Esto indica si hubo cambios en los documentos
existing_hashes = set(d.meta.get("file_hash") for d in existing_docs)

files_to_index = list()

for file in files:
    hashing = file_hash(file)
    if hashing not in existing_hashes:
        files_to_index.append(file)


if len(files_to_index) == 0:
    print("No se encontraron archivos nuevos o modificados. Cancelando indexing pipeline")
    exit(2)

print(f"Indexando {len(files)} archivo(s)...")
for f in files:
    print(f"  - {f.name}")

result = indexing_pipeline.run({"router": {"sources": files}})
docs_with_embeddings = result["embedder"]["documents"]


# Agregamos como metadata un hash
for doc in docs_with_embeddings:
    
    source_path = doc.meta.get("file_path") or doc.meta.get("source")

if source_path:
    source_path = Path(source_path)

    # Si no es absoluto, lo resolvemos contra DATA_DIR
    if not source_path.is_absolute():
        source_path = (DATA_DIR / source_path).resolve()

    doc.meta["file_path"] = str(source_path)
    doc.meta["file_hash"] = file_hash(source_path)

document_store.write_documents(docs_with_embeddings)

print(f"Indexados {len(docs_with_embeddings)} chunks.\n")
