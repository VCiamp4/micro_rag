"""
title: RAG Microbiologia Haystack
author: Grupo 7
description: Pipeline de Haystack para microbiología médica.
version: 1.0
requirements: haystack-ai, ollama-haystack, pypdf
"""

from haystack import Pipeline
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack.components.builders import PromptBuilder
<<<<<<< HEAD
from providers import *
=======
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder, OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator


# --- CONFIGURACION ---
EMBEDDING_MODEL = "bge-m3"
#GENERATION_MODEL = "ministral-3:3b" #llama3.2 #jfm2.5 # gooogle colab #api de groq gratuita 7 tok/min
GENERATION_MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
DATA_DIR = Path(__file__).parent / "data"

# --- DOCUMENT STORE ---
document_store = InMemoryDocumentStore() # Candidato para pasar a PgvectorDocumentStore()

'''
document_store = PgvectorDocumentStore(
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="hnsw",
)
'''

# --- INDEXING PIPELINE ---
# Lee archivos PDF y TXT desde data/, los parte en chunks y los embeddea

file_router = FileTypeRouter(mime_types=["application/pdf", "text/plain"])
pdf_converter = PyPDFToDocument()
txt_converter = TextFileToDocument()
joiner = DocumentJoiner()

# Cada chunk: ~10 oraciones con overlap de 2 para no perder contexto entre chunks
splitter = DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2)


doc_embedder = OllamaDocumentEmbedder(
    model=EMBEDDING_MODEL,
    url=OLLAMA_BASE_URL,
    batch_size=32
)

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
files = list(DATA_DIR.glob("**/*.pdf")) + list(DATA_DIR.glob("**/*.txt"))

if not files:
    print(f"No se encontraron archivos en {DATA_DIR}/")
    print("Agregá PDFs o archivos .txt con contenido de microbiología y volvé a correr.")
    exit(1)

print(f"Indexando {len(files)} archivo(s)...")
for f in files:
    print(f"  - {f.name}")

result = indexing_pipeline.run({"router": {"sources": files}})
docs_with_embeddings = result["embedder"]["documents"]
document_store.write_documents(docs_with_embeddings)
print(f"Indexados {len(docs_with_embeddings)} chunks.\n")
>>>>>>> main


# --- QUERY PIPELINE ---

template = """
Sos un asistente de microbiología médica universitaria. Respondé en español, de forma clara y clínica.
Basate únicamente en el contexto provisto. Si no hay información suficiente, decí: "No hay suficiente información en el contexto."

Contexto:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Pregunta clínica: {{question}}

Respuesta:"""

text_embedder = get_text_embedder()

document_store = get_document_store()

retriever = PgvectorEmbeddingRetriever(
    document_store=document_store, 
    top_k=5
)

prompt_builder = PromptBuilder(template=template)

generator = get_generator()

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", text_embedder)
query_pipeline.add_component("retriever", retriever)
query_pipeline.add_component("prompt_builder", prompt_builder)
query_pipeline.add_component("llm", generator)

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "llm")


# --- LOOP INTERACTIVO ---
print("RAG Microbiologia - Preguntas Clinicas")
print("Escribi 'salir' para terminar\n")

while True:
    question = input("Pregunta: ").strip()
    if question.lower() in ("salir", "exit", "quit"):
        break
    if not question:
        continue

    results = query_pipeline.run({
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question}
    },
    include_outputs_from=["retriever", "prompt_builder", "llm"]
    )

    print("\nRespuesta:")
    print(results["llm"]["replies"][0])
    print(f"{results["prompt_builder"]["prompt"]} results[prompt_builder]")
    print(f"{results["llm"]} llm")
    print("-" * 60 + "\n")
