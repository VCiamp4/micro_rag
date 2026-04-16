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
from providers import *


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
