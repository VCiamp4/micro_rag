import os
from pathlib import Path

from docling.document_converter import DocumentConverter

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter

from haystack_integrations.components.embedders.ollama import (
    OllamaTextEmbedder,
    OllamaDocumentEmbedder,
)

from haystack_integrations.components.generators.ollama import (
    OllamaGenerator,
)

from sentence_transformers import SentenceTransformer, util

from providers import get_generator


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).parent
# PDF_PATH = BASE_DIR / "pdf_para_prueba" / "funciones_de_costo_y_demas.pdf" # PDF SIMPLE

PDF_PATH = BASE_DIR / "pdf_para_prueba" / "Informe_Trabajo_Practico_1.pdf" # PDF COMPLEJO

"""
QUESTIONS = [
    "¿Qué es una función de activación?",
    "¿Cuáles son las funciones de activación?",
    "¿Qué es un optimizador?",
    "¿Por qué no es ideal usar Entropía Cruzada Binaria con tanh", # Pregunta que tiene que responder mirando a una tabla
    "¿Qué es un grafo de conocimiento?" # Pregunta que no debería responder
]
"""

QUESTIONS = [
    "¿Cuál fue la principal optimización introducida en quadratic3.c para operaciones float?",
    "¿Por qué el uso de funciones como sqrtf y powf mejoró el rendimiento?",
    "¿Qué mejora de rendimiento obtuvo float en quadratic3.c dentro del clúster?",
    "¿Por qué float puede mejorar el aprovechamiento de caché respecto de double?",
    "¿Qué resultado inesperado se observó en el equipo hogareño al comparar float y double en quadratic2.c?",
    "¿Qué característica del CPU fue mencionada como posible explicación del buen rendimiento de double en el equipo hogareño?",
    "¿Qué ventajas podría aportar utilizar Linux Mint Xia 22.1 en un entorno de experimentación científica?",
    "¿Cómo podrían influir las características del procesador AMD Ryzen 5 7530U en el rendimiento de los experimentos realizados?",
    "¿Por qué disponer de varios niveles de caché puede afectar los tiempos de ejecución de un algoritmo?",
    "¿Qué diferencias estructurales existían entre el clúster remoto y el equipo hogareño utilizado?",
    "¿Por qué los autores decidieron no utilizar el flag -Ofast a pesar de obtener mejores tiempos?",
]


OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "bge-m3"

MODEL_NAME = "qwen3.5:0.8b"

STS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# PDF -> MD/TXT USING DOCLING
# ============================================================

def convert_pdf(pdf_path):
    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    md_content = result.document.export_to_markdown()
    txt_content = result.document.export_to_text()

    pdf_stem = Path(pdf_path).stem

    md_path = f"{pdf_stem}.md"
    txt_path = f"{pdf_stem}.txt"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    return md_content, txt_content


# ============================================================
# SIMPLE CHUNKER
# ============================================================

splitter = DocumentSplitter(
    split_by="sentence",
    split_length=10,
    split_overlap=2
)

def split_text(text):

    doc = Document(content=text)

    result = splitter.run(documents=[doc])

    chunks = [
        d.content
        for d in result["documents"]
    ]

    return chunks

# ============================================================
# BUILD HAYSTACK RAG PIPELINE
# ============================================================

def build_rag_pipeline(chunks):

    document_store = InMemoryDocumentStore()

    docs = [
        Document(content=chunk)
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # DOCUMENT EMBEDDINGS
    # --------------------------------------------------------

    doc_embedder = OllamaDocumentEmbedder(
        model=EMBEDDING_MODEL,
        url=OLLAMA_BASE_URL,
        batch_size=32,
    )

    docs_with_embeddings = doc_embedder.run(docs)

    document_store.write_documents(
        docs_with_embeddings["documents"]
    )

    # --------------------------------------------------------
    # QUERY EMBEDDER
    # --------------------------------------------------------

    text_embedder = OllamaTextEmbedder(
        model=EMBEDDING_MODEL,
        url=OLLAMA_BASE_URL,
    )

    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=3,
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    template = """
You must answer ONLY using the provided context.

If the answer is not present in the context, say:
"No puedo responder esa pregunta"

Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question:
{{question}}

Answer:
"""

    prompt_builder = PromptBuilder(template=template)

    # --------------------------------------------------------
    # GENERATOR
    # --------------------------------------------------------
    
    generator = get_generator()
    
    """
    generator = OllamaGenerator(
        model=MODEL_NAME,
        url=OLLAMA_BASE_URL,
        generation_kwargs={
            "num_predict": 1000,
            "temperature": 0.2,
        },
        timeout=91218,
    )
    """
    

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    rag_pipeline = Pipeline()

    rag_pipeline.add_component(
        "text_embedder",
        text_embedder
    )

    rag_pipeline.add_component(
        "retriever",
        retriever
    )

    rag_pipeline.add_component(
        "prompt_builder",
        prompt_builder
    )

    rag_pipeline.add_component(
        "llm",
        generator
    )

    rag_pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding"
    )

    rag_pipeline.connect(
        "retriever",
        "prompt_builder.documents"
    )

    rag_pipeline.connect(
        "prompt_builder",
        "llm"
    )

    return rag_pipeline


# ============================================================
# ASK QUESTIONS
# ============================================================

def ask_questions(pipeline, questions):

    answers = []

    for question in questions:

        results = pipeline.run({
            "text_embedder": {
                "text": question
            },
            "prompt_builder": {
                "question": question
            }
        },
        include_outputs_from=["retriever", "prompt_builder", "llm"])

        answer = results["llm"]["replies"][0]

        answers.append({
            "question": question,
            "answer": answer,
        })

    return answers


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== CONVERTING PDF ===\n")

    md_content, txt_content = convert_pdf(PDF_PATH)

    # --------------------------------------------------------
    # SPLIT DOCUMENTS
    # --------------------------------------------------------

    md_chunks = split_text(md_content)
    txt_chunks = split_text(txt_content)

    # --------------------------------------------------------
    # BUILD RAG PIPELINES
    # --------------------------------------------------------

    print("\n=== BUILDING MD PIPELINE ===\n")

    md_pipeline = build_rag_pipeline(md_chunks)

    print("\n=== BUILDING TXT PIPELINE ===\n")

    txt_pipeline = build_rag_pipeline(txt_chunks)

    # --------------------------------------------------------
    # RUN QUESTIONS
    # --------------------------------------------------------

    print("\n=== GENERATING ANSWERS FROM MD ===\n")

    md_answers = ask_questions(
        md_pipeline,
        QUESTIONS
    )

    print("\n=== GENERATING ANSWERS FROM TXT ===\n")

    txt_answers = ask_questions(
        txt_pipeline,
        QUESTIONS
    )

    # --------------------------------------------------------
    # STS MODEL
    # --------------------------------------------------------

    sts_model = SentenceTransformer(STS_MODEL)

    print("\n=== STS COMPARISON ===\n")

    for md_result, txt_result in zip(md_answers, txt_answers):

        question = md_result["question"]

        md_answer = md_result["answer"]

        txt_answer = txt_result["answer"]

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        md_embedding = sts_model.encode(
            md_answer,
            convert_to_tensor=True
        )

        txt_embedding = sts_model.encode(
            txt_answer,
            convert_to_tensor=True
        )

        # ----------------------------------------------------
        # STS SCORE
        # ----------------------------------------------------

        sts_score = util.cos_sim(
            md_embedding,
            txt_embedding
        ).item()

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------
        

        output_file = "sts_results.txt"

        with open(output_file, "w", encoding="utf-8") as f:

            for md_result, txt_result in zip(md_answers, txt_answers):

                question = md_result["question"]

                md_answer = md_result["answer"]

                txt_answer = txt_result["answer"]

                # ----------------------------------------------------
                # EMBEDDINGS
                # ----------------------------------------------------

                md_embedding = sts_model.encode(
                    md_answer,
                    convert_to_tensor=True
                )

                txt_embedding = sts_model.encode(
                    txt_answer,
                    convert_to_tensor=True
                )

                # ----------------------------------------------------
                # STS SCORE
                # ----------------------------------------------------

                sts_score = util.cos_sim(
                    md_embedding,
                    txt_embedding
                ).item()

                # ----------------------------------------------------
                # WRITE TO FILE
                # ----------------------------------------------------

                f.write("=" * 80 + "\n")

                f.write(f"\nQUESTION:\n{question}\n")

                f.write("\n--- MARKDOWN ANSWER ---\n")
                f.write(md_answer + "\n")

                f.write("\n--- TXT ANSWER ---\n")
                f.write(txt_answer + "\n")

                f.write(f"\nSTS SCORE: {sts_score:.4f}\n")

                f.write("\n")

    print(f"\nResults saved to: {output_file}")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()