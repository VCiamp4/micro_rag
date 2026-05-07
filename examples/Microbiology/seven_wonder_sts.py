import os
from pathlib import Path

from docling.document_converter import DocumentConverter

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder

from haystack_integrations.components.embedders.ollama import (
    OllamaTextEmbedder,
    OllamaDocumentEmbedder,
)

from haystack_integrations.components.generators.ollama import (
    OllamaGenerator,
)

from sentence_transformers import SentenceTransformer, util


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = "document.pdf"

QUESTIONS = [
    "What is the main topic of the document?",
    "What conclusions are presented?",
    "What technologies are mentioned?",
]

OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "bge-m3"

MODEL_NAME = "ministral-3:3b"

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

def split_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

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
"I cannot answer from the provided context."

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

    generator = OllamaGenerator(
        model=MODEL_NAME,
        url=OLLAMA_BASE_URL,
        generation_kwargs={
            "num_predict": 300,
            "temperature": 0.2,
        },
        timeout=450,
    )

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
        })

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

        print("=" * 80)

        print(f"\nQUESTION:\n{question}")

        print("\n--- MARKDOWN ANSWER ---\n")
        print(md_answer)

        print("\n--- TXT ANSWER ---\n")
        print(txt_answer)

        print(f"\nSTS SCORE: {sts_score:.4f}")

        print("\n")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()