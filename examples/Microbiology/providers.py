from dotenv import load_dotenv
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.embedders.ollama import (
    OllamaTextEmbedder,
    OllamaDocumentEmbedder
)
from haystack_integrations.components.generators.ollama import OllamaGenerator
import hashlib
import os
from haystack.utils import Secret

load_dotenv("infrastructure/.env")


user = os.getenv('PGVECTOR_USR')
password = os.getenv('PGVECTOR_PASS')
host = os.getenv('VDB_HOST', 'localhost') 
port = os.getenv('VDB_PORT', '5433')      
db = os.getenv('PGVECTOR_DB')

conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db}"


EMBEDDING_MODEL = "bge-m3"
GENERATION_MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
SECONDS_TIMEOUT = 91218


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_document_store():
    return PgvectorDocumentStore(
        embedding_dimension=1024,
        vector_function="cosine_similarity",
        recreate_table=False,
        connection_string=Secret.from_token(conn_str)
    )


def get_doc_embedder():
    return OllamaDocumentEmbedder(
        model=EMBEDDING_MODEL,
        url=OLLAMA_BASE_URL,
        batch_size=32,
        timeout=SECONDS_TIMEOUT
    )


def get_text_embedder():
    return OllamaTextEmbedder(
        model=EMBEDDING_MODEL,
        url=OLLAMA_BASE_URL,
        timeout=SECONDS_TIMEOUT
    )


def get_generator():
    return OllamaGenerator(
        model=GENERATION_MODEL,
        url=OLLAMA_BASE_URL,
        generation_kwargs={
            "num_predict": 1000,
            "temperature": 0.3,
        },
        timeout=SECONDS_TIMEOUT
    )