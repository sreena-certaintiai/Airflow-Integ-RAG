# db_manager.py

import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration Constants ---
DB_PATH = "../vector_db"
COLLECTION_NAME = "pdf_docs_mini_lm"
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Singleton instances ---
# We initialize these once and reuse them to save resources.
_client = chromadb.PersistentClient(path=DB_PATH)
_embedding_model = SentenceTransformer(MODEL_NAME)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def get_collection():
    """Returns the singleton ChromaDB collection instance."""
    return _collection

def get_embedding_model():
    """Returns the singleton SentenceTransformer model instance."""
    return _embedding_model