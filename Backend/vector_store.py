# vector_store.py
import faiss
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the embedding model (Requirement: The Duplicate Hunter)
# all-MiniLM-L6-v2 produces 384-dimensional vectors used for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2') 
INDEX_PATH = "docs.index"
HASH_PATH = "hash.json"

def get_faiss_index():
    """
    Helper to load the FAISS index or create a fresh one if missing.
    Ensures the system doesn't crash after a reset or on first deployment.
    """
    dimension = 384
    if os.path.exists(INDEX_PATH):
        try:
            return faiss.read_index(INDEX_PATH)
        except Exception as e:
            print(f"Index corrupted or empty, recreating: {e}")
    
    # Create a fresh index using Inner Product (for Cosine Similarity)
    index = faiss.IndexFlatIP(dimension)
    faiss.write_index(index, INDEX_PATH)
    return index

def search_duplicate(text: str, img_hash: str):
    """
    Requirement: The Duplicate Hunter. 
    Checks for exact visual matches via pHash and semantic matches via FAISS vectors.
    """
    
    # 1. Check Image Hash Database (Visual Match)
    if os.path.exists(HASH_PATH):
        try:
            with open(HASH_PATH, "r") as f:
                hashes = json.load(f)
                if img_hash and img_hash in hashes:
                    # Exact match found in the hash storage
                    return True, 1.0 
        except Exception:
            pass

    # 2. Check Vector Similarity (Semantic Match)
    index = get_faiss_index()
    if index.ntotal == 0:
        return False, 0.0

    # Convert text to vector and normalize for Cosine Similarity
    vector = model.encode([text])
    faiss.normalize_L2(vector) 
    
    # Search for the single nearest neighbor
    # D = Distances (Similarity Scores), I = Indices
    D, I = index.search(vector, 1)
    
    # Threshold for duplicate detection: 0.90 (90% similarity)
    if D[0][0] >= 0.90:
        return True, float(D[0][0])
    
    return False, 0.0

def add_to_index(text: str, img_hash: str):
    """
    Stores the document's vector and image hash to the local database.
    This allows future uploads to be compared against this document.
    """
    
    # Update Hash JSON (Visual fingerprinting)
    hashes = {}
    if os.path.exists(HASH_PATH):
        try:
            with open(HASH_PATH, "r") as f:
                hashes = json.load(f)
        except Exception:
            hashes = {}
    
    if img_hash:
        hashes[img_hash] = True
        with open(HASH_PATH, "w") as f:
            json.dump(hashes, f)

    # Update FAISS Index (Semantic fingerprinting)
    vector = model.encode([text])
    faiss.normalize_L2(vector)
    
    index = get_faiss_index()
    index.add(vector)
    
    # Save the updated index back to the local file
    faiss.write_index(index, INDEX_PATH)