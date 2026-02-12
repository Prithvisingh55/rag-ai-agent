import json
import numpy as np
import faiss
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "phi"

def embed_texts(texts):
    vectors = []

    for text in texts:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": EMBED_MODEL,
                "prompt": text
            }
        )

        embedding = response.json()["embedding"]
        vectors.append(embedding)

    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr

def build_and_save_index(chunks, index_path, meta_path):
    vectors = embed_texts(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

def load_index(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["chunks"]
