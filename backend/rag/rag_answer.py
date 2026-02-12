import numpy as np
import requests
import faiss

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi"

def embed_query(query: str):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": MODEL_NAME,
            "prompt": query
        }
    )

    embedding = response.json()["embedding"]
    vec = np.array([embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec

def retrieve(query, index, chunks, k=4):
    qvec = embed_query(query)
    scores, ids = index.search(qvec, k)

    results = []
    for i in ids[0]:
        if i != -1:
            results.append(chunks[i])
    return results

def generate_answer(user_question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an Insurance Agency Customer Care assistant.
Use ONLY the provided context to answer.
If the answer is not in the context, say you don't have that information.

Context:
{context}

Question:
{user_question}

Answer:
"""

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]
