import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load FAISS index, embeddings, and chunks
def load_index(processed_dir="data/processed"):
    embeddings = np.load(f"{processed_dir}/embeddings.npy")
    with open(f"{processed_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(f"{processed_dir}/faiss.index")
    print(f"✅ Loaded FAISS index with {len(chunks)} chunks")
    return embeddings, chunks, index


def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def search(query, top_k=5, processed_dir="data/processed"):
    embeddings, chunks, index = load_index(processed_dir)
    embedder = get_embedder()

    # Encode query
    query_vec = embedder.encode([query]).astype("float32")

    # Search top-k
    distances, indices = index.search(query_vec, top_k)

    print(f"\n🔍 Query: {query}\n")
    results = []
    for rank, idx in enumerate(indices[0]):
        snippet = chunks[idx][:300]  # first 300 chars
        score = distances[0][rank]
        print(f"{rank+1}. {snippet}... (distance={score:.4f})")
        results.append((chunks[idx], score))
    return results


if __name__ == "__main__":
    search("romantic movie set in Paris", top_k=3)
