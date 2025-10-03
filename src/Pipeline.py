from LoadData import load_dataset
from Chunking import dataframe_to_chunks
from Embeddings import create_embeddings, get_embedder, build_index, save_embeddings
from Retrieval import search
from Generation import generate_answer
import numpy as np
import os
import pickle
import faiss


def run_pipeline(query, rebuild=False):
    """
    End-to-end RAG pipeline:
    1. Load dataset
    2. Chunk text
    3. Generate embeddings + FAISS index
    4. Retrieve top-k
    5. Generate final answer
    """
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Step 1 + 2: Load + Chunk
    if rebuild or not os.path.exists(f"{processed_dir}/chunks.pkl"):
        print("🔹 Loading dataset & creating chunks...")
        df = load_dataset()
        chunks = dataframe_to_chunks(df)
    else:
        with open(f"{processed_dir}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

    # Step 3: Embeddings + Index
    if rebuild or not os.path.exists(f"{processed_dir}/faiss.index"):
        print("🔹 Generating embeddings & building FAISS index...")
        embedder = get_embedder()
        embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=32)
        index = build_index(embeddings)
        save_embeddings(embeddings, chunks, index, processed_dir)
    else:
        print("✅ Using existing embeddings + index")
        embeddings = np.load(f"{processed_dir}/embeddings.npy")
        index = faiss.read_index(f"{processed_dir}/faiss.index")

    # Step 4: Retrieval
    print("🔹 Retrieving top documents...")
    results = search(query, top_k=3)

    # Step 5: Generation
    print("🔹 Generating final answer with LLM...")
    answer = generate_answer(query, top_k=2)

    print("\n💡 Final Answer:\n", answer)
    return answer


if __name__ == "__main__":
    # Example query
    run_pipeline("Tell me about a romantic movie set in Paris", rebuild=False)
