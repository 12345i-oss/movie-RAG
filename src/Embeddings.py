import os
import pickle
import numpy as np

import faiss


from sentence_transformers import SentenceTransformer
from Chunking import dataframe_to_chunks
from LoadData import load_dataset

# =============================
# 1. Embedding Model
# =============================
def get_embedder():
    """Load SentenceTransformer model (Hugging Face online -> local fallback)."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    local_model_path = "/Users/aks/Desktop/RAG/models/all-MiniLM-L6-v2"

    try:
        print(f"🔹 Trying to load model from Hugging Face Hub: {model_name}")
        return SentenceTransformer(model_name)   # downloads if not cached
    except Exception as e:
        print(f"⚠️ Failed to load from Hugging Face Hub: {e}")
        if os.path.exists(local_model_path):
            print(f"✅ Falling back to local model at {local_model_path}")
            return SentenceTransformer(local_model_path)
        else:
            raise RuntimeError(
                f"❌ Could not load {model_name} from Hugging Face Hub "
                f"and no local copy found at {local_model_path}"
            )



# =============================
# 2. Build FAISS Index
# =============================
def build_index(embeddings):
    """Create FAISS index from embeddings"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index

# =============================
# 3. Save & Load Utilities
# =============================
def save_embeddings(embeddings, chunks, index, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Save numpy embeddings
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    
    # Save chunks list
    with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    
    print(f"✅ Saved embeddings, chunks, and FAISS index to {out_dir}")

def load_embeddings(out_dir="data/processed"):
    embeddings = np.load(os.path.join(out_dir, "embeddings.npy"))
    
    with open(os.path.join(out_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    
    index = faiss.read_index(os.path.join(out_dir, "faiss.index"))
    
    print(f"✅ Loaded embeddings, chunks, and FAISS index from {out_dir}")
    return embeddings, chunks, index

# =============================
# 4. Main Execution
# =============================
if __name__ == "__main__":
    # Load dataset
    df = load_dataset("/Users/aks/Desktop/RAG/data/wiki_movie_plots.csv")
    
    # Convert to chunks
    print("🔹 Chunking dataset...")
    chunks = dataframe_to_chunks(df, chunk_size=500, overlap=50)
    
    # Generate embeddings
    print("🔹 Generating embeddings...")
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=32)
    
    # Build FAISS index
    print("🔹 Building FAISS index...")
    index = build_index(embeddings)
    
    # Save all outputs
    save_embeddings(embeddings, chunks, index)