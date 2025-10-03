import numpy as np
import faiss
import os

def build_faiss_index(embeddings, out_dir="data/processed"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    print(f"✅ FAISS index built & saved at {out_dir}/faiss.index")
    return index

if __name__ == "__main__":
    embeddings = np.load("data/processed/embeddings.npy")
    build_faiss_index(embeddings)
