# 🎬 CineRAG – Local Retrieval-Augmented Generation App

CineRAG is a **Retrieval-Augmented Generation (RAG)** system built in Python.  
It combines **FAISS** for fast semantic search with a **Hugging Face LLM** for local answer generation.  
Originally demoed on movie descriptions, but you can adapt it to **any text corpus**.

---

## 📂 Dataset  
We use the **Wikipedia Movie Plots Dataset** from Kaggle:  
👉 [Wikipedia Movie Plots (Deduplicated)](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots?select=wiki_movie_plots_deduped.csv)  

The dataset contains **34,886 English-language movie plot summaries** extracted from Wikipedia, including:  
- Title  
- Genre  
- Director  
- Actors  
- Plot summary  

---

## ⚡ Pipeline Overview  

1. **Data Preprocessing & Embeddings**  
   - Load movie plot dataset  
   - Split long plots into **chunks**  
   - Encode chunks into **vector embeddings**  

2. **Indexing with FAISS**  
   - Store embeddings in a FAISS index (`index.faiss`)  
   - Keep track of original chunks (`chunks.npy`)  

3. **Retrieval**  
   - User query → embedded into vector  
   - FAISS retrieves **top-k most relevant chunks**  

4. **Answer Generation**  
   - Retrieved chunks are assembled into a **context prompt**  
   - Prompt is passed to:  
     - **Hugging Face FLAN-T5** (local, open-source)  
     - Or **OpenAI GPT models** (if API key is available)  
   - Generates structured, context-aware answers  

---

## 🚀 Features
- **Local vector search** with FAISS (fast similarity search across 200k+ chunks).
- **Embeddings** via Hugging Face (customizable).
- **Question answering** with Hugging Face LLMs (FLAN-T5, Mistral, LLaMA, etc.).
- **No API keys required** (runs entirely offline).
- Modular design: `Retrieval.py` for searching, `Generation.py` for answering.

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/cinerag.git
cd cinerag
