import pandas as pd

# --- Chunking Function ---
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks with overlap.
    Args:
        text (str): The input text.
        chunk_size (int): Max size of each chunk.
        overlap (int): Number of characters to overlap between chunks.
    Returns:
        list: A list of chunk strings.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary if possible
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.5:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap
    return chunks


# --- Convert DataFrame Rows to Chunks ---
def dataframe_to_chunks(df, chunk_size=500, overlap=50):
    """
    Converts each movie row into text, then splits into chunks.
    Args:
        df (pd.DataFrame): Dataframe with at least Title + Plot columns.
        chunk_size (int): Max size of each chunk.
        overlap (int): Overlap between chunks.
    Returns:
        list: A list of text chunks.
    """
    all_chunks = []
    for _, row in df.iterrows():
        content = f"Title: {row['Title']}\nPlot: {row['Plot']}"
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
    return all_chunks


# --- Run directly ---
if __name__ == "__main__":
    from LoadData import load_dataset  # import your loader
    
    # Load dataset
    df = load_dataset("/Users/aks/Desktop/RAG/data/wiki_movie_plots.csv")

    # Convert to chunks
    chunks = dataframe_to_chunks(df, chunk_size=500, overlap=50)

    print(f"✅ Created {len(chunks)} chunks from {len(df)} movies")
    print("🔹 Example Chunk:\n", chunks[0][:300])
