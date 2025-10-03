import os
import pandas as pd

def load_dataset(path="/Users/aks/Desktop/RAG/data/wiki_movie_plots.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please put CSV file in data/ folder.")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    return df

# Call the function
if __name__ == "__main__":
    df = load_dataset()
    print(df.head())   # Show first 5 rows
