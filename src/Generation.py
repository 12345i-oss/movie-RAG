
import torch
from transformers import pipeline
from Retrieval import search, get_embedder

# 🔹 Load Hugging Face LLM
# Replace UL2 (20B params) with flan-t5-large (~780M params, fits on 8–12GB VRAM)
hf_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",   # options: flan-t5-large / flan-t5-xl / flan-t5-xxl
    device=0 if torch.cuda.is_available() else -1
)

def generate_answer(query, top_k=5):
    # Run retrieval
    embedder = get_embedder()
    results = search(query, top_k=top_k)
    
    print(f"\n🔍 Query: {query}")
    for i, (chunk, distance) in enumerate(results, 1):
        print(f"{i}. {chunk[:120]}... (distance={distance:.4f})")

    # Build context
    context_chunks = [chunk for chunk, _ in results]
    context = "\n".join(context_chunks)
    
    # Stronger & structured prompt
    prompt = f"""
You are a movie expert. Read the following movie descriptions
and answer the question in a structured list.

MOVIE DESCRIPTIONS:
{context}

QUESTION: {query}

INSTRUCTIONS:
- ONLY use the information from the movie descriptions above.
- Write a clear, detailed answer about the romantic movies set in Paris.
- Format the answer as a numbered list with movie title and summary.
- Do not stop early. Cover all provided movies.

ANSWER (follow the format strictly):
1. "Movie Title" – summary
2. "Movie Title" – summary
"""
    
    print(f"\n📝 Prompt length: {len(prompt)} characters")
    
    # Generate with Hugging Face model
    out = hf_generator(
        prompt.strip(),
        max_new_tokens=500,
        num_beams=5,              # beam search for quality
        repetition_penalty=1.2,   # discourage short outputs
        early_stopping=False
    )
    
    return out[0]['generated_text'].strip()

if __name__ == "__main__":
    query = "Tell me about a romantic movie set in Paris"
    answer = generate_answer(query, top_k=2)
    print("\n💡 Final Answer:\n", answer)



