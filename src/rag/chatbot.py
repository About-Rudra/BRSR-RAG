from src.rag.retrieval import retrieve

def build_prompt(query, contexts):
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in contexts
    )

    return f"""
You are a compliance assistant.
Answer ONLY using the provided context.

Context:
{context_block}

Question:
{query}

Answer clearly and cite sources.
"""

def chat(query):
    contexts = retrieve(query)

    if not contexts:
        return "No relevant information found in the knowledge base."

    prompt = build_prompt(query, contexts)

    # Placeholder for LLM call
    # Replace later with Gemini / Ollama / anything
    return {
        "prompt_sent_to_llm": prompt,
        "sources": list(set(c["source"] for c in contexts))
    }

if __name__ == "__main__":
    q = input("Ask a question: ")
    response = chat(q)
    print(response)
