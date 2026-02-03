from src.rag.retrieval import retrieve
from src.rag.llm import call_llm

def build_prompt(query, contexts):
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text'][:400]}"
        for c in contexts
    )

    return f"""
You are a compliance assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say so.

Context:
{context_block}

Question:
{query}

Answer clearly and cite sources.
""".strip()


def chat(query):
    contexts = sorted(retrieve(query), key=lambda x: x["score"], reverse=True)[:2]



    if not contexts:
        return {
            "answer": "No relevant information found in the knowledge base.",
            "sources": []
        }

    # IMPORTANT: limit context size
    contexts = sorted(contexts, key=lambda x: x["score"], reverse=True)[:4]

    prompt = build_prompt(query, contexts)

    answer = call_llm(prompt)


    return {
        "answer": answer,
        "sources": list(set(c["source"] for c in contexts))
    }

if __name__ == "__main__":
    q = input("Ask a question: ")
    response = chat(q)
    print("\nAnswer:\n", response["answer"])
    print("\nSources:\n", response["sources"])
