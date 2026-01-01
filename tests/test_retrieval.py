from src.rag.retrieval import retrieve

queries = [
    "What is BRSR value chain disclosure?",
    "What are obligations under battery waste EPR?",
    "What is environmental compensation under PWM rules?",
    "What is the compliance procedure under CCTS?",
]

for q in queries:
    print("\n" + "=" * 100)
    print("QUERY:", q)

    results = retrieve(q, top_k=5)

    if not results:
        print("❌ NO RESULTS")
        continue

    for r in results:
        print("\nScore:", round(r["score"], 3))
        print("Source:", r["source"])
        print(r["text"][:400])
