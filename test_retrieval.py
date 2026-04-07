from retriever import load_vectorstore, build_bm25_index, retrieve
print("Script started")
# ─── Change your questions here ──────────────────────────
QUESTIONS = [
    "Who is the king of Hybern?",
    "Who is Morrigan?",
    "How did Mor save Feyre",
    "Did Feyre and Rhys dance at the Starfall?",
    "Who is Feyre's mate?",
]
# ─────────────────────────────────────────────────────────

def main():
    print("Loading retrieval components...")
    vs = load_vectorstore()
    bm25, docs, metas = build_bm25_index(vs)
    print("Ready.\n")

    for q in QUESTIONS:
        print(f"QUESTION: {q}")
        print("-" * 60)
        results = retrieve(q, vs, bm25, docs, metas)
        for i, r in enumerate(results):
            page = r["meta"].get("page", "?")
            print(f"  Chunk {i+1} (page {page}):")
            print(f"  {r['doc'][:300]}")
            print()
        print("=" * 60)
        print()

if __name__ == "__main__":
    main()