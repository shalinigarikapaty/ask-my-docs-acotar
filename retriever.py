from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

# ─── Configuration ───────────────────────────────────────
EMBEDDING_MODEL   = "amazon.titan-embed-text-v2:0"
REGION            = "us-east-1"
CHROMA_DIR        = "./chroma_db"
TOP_K_RETRIEVAL   = 20   # candidates from each retriever
TOP_K_FINAL       = 5    # final chunks after reranking
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── Load vector store ────────────────────────────────────
def load_vectorstore():
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL,
        region_name=REGION
    )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

# ─── Build BM25 index from ChromaDB chunks ────────────────
def build_bm25_index(vectorstore):
    print("Building BM25 index...")
    results = vectorstore.get()
    documents = results["documents"]
    metadatas = results["metadatas"]
    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index built over {len(documents)} chunks")
    return bm25, documents, metadatas

# ─── Reciprocal Rank Fusion ───────────────────────────────
def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    scores = {}

    for rank, (doc, meta) in enumerate(bm25_results):
        key = doc[:100]
        scores[key] = scores.get(key, {"score": 0, "doc": doc, "meta": meta})
        scores[key]["score"] += 1 / (rank + k)

    for rank, (doc, meta) in enumerate(vector_results):
        key = doc[:100]
        scores[key] = scores.get(key, {"score": 0, "doc": doc, "meta": meta})
        scores[key]["score"] += 1 / (rank + k)

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return ranked

# ─── Hybrid retriever ─────────────────────────────────────
def hybrid_retrieve(query, vectorstore, bm25, documents, metadatas, top_k=TOP_K_RETRIEVAL):

    # BM25 retrieval
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [(documents[i], metadatas[i]) for i in top_bm25_indices]

    # Vector retrieval
    vector_docs = vectorstore.similarity_search(query, k=top_k)
    vector_results = [(doc.page_content, doc.metadata) for doc in vector_docs]

    # Combine with RRF
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    return fused[:top_k]

# ─── Cross-encoder reranker ───────────────────────────────
def rerank(query, candidates, top_k=TOP_K_FINAL):
    print(f"Reranking {len(candidates)} candidates...")
    reranker = CrossEncoder(RERANKER_MODEL)
    pairs = [[query, c["doc"]] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )
    top = ranked[:top_k]
    return [c for _, c in top], [float(s) for s, _ in top]

# ─── Confidence level from top reranker score ─────────────
def score_to_confidence(scores: list) -> str:
    if not scores:
        return "none"
    top = scores[0]
    if top < 0.1:
        return "low"
    if top < 0.3:
        return "moderate"
    return "high"

# ─── Full pipeline ────────────────────────────────────────
def retrieve(query, vectorstore, bm25, documents, metadatas):
    candidates          = hybrid_retrieve(query, vectorstore, bm25, documents, metadatas)
    top_chunks, scores  = rerank(query, candidates)
    confidence          = score_to_confidence(scores)
    print(f"Confidence: {confidence} (top score: {scores[0]:.4f})" if scores else "Confidence: none")
    return top_chunks, confidence