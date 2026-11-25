"""
Hybrid RAG with re-ranking:
  BM25 (sparse)  +  FAISS dense retrieval  ->  Reciprocal Rank Fusion  ->  cross-encoder re-rank  ->  LLM
"""
import argparse
import glob
import os
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "google/flan-t5-base"


def tokenize(text):
    return re.findall(r"\w+", text.lower())


# ------------------------------------------------------------ load + chunk
def load_and_chunk(folder, size=100, overlap=25):
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        step = size - overlap
        for i, s in enumerate(range(0, max(len(words) - overlap, 1), step)):
            chunks.append({"source": os.path.basename(path), "chunk_id": i,
                           "text": " ".join(words[s:s + size])})
    print(f"[load+chunk] {len(chunks)} chunks")
    return chunks


# ------------------------------------------------------------ indexes
class HybridRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        # sparse index
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        # dense index
        self.embedder = SentenceTransformer(EMBED_MODEL)
        vecs = self.embedder.encode(texts, normalize_embeddings=True,
                                    show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        # re-ranker
        self.reranker = CrossEncoder(RERANK_MODEL)
        print(f"[index] BM25 + FAISS built over {len(chunks)} chunks")

    def sparse(self, q, n):
        scores = self.bm25.get_scores(tokenize(q))
        return list(np.argsort(scores)[::-1][:n])

    def dense(self, q, n):
        v = self.embedder.encode([q], normalize_embeddings=True).astype("float32")
        _, ids = self.index.search(v, n)
        return list(ids[0])

    @staticmethod
    def rrf(rank_lists, k=60):
        """Reciprocal Rank Fusion: score = sum 1/(k + rank)."""
        fused = {}
        for ranks in rank_lists:
            for r, doc_id in enumerate(ranks):
                fused[doc_id] = fused.get(doc_id, 0) + 1.0 / (k + r + 1)
        return sorted(fused, key=fused.get, reverse=True)

    def search(self, q, n_candidates=8, top_k=3, verbose=True):
        sp, de = self.sparse(q, n_candidates), self.dense(q, n_candidates)
        fused = self.rrf([sp, de])[:n_candidates]
        scores = self.reranker.predict([(q, self.chunks[i]["text"]) for i in fused])
        reranked = [fused[i] for i in np.argsort(scores)[::-1][:top_k]]
        if verbose:
            show = lambda name, ids: print(f"   {name:<10}", [f"{self.chunks[i]['source'][:-4]}#{self.chunks[i]['chunk_id']}" for i in ids[:4]])
            print("[retrieve]")
            show("BM25", sp); show("Dense", de); show("RRF", fused); show("Reranked", reranked)
        return [self.chunks[i] for i in reranked]


# ------------------------------------------------------------ generation
def build_prompt(q, hits):
    ctx = "\n\n".join(f"[{i+1}] ({h['source']}) {h['text']}" for i, h in enumerate(hits))
    return ("Answer the question using ONLY the context below. "
            "If the answer is not in the context, say \"I don't know\".\n\n"
            f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")


def generate(prompt):
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}], temperature=0)
        return r.choices[0].message.content.strip()
    from transformers import pipeline
    if not hasattr(generate, "pipe"):
        generate.pipe = pipeline("text2text-generation", model=GEN_MODEL)
    return generate.pipe(prompt, max_new_tokens=150, truncation=True)[0]["generated_text"].strip()


def main():
    ap = argparse.ArgumentParser(description="Hybrid RAG demo")
    ap.add_argument("--data", default="data")
    ap.add_argument("--question")
    ap.add_argument("--top_k", type=int, default=3)
    a = ap.parse_args()

    retriever = HybridRetriever(load_and_chunk(a.data))

    def ask(q):
        hits = retriever.search(q, top_k=a.top_k)
        print("\nSOURCES:", ", ".join(sorted({h["source"] for h in hits})))
        print("ANSWER:", generate(build_prompt(q, hits)))

    if a.question:
        return ask(a.question)
    print("\nAsk questions (type 'exit' to quit).")
    while True:
        q = input("\nQ> ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        ask(q)


if __name__ == "__main__":
    main()
