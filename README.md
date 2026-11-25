# Hybrid RAG (BM25 + Dense + RRF + Cross-Encoder Re-ranking)

An **advanced Hybrid RAG** system. It is the second, distinct approach for the assignment, contrasting with the Naive RAG repository.

## Approach
| Stage | Method |
|-------|--------|
| Load + chunk | 100-word chunks, 25-word overlap, from `data/*.txt` |
| Sparse retrieval | **BM25** (`rank-bm25`) - exact keyword matching |
| Dense retrieval | `all-MiniLM-L6-v2` embeddings + **FAISS** - semantic matching |
| Fusion | **Reciprocal Rank Fusion (RRF)**: `score = sum(1 / (60 + rank))` |
| Re-ranking | Cross-encoder `ms-marco-MiniLM-L-6-v2` re-scores the fused candidates |
| Generation | Local `flan-t5-base` (default) or OpenAI if `OPENAI_API_KEY` is set |

```
Question -> [BM25 || Dense] -> RRF merge -> Cross-encoder re-rank -> top-k -> LLM -> Answer + sources
```

The program prints the ranking at every stage (BM25, Dense, RRF, Reranked) so you can see how documents are retrieved and reordered.

## Dependencies
Python 3.9+; see `requirements.txt` (sentence-transformers, faiss-cpu, rank-bm25, transformers, torch, numpy).

## Setup
```bash
git clone <this-repo-url>
cd hybrid-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/hybrid_rag.py --question "What does CALCULATE do in DAX?"
python src/hybrid_rag.py                      # interactive mode
export OPENAI_API_KEY=sk-...                  # optional, for a stronger LLM
```
Add your own `.txt` files to `data/` to use another document collection.

## Sample questions
- What is the F1-score and when is it used?          (keyword-heavy: BM25 helps)
- How do I stop a model from memorising the training data?   (semantic: dense helps)
- What is the Adam optimiser?
- Difference between drill-down and drill-through?

## Why hybrid?
BM25 finds exact terms (e.g., "CALCULATE", "FAISS") but misses paraphrases; dense retrieval understands meaning but may miss rare tokens. RRF merges both, and the cross-encoder reads the query and passage together for a more accurate final ordering.

## Example output
(Paste your own real run output / screenshot here before submitting.)
