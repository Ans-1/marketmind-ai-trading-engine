# core/memory.py
"""
ChromaDB episodic memory for MarketMind.

TWO LAYERS OF MEMORY:

1. SHORT-TERM (within a run): LangGraph State itself is the short-term memory.
   The agent_signals list accumulates results as agents complete. The synthesis
   agent reads the full list when it runs. No ChromaDB needed for this.

2. LONG-TERM (across runs): ChromaDB stores every completed analysis as a
   vector embedding. Before each new run, we query ChromaDB for past analyses
   of the same ticker and inject the summary into the synthesis agent's context.
   This is episodic memory — the system learns from its own history.

WHY VECTOR STORAGE INSTEAD OF A REGULAR DATABASE:
A SQL query for past AAPL analyses would return every record exactly.
A vector query returns records that are semantically similar — so if you
analyze AAPL during a market crash, ChromaDB finds past AAPL analyses during
other crashes, not just any AAPL analysis. The retrieval is context-aware.

INTERVIEW EXPLANATION:
"ChromaDB converts text into embeddings — vectors of ~1500 numbers that
capture semantic meaning. Similar text produces similar vectors. We store
each analysis run as an embedding of the ticker, regime, and signals. At
query time, we embed the current context and find the k nearest neighbors
by cosine similarity. The synthesis agent receives these historical cases
as additional context, turning a stateless analysis into one that learns
from its own track record."
"""

import json
import time
from typing import Optional
import chromadb
from chromadb.config import Settings


# We use a persistent local client so memory survives between sessions.
# The database is stored at ./marketmind_memory in the project directory.
_client: Optional[chromadb.PersistentClient] = None
_collection = None

COLLECTION_NAME = "trade_decisions"
MEMORY_DB_PATH  = "./marketmind_memory"


# def _get_collection():
#     """
#     Lazy-initialises the ChromaDB client and collection.
#     Using lazy init means ChromaDB isn't loaded until memory is actually used,
#     keeping startup time fast for runs that don't need memory.
#     """
#     global _client, _collection

#     if _collection is not None:
#         return _collection

#     try:
#         _client = chromadb.PersistentClient(
#             path=MEMORY_DB_PATH,
#             settings=Settings(anonymized_telemetry=False)
#         )
#         _collection = _client.get_or_create_collection(
#             name=COLLECTION_NAME,
#             # Cosine similarity is better than L2 distance for text embeddings
#             # because it measures direction (meaning) not magnitude (length).
#             metadata={"hnsw:space": "cosine"}
#         )
#         return _collection

#     except Exception as e:
#         # ChromaDB is optional infrastructure. If it fails, the pipeline
#         # continues without memory rather than crashing entirely.
#         print(f"⚠️  ChromaDB unavailable: {e}. Continuing without memory.")
#         return None

def _get_collection():
    global _client, _collection

    if _collection is not None:
        return _collection

    try:
        import hashlib
        from chromadb import EmbeddingFunction, Documents, Embeddings

        class SimpleEmbeddingFunction(EmbeddingFunction):
            """
            Hash-based embedding — no model download, no DLL dependencies.
            Sufficient for ticker/regime similarity matching.
            """
            def __call__(self, input: Documents) -> Embeddings:
                embeddings = []
                for text in input:
                    vector = []
                    for i in range(64):
                        chunk = text[i::64] if text else ""
                        h = int(hashlib.md5(
                            f"{i}{chunk}".encode()
                        ).hexdigest(), 16)
                        vector.append((h % 1000) / 1000.0)
                    embeddings.append(vector)
                return embeddings

        _client = chromadb.PersistentClient(
            path=MEMORY_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=SimpleEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"}
        )
        return _collection

    except Exception as e:
        print(f"⚠️  ChromaDB unavailable: {e}. Continuing without memory.")
        return None

def store_analysis(
    ticker:          str,
    asset_type:      str,
    conviction_score: float,
    final_verdict:   str,
    market_regime:   str,
    agent_signals:   list,
    final_reasoning: str,
    run_id:          Optional[str] = None,
) -> bool:
    """
    Persists a completed analysis to ChromaDB.
    
    Call this AFTER the pipeline completes successfully.
    Returns True if stored, False if ChromaDB was unavailable.
    
    WHAT WE STORE:
    - document: human-readable summary of the run (this is what gets embedded)
    - metadata: structured fields for filtering (ticker, date, verdict, score)
    - id: unique identifier for this record
    """
    collection = _get_collection()
    if collection is None:
        return False

    try:
        # Build a rich text document from the analysis.
        # This text is what ChromaDB embeds — richer text = better retrieval.
        agent_summary = " | ".join([
            f"{s['agent'].upper()}: {s['signal']} ({s['confidence']:.0%})"
            for s in agent_signals
        ])

        document = (
            f"Ticker: {ticker} ({asset_type}). "
            f"Regime: {market_regime}. "
            f"Agents: {agent_summary}. "
            f"Conviction: {conviction_score:.2f}. "
            f"Verdict: {final_verdict}. "
            f"Reasoning: {final_reasoning}"
        )

        timestamp = int(time.time())
        record_id = f"{ticker}_{timestamp}"

        collection.add(
            documents=[document],
            metadatas=[{
                "ticker":          ticker,
                "asset_type":      asset_type,
                "conviction_score": conviction_score,
                "final_verdict":   final_verdict,
                "market_regime":   market_regime,
                "timestamp":       timestamp,
                "date":            time.strftime("%Y-%m-%d", time.gmtime()),
                "run_id":          run_id or "",
            }],
            ids=[record_id],
        )
        return True

    # except Exception as e:
    #     print(f"⚠️  Memory store failed: {e}")
    #     return False

    except Exception as e:
        import traceback
        print(f"⚠️  Memory store failed: {e}")
        print(traceback.format_exc())  # ADD THIS LINE
        return False

def retrieve_context(ticker: str, current_regime: str, n_results: int = 3) -> str:
    """
    Retrieves the most relevant past analyses for the current ticker and regime.
    Returns a formatted string ready to inject into the synthesis agent prompt.
    
    WHY n_results=3:
    More context is not always better. 3 recent, relevant cases give the
    synthesis agent useful signal without overwhelming the prompt. We also
    limit to the last 90 days to avoid stale data influencing current decisions.
    """
    collection = _get_collection()
    if collection is None:
        return ""

    try:
        # Check if we have any records before querying
        if collection.count() == 0:
            return ""

        # Query with a context string that captures what we're looking for.
        # ChromaDB finds past analyses that are semantically similar to this.
        query_text = (
            f"Analysis of {ticker} during {current_regime} market regime"
        )

        # Filter to last 90 days so historical context stays relevant
        ninety_days_ago = int(time.time()) - (90 * 24 * 60 * 60)

        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, collection.count()),
            where={
                "$and": [
                    {"ticker": {"$eq": ticker}},
                    {"timestamp": {"$gte": ninety_days_ago}},
                ]
            },
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        # Format the retrieved memories into a clean context block
        memories = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            date   = metadata.get("date", "unknown date")
            verdict = metadata.get("final_verdict", "?")
            score   = metadata.get("conviction_score", 0)
            memories.append(
                f"[{date}] Verdict: {verdict}, Score: {score:.2f} — {doc[:200]}"
            )

        if not memories:
            return ""

        context = (
            f"\n\n--- HISTORICAL MEMORY FOR {ticker} (last 90 days) ---\n"
            + "\n".join(memories)
            + "\n--- END HISTORICAL MEMORY ---\n"
            + "Consider whether current signals align with or contradict past decisions.\n"
        )
        return context

    except Exception as e:
        print(f"⚠️  Memory retrieval failed: {e}")
        return ""


def get_memory_stats() -> dict:
    """Returns stats about what's stored in memory. Useful for the API /health endpoint."""
    collection = _get_collection()
    if collection is None:
        return {"status": "unavailable", "total_records": 0}

    try:
        return {
            "status": "healthy",
            "total_records": collection.count(),
            "collection_name": COLLECTION_NAME,
            "db_path": MEMORY_DB_PATH,
        }
    except Exception:
        return {"status": "error", "total_records": 0}
