
"""
supplysage_rag_retriever.py
Hybrid retrieval function for SupplySage AI chatbot and LangGraph agent.
Reads from supplysage_gold.gold_rag_embeddings and gold_rag_retrieval_index.
Import this module in the LangGraph agent notebook.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def embed_query(query_text: str, backend: str = "databricks_ai",
                endpoint: str = "databricks-bge-large-en",
                st_model=None) -> List[float]:
    """
    Embed a query string using the same backend as the corpus.
    backend: "databricks_ai" | "sentence_transformers"
    """
    if backend == "databricks_ai":
        spark = SparkSession.builder.getOrCreate()
        result = spark.sql(f"""
            SELECT ai_embed_text('{endpoint}', '{query_text.replace("'", "\'")}') AS embedding
        """).collect()[0]["embedding"]
        return list(result)

    elif backend == "sentence_transformers":
        if st_model is None:
            raise ValueError("Pass st_model when using sentence_transformers backend")
        embedding = st_model.encode([query_text], normalize_embeddings=True)[0]
        return [float(x) for x in embedding]

    else:
        raise ValueError(f"Unknown backend: {backend}")


def retrieve_evidence(
    query_text: str,
    supplier_id: Optional[str] = None,
    sku_id: Optional[str] = None,
    risk_category: Optional[str] = None,
    date_window_days: int = 30,
    severity_filter: Optional[List[str]] = None,
    top_k: int = 6,
    freshness_boost: float = 0.15,
    embedding_backend: str = "databricks_ai",
    embedding_endpoint: str = "databricks-bge-large-en",
    st_model=None
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: metadata pre-filter → semantic re-rank → freshness boost.

    Args:
        query_text:        The user question or agent query string
        supplier_id:       Filter to this supplier only (e.g. "SUP_100")
        sku_id:            Filter to this SKU only (e.g. "FOODS_1_001")
        risk_category:     Filter by risk category (e.g. "weather", "recall")
        date_window_days:  Only retrieve chunks from last N days (0 = no filter)
        severity_filter:   List of severities to include (e.g. ["critical", "high"])
        top_k:             Number of chunks to return
        freshness_boost:   How much to boost freshness_weight in final score (0-1)
        embedding_backend: "databricks_ai" | "sentence_transformers"
        embedding_endpoint: Databricks Model Serving endpoint name
        st_model:          SentenceTransformer model instance (if using ST backend)

    Returns:
        List of dicts, sorted by retrieval_score descending.
        Each dict has: chunk_id, source_name, event_date, severity, chunk_text,
                       source_url, supplier_id, freshness_weight,
                       semantic_score, retrieval_score
    """
    spark = SparkSession.builder.getOrCreate()

    # ── Phase 1: Metadata pre-filter ────────────────────────────────────────
    index = spark.table("supplysage_gold.gold_rag_retrieval_index")

    if supplier_id:
        index = index.filter(
            (F.col("supplier_id") == supplier_id) |
            F.col("supplier_id").isNull()
        )

    if sku_id:
        index = index.filter(
            (F.col("sku_id") == sku_id) |
            F.col("sku_id").isNull()
        )

    if risk_category:
        index = index.filter(F.col("risk_category") == risk_category)

    if date_window_days > 0:
        index = index.filter(
            F.col("event_date") >= F.date_sub(F.current_date(), date_window_days)
        )

    if severity_filter:
        index = index.filter(F.col("severity").isin(severity_filter))

    candidate_ids = [row["chunk_id"] for row in index.select("chunk_id").collect()]

    if not candidate_ids:
        # Widen the search — drop date and severity filters, keep entity
        print(f"[RAG] No candidates after strict filter — widening search.")
        index2 = spark.table("supplysage_gold.gold_rag_retrieval_index")
        if supplier_id:
            index2 = index2.filter(
                (F.col("supplier_id") == supplier_id) | F.col("supplier_id").isNull()
            )
        candidate_ids = [row["chunk_id"] for row in index2.select("chunk_id").collect()]

    if not candidate_ids:
        return []

    print(f"[RAG] {len(candidate_ids)} candidate chunks after metadata filter.")

    # ── Phase 2: Load candidate embeddings ──────────────────────────────────
    embeddings_tbl = spark.table("supplysage_gold.gold_rag_embeddings")

    candidates = (
        embeddings_tbl
        .filter(F.col("chunk_id").isin(candidate_ids))
        .select(
            "chunk_id", "chunk_text", "source_name", "risk_category",
            "event_date", "severity", "freshness_weight",
            "source_url", "evidence_doc_id", "supplier_id", "sku_id",
            "embedding"
        )
        .collect()
    )

    if not candidates:
        return []

    # ── Phase 3: Embed query ─────────────────────────────────────────────────
    query_embedding = embed_query(
        query_text,
        backend=embedding_backend,
        endpoint=embedding_endpoint,
        st_model=st_model
    )

    # ── Phase 4: Score = cosine_similarity × (1 + freshness_boost × freshness_weight) ──
    scored = []
    for row in candidates:
        if row["embedding"] is None:
            continue
        semantic_score = cosine_similarity(query_embedding, row["embedding"])
        fw = row["freshness_weight"] if row["freshness_weight"] is not None else 0.5
        retrieval_score = semantic_score * (1.0 + freshness_boost * fw)
        scored.append({
            "chunk_id":         row["chunk_id"],
            "source_name":      row["source_name"],
            "risk_category":    row["risk_category"],
            "event_date":       str(row["event_date"]) if row["event_date"] else None,
            "severity":         row["severity"],
            "chunk_text":       row["chunk_text"],
            "source_url":       row["source_url"],
            "evidence_doc_id":  row["evidence_doc_id"],
            "supplier_id":      row["supplier_id"],
            "sku_id":           row["sku_id"],
            "freshness_weight": fw,
            "semantic_score":   round(semantic_score, 4),
            "retrieval_score":  round(retrieval_score, 4),
        })

    # Sort by retrieval_score descending and return top_k
    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    results = scored[:top_k]

    print(f"[RAG] Returning {len(results)} chunks. Top score: {results[0]['retrieval_score']:.4f}" if results else "[RAG] No results.")
    return results


def format_evidence_for_llm(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.
    Each chunk gets a citation label [1], [2], etc.
    Returns a string to inject into the system prompt context.
    """
    if not retrieved_chunks:
        return "No external evidence retrieved for this query."

    lines = ["## Retrieved Evidence\n"]
    for i, chunk in enumerate(retrieved_chunks, 1):
        source_label = f"[{i}] {chunk.get('source_name', 'Unknown')} · {chunk.get('event_date', 'N/A')}"
        severity_label = f"Severity: {chunk.get('severity', 'N/A')}"
        text = chunk.get("chunk_text", "")
        url = chunk.get("source_url", "")
        url_str = f"URL: {url}" if url else ""

        lines.append(f"{source_label} | {severity_label}")
        lines.append(f"  {text}")
        if url_str:
            lines.append(f"  {url_str}")
        lines.append("")

    lines.append("---")
    lines.append("Cite evidence using [1], [2], etc. in your response.")
    return "\n".join(lines)
