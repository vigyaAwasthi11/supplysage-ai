# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 25 — RAG Embedding Pipeline
# MAGIC
# MAGIC **Input:**  `supplysage_gold.gold_rag_evidence_chunks`
# MAGIC **Outputs:**
# MAGIC   - `supplysage_gold.gold_rag_embeddings`  — chunk text + embedding vectors (1536-dim)
# MAGIC   - `supplysage_gold.gold_rag_retrieval_index` — metadata index for hybrid retrieval
# MAGIC   - Databricks Vector Search index registered on `gold_rag_embeddings` (if VS enabled)
# MAGIC
# MAGIC **Embedding strategy:**
# MAGIC   This notebook supports TWO embedding backends. Choose one by setting EMBEDDING_BACKEND below.
# MAGIC   - `databricks_ai`   → uses `ai_embed_text()` SQL function (Databricks DBRX / BGE via
# MAGIC                         Databricks AI Functions). Requires DBR 14.3+ or Unity Catalog AI.
# MAGIC   - `sentence_transformers` → uses `sentence-transformers` library locally on the driver.
# MAGIC                         Works on any DBR. Slower but fully self-contained. Good for dev.
# MAGIC
# MAGIC **Chunking note:**
# MAGIC   Notebook 24 already wrote chunk_text to gold_rag_evidence_chunks.
# MAGIC   This notebook reads those chunks, generates embeddings, and writes a separate
# MAGIC   embeddings table. We keep them separate so chunk metadata can be refreshed
# MAGIC   without re-embedding everything, and so embeddings can be re-generated with a
# MAGIC   different model without touching the chunk table.
# MAGIC
# MAGIC **Run after:** Notebook 24 (gold_rag_evidence_chunks must exist and be non-empty)
# MAGIC **Run before:** Notebook 25b (gold_chat_context_snapshots) and LangGraph agent setup

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Notebook 25 — Clean RAG Embedding Pipeline
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    StringType,
    StructType,
    StructField,
    IntegerType,
    DateType,
    TimestampType
)
from datetime import datetime
import pandas as pd
import subprocess

GOLD_SCHEMA = "supplysage_gold"

SOURCE_CHUNKS_TABLE = f"{GOLD_SCHEMA}.gold_rag_evidence_chunks"
EMBEDDINGS_TABLE = f"{GOLD_SCHEMA}.gold_rag_embeddings"
RETRIEVAL_INDEX_TABLE = f"{GOLD_SCHEMA}.gold_rag_retrieval_index"

# Use sentence_transformers for now.
# This avoids Databricks AI endpoint / ai_embed_text dependency.
EMBEDDING_BACKEND = "sentence_transformers"

# Fast demo model.
# Dimension = 384.
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Keep batch size moderate on Databricks driver.
EMBEDDING_BATCH_SIZE = 64

# This will be overwritten after model loads.
EMBEDDING_DIM = 384

# Skip Vector Search for now.
# First get Delta embeddings + retrieval index working.
ENABLE_VECTOR_SEARCH = False

print("Config loaded")
print(f"SOURCE_CHUNKS_TABLE = {SOURCE_CHUNKS_TABLE}")
print(f"EMBEDDINGS_TABLE = {EMBEDDINGS_TABLE}")
print(f"RETRIEVAL_INDEX_TABLE = {RETRIEVAL_INDEX_TABLE}")
print(f"EMBEDDING_BACKEND = {EMBEDDING_BACKEND}")
print(f"ST_MODEL_NAME = {ST_MODEL_NAME}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Step 1 — Validate source chunks table
# ─────────────────────────────────────────────────────────────

chunks_raw = spark.table(SOURCE_CHUNKS_TABLE)

print(f"Raw chunk count: {chunks_raw.count()}")

display(
    chunks_raw
    .groupBy("chunk_type")
    .count()
    .orderBy("chunk_type")
)

chunks_raw.printSchema()

display(
    chunks_raw
    .select(
        "chunk_id",
        "chunk_type",
        "supplier_id",
        "sku_id",
        "source_name",
        "risk_category",
        "event_date",
        "severity",
        "freshness_weight",
        "chunk_text"
    )
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Step 2 — Clean chunk text and create stable unique IDs
# ─────────────────────────────────────────────────────────────

CHARS_PER_TOKEN_APPROX = 4
MAX_CHUNK_TOKENS = 512
MAX_CHARS = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN_APPROX

cols = set(chunks_raw.columns)

def col_or_null(name, dtype="string", default=None):
    if name in cols:
        return F.col(name).cast(dtype)
    return F.lit(default).cast(dtype)

text_col = "chunk_text_clean" if "chunk_text_clean" in cols else "chunk_text"

chunks_clean = (
    chunks_raw
    .select(
        col_or_null("chunk_id", "string").alias("original_chunk_id"),
        col_or_null("chunk_type", "string").alias("chunk_type"),
        col_or_null("supplier_id", "string").alias("supplier_id"),
        col_or_null("sku_id", "string").alias("sku_id"),
        col_or_null("source_name", "string").alias("source_name"),
        col_or_null("risk_category", "string").alias("risk_category"),
        F.to_date(col_or_null("event_date", "string")).alias("event_date"),
        col_or_null("severity", "string").alias("severity"),
        col_or_null("freshness_weight", "float", 0.5).alias("freshness_weight"),
        col_or_null("source_url", "string").alias("source_url"),
        col_or_null("evidence_doc_id", "string").alias("evidence_doc_id"),
        F.col(text_col).cast("string").alias("chunk_text_raw")
    )
    .withColumn(
        "chunk_text_clean",
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.substring(F.coalesce(F.col("chunk_text_raw"), F.lit("")), 1, MAX_CHARS),
                    r"[\r\n\t]+",
                    " "
                ),
                r"\s+",
                " "
            )
        )
    )
    .drop("chunk_text_raw")
    .filter(F.length(F.col("chunk_text_clean")) > 0)
    .withColumn(
        "freshness_weight",
        F.coalesce(F.col("freshness_weight"), F.lit(0.5).cast("float"))
    )
    # Create a stable unique ID per embedding row.
    # This prevents duplicate chunk_id issues while preserving supplier context.
    .withColumn(
        "chunk_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col("original_chunk_id"), F.lit("")),
                F.coalesce(F.col("supplier_id"), F.lit("")),
                F.coalesce(F.col("sku_id"), F.lit("")),
                F.coalesce(F.col("source_name"), F.lit("")),
                F.coalesce(F.col("risk_category"), F.lit("")),
                F.coalesce(F.col("event_date").cast("string"), F.lit(""))
            ),
            256
        )
    )
)

clean_count = chunks_clean.count()
distinct_ids = chunks_clean.select("chunk_id").distinct().count()

print(f"Clean chunks: {clean_count}")
print(f"Distinct chunk_id values: {distinct_ids}")

assert clean_count > 0, "No clean chunks found."
assert clean_count == distinct_ids, "chunk_id is not unique. Stop and fix ID generation."

display(
    chunks_clean
    .select(
        "chunk_id",
        "original_chunk_id",
        "chunk_type",
        "supplier_id",
        "event_date",
        "freshness_weight",
        "chunk_text_clean"
    )
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Step 3 — Generate embeddings with sentence-transformers
# ─────────────────────────────────────────────────────────────

subprocess.run(
    ["pip", "install", "sentence-transformers", "--quiet"],
    check=True
)

from sentence_transformers import SentenceTransformer

print(f"Loading model: {ST_MODEL_NAME}")
model = SentenceTransformer(ST_MODEL_NAME)

EMBEDDING_DIM = int(
    model.get_embedding_dimension()
    if hasattr(model, "get_embedding_dimension")
    else model.get_sentence_embedding_dimension()
)

print(f"Model loaded. Embedding dimension: {EMBEDDING_DIM}")

chunks_pd = chunks_clean.select(
    "chunk_id",
    "original_chunk_id",
    "chunk_type",
    "supplier_id",
    "sku_id",
    "source_name",
    "risk_category",
    "event_date",
    "severity",
    "chunk_text_clean",
    "freshness_weight",
    "source_url",
    "evidence_doc_id"
).toPandas()

print(f"Collected {len(chunks_pd)} chunks to driver.")

# Normalize text
chunks_pd["chunk_text_clean"] = chunks_pd["chunk_text_clean"].fillna("").astype(str)
chunks_pd = chunks_pd[chunks_pd["chunk_text_clean"].str.strip().str.len() > 0].copy()

print(f"Embedding {len(chunks_pd)} non-empty chunks.")

texts = chunks_pd["chunk_text_clean"].tolist()
all_embeddings = []

for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
    batch = texts[i : i + EMBEDDING_BATCH_SIZE]

    batch_embeddings = model.encode(
        batch,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    all_embeddings.extend(batch_embeddings.tolist())

    print(f"Embedded {min(i + EMBEDDING_BATCH_SIZE, len(texts))} / {len(texts)}")

chunks_pd["embedding"] = all_embeddings
chunks_pd["embedded_at"] = datetime.utcnow()
chunks_pd["embedding_model"] = ST_MODEL_NAME
chunks_pd["embedding_dim"] = EMBEDDING_DIM

# Normalize event_date
chunks_pd["event_date"] = pd.to_datetime(
    chunks_pd["event_date"],
    errors="coerce"
).dt.date

chunks_pd["event_date"] = chunks_pd["event_date"].where(
    pd.notna(chunks_pd["event_date"]),
    None
)

# Normalize strings
string_cols = [
    "chunk_id",
    "original_chunk_id",
    "chunk_type",
    "supplier_id",
    "sku_id",
    "source_name",
    "risk_category",
    "severity",
    "chunk_text_clean",
    "source_url",
    "evidence_doc_id",
    "embedding_model"
]

for c in string_cols:
    chunks_pd[c] = chunks_pd[c].where(pd.notna(chunks_pd[c]), None)
    chunks_pd[c] = chunks_pd[c].apply(lambda x: str(x) if x is not None else None)

chunks_pd["freshness_weight"] = pd.to_numeric(
    chunks_pd["freshness_weight"],
    errors="coerce"
).fillna(0.5).astype("float32")

chunks_pd["embedding_dim"] = chunks_pd["embedding_dim"].astype("int32")

chunks_pd["embedding"] = chunks_pd["embedding"].apply(
    lambda v: [float(x) for x in v]
)

embedding_schema = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("original_chunk_id", StringType(), True),
    StructField("chunk_type", StringType(), True),
    StructField("supplier_id", StringType(), True),
    StructField("sku_id", StringType(), True),
    StructField("source_name", StringType(), True),
    StructField("risk_category", StringType(), True),
    StructField("event_date", DateType(), True),
    StructField("severity", StringType(), True),
    StructField("chunk_text_clean", StringType(), True),
    StructField("freshness_weight", FloatType(), True),
    StructField("source_url", StringType(), True),
    StructField("evidence_doc_id", StringType(), True),
    StructField("embedding", ArrayType(FloatType()), False),
    StructField("embedded_at", TimestampType(), False),
    StructField("embedding_model", StringType(), True),
    StructField("embedding_dim", IntegerType(), True),
])

chunks_with_embeddings = (
    spark.createDataFrame(chunks_pd, schema=embedding_schema)
    .withColumnRenamed("chunk_text_clean", "chunk_text")
)

print(f"Embedding complete: {chunks_with_embeddings.count()} rows")
print(f"Embedding dimension: {EMBEDDING_DIM}")

display(
    chunks_with_embeddings
    .select(
        "chunk_id",
        "original_chunk_id",
        "chunk_type",
        "supplier_id",
        "event_date",
        "embedding_model",
        "embedding_dim",
        F.size("embedding").alias("vector_length")
    )
    .limit(10)
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Step 4 — Write embeddings to Gold Delta
# ─────────────────────────────────────────────────────────────

assert "chunks_with_embeddings" in globals(), "chunks_with_embeddings not found. Rerun Cell 4."

(
    chunks_with_embeddings
    .withColumn("gold_created_at", F.current_timestamp())
    .withColumn("gold_source_notebook", F.lit("25_gold_rag_embeddings_clean"))
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(EMBEDDINGS_TABLE)
)

print(f"Wrote table: {EMBEDDINGS_TABLE}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Step 5 — Validate embeddings
# ─────────────────────────────────────────────────────────────

embedding_validation = spark.sql("""
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT chunk_id) AS distinct_chunks,
    MIN(embedding_dim) AS min_dim,
    MAX(embedding_dim) AS max_dim,
    MIN(size(embedding)) AS min_vector_length,
    MAX(size(embedding)) AS max_vector_length,
    SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) AS null_embedding_count,
    SUM(CASE WHEN chunk_text IS NULL OR length(trim(chunk_text)) = 0 THEN 1 ELSE 0 END) AS empty_text_count
FROM supplysage_gold.gold_rag_embeddings
""")

display(embedding_validation)

v = embedding_validation.collect()[0]

assert v["row_count"] > 0, "No embeddings written."
assert v["row_count"] == v["distinct_chunks"], "Duplicate chunk_id values found."
assert v["null_embedding_count"] == 0, "Some embeddings are NULL."
assert v["empty_text_count"] == 0, "Some chunk_text values are empty."
assert v["min_dim"] == v["max_dim"], "Embedding dimensions are inconsistent."
assert v["min_vector_length"] == v["max_vector_length"], "Vector lengths are inconsistent."
assert v["min_dim"] == v["min_vector_length"], "embedding_dim does not match vector length."

print("Embedding validation passed.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Step 6 — Build metadata retrieval index
# ─────────────────────────────────────────────────────────────

from pyspark.sql import functions as F

GOLD_SCHEMA = "supplysage_gold"
EMBEDDINGS_TABLE = f"{GOLD_SCHEMA}.gold_rag_embeddings"
RETRIEVAL_INDEX_TABLE = f"{GOLD_SCHEMA}.gold_rag_retrieval_index"

embeddings_df = spark.table(EMBEDDINGS_TABLE)

retrieval_index = (
    embeddings_df
    .select(
        "chunk_id",
        "original_chunk_id",
        "chunk_type",
        "supplier_id",
        "sku_id",
        "source_name",
        "risk_category",
        "event_date",
        "severity",
        "freshness_weight",
        "source_url",
        "evidence_doc_id",
        "embedding_model",
        "embedding_dim",
        "embedded_at",
        "gold_created_at",
        "gold_source_notebook"
    )
    .withColumn(
        "supplier_id_partition",
        F.coalesce(F.col("supplier_id"), F.lit("__unmatched__"))
    )
    .withColumn(
        "is_fresh",
        F.when(
            F.col("event_date").isNotNull(),
            F.col("event_date") >= F.date_sub(F.current_date(), 30)
        ).otherwise(F.lit(False))
    )
    .withColumn(
        "is_critical_or_high",
        F.lower(F.col("severity")).isin("critical", "high")
    )
    .withColumn(
        "days_since_event",
        F.when(
            F.col("event_date").isNotNull(),
            F.datediff(F.current_date(), F.col("event_date"))
        ).otherwise(None).cast("int")
    )
    .withColumn(
        "retrieval_text",
        F.concat_ws(
            " | ",
            F.coalesce(F.col("chunk_type"), F.lit("")),
            F.coalesce(F.col("supplier_id"), F.lit("")),
            F.coalesce(F.col("sku_id"), F.lit("")),
            F.coalesce(F.col("source_name"), F.lit("")),
            F.coalesce(F.col("risk_category"), F.lit("")),
            F.coalesce(F.col("severity"), F.lit(""))
        )
    )
)

(
    retrieval_index
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("supplier_id_partition")
    .saveAsTable(RETRIEVAL_INDEX_TABLE)
)

print(f"Wrote table: {RETRIEVAL_INDEX_TABLE}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# Step 7 — Final validation
# ─────────────────────────────────────────────────────────────

EMBEDDINGS_TABLE = "supplysage_gold.gold_rag_embeddings"
RETRIEVAL_INDEX_TABLE = "supplysage_gold.gold_rag_retrieval_index"

emb_tbl = spark.table(EMBEDDINGS_TABLE)
idx_tbl = spark.table(RETRIEVAL_INDEX_TABLE)

emb_count = emb_tbl.count()
idx_count = idx_tbl.count()

print(f"Embeddings rows: {emb_count}")
print(f"Retrieval index rows: {idx_count}")

assert emb_count > 0, "Embeddings table is empty."
assert idx_count == emb_count, "Retrieval index row count does not match embeddings."

display(
    emb_tbl
    .groupBy("chunk_type")
    .count()
    .orderBy("chunk_type")
)

display(
    emb_tbl
    .select(
        "chunk_id",
        "original_chunk_id",
        "chunk_type",
        "supplier_id",
        "sku_id",
        "source_name",
        "risk_category",
        "event_date",
        "embedding_model",
        "embedding_dim",
        F.size("embedding").alias("vector_length")
    )
    .limit(20)
)

print("Notebook 25 clean run completed successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration — set these before running

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Load chunks from gold_rag_evidence_chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Write gold_rag_embeddings to Delta
# MAGIC Enable Change Data Feed (CDF) — required for Databricks Vector Search
# MAGIC to sync the index incrementally without full re-indexing on each run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Inspect embedding quality
# MAGIC Sanity checks before registering the Vector Search index.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Build the retrieval function
# MAGIC
# MAGIC This cell defines `retrieve_evidence()` — the Python function the LangGraph
# MAGIC agent will import and call during an investigation. It runs hybrid retrieval:
# MAGIC
# MAGIC **Phase 1 — Metadata pre-filter** (from gold_rag_retrieval_index):
# MAGIC   Filter by supplier_id, date window, risk_category, severity.
# MAGIC   Returns candidate chunk_ids.
# MAGIC
# MAGIC **Phase 2 — Semantic re-rank** (from gold_rag_embeddings):
# MAGIC   Embed the query text, compute cosine similarity against candidate embeddings,
# MAGIC   apply freshness weight boost, return top-k chunks.
# MAGIC
# MAGIC This function is written to a Python file in DBFS for import by the agent.

# COMMAND ----------

print("chunks_clean exists:", "chunks_clean" in globals())
print("chunks_pd exists:", "chunks_pd" in globals())
print("chunks_with_embeddings exists:", "chunks_with_embeddings" in globals())

if "chunks_pd" in globals():
    print("chunks_pd rows:", len(chunks_pd))
    print("chunks_pd columns:", chunks_pd.columns.tolist())
    print("has embedding column:", "embedding" in chunks_pd.columns)
    if "embedding" in chunks_pd.columns:
        print("sample embedding length:", len(chunks_pd["embedding"].iloc[0]))

# COMMAND ----------

retrieval_code = '''
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
            SELECT ai_embed_text(\'{endpoint}\', \'{query_text.replace("\'", "\\\'")}\') AS embedding
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

    lines = ["## Retrieved Evidence\\n"]
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
    return "\\n".join(lines)
'''


print("✅ Skipped /FileStore helper-module export.")
print("Notebook 27 contains the retrieval logic directly and reads:")
print("- supplysage_gold.gold_chat_context_snapshots")
print("- supplysage_gold.gold_rag_embeddings")
print("- supplysage_gold.gold_rag_retrieval_index")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Test retrieval end-to-end
# MAGIC Run a sample query to verify the full pipeline works before connecting the agent.

# COMMAND ----------

# Test query simulating what the chatbot would send
TEST_QUERIES = [
    {
        "query": "Why did Pacific Rim Foods risk score spike overnight?",
        "supplier_id": None,  # will be resolved by agent from entity context
        "top_k": 4
    },
    {
        "query": "Which external events are affecting China-sourced suppliers?",
        "supplier_id": None,
        "top_k": 5
    },
    {
        "query": "What evidence supports the FOODS_1_001 stockout alert?",
        "sku_id": None,
        "top_k": 4
    }
]

print("=" * 70)
print("RAG RETRIEVAL TEST")
print("=" * 70)

# For the test, import the retriever directly
import sys
sys.path.insert(0, '/dbfs/FileStore/supplysage')

try:
    from supplysage_rag_retriever import retrieve_evidence, format_evidence_for_llm

    for test in TEST_QUERIES:
        print(f"\nQuery: {test['query']}")
        print("-" * 50)

        # Use sentence_transformers for the test if no VS endpoint configured
        results = retrieve_evidence(
            query_text=test["query"],
            supplier_id=test.get("supplier_id"),
            sku_id=test.get("sku_id"),
            top_k=test["top_k"],
            embedding_backend=EMBEDDING_BACKEND,
            embedding_endpoint=DATABRICKS_EMBEDDING_ENDPOINT
        )

        if results:
            for r in results:
                print(f"  [{r['source_name']}] score={r['retrieval_score']:.3f} "
                      f"fresh={r['freshness_weight']:.1f} | {r['chunk_text'][:120]}...")
        else:
            print("  No results returned.")

    print("\n" + "=" * 70)
    print("✅ Retrieval test complete.")

except Exception as e:
    print(f"⚠️  Test failed: {e}")
    print("   Check embedding backend configuration and retry.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 — Final validation

# COMMAND ----------

results = []

emb_tbl = spark.table("supplysage_gold.gold_rag_embeddings")
idx_tbl = spark.table("supplysage_gold.gold_rag_retrieval_index")

# V1: embeddings table non-empty
rc_emb = emb_tbl.count()
results.append({
    "check": "embeddings_table_non_empty",
    "status": "PASS" if rc_emb > 0 else "FAIL",
    "detail": str(rc_emb)
})

# V2: no null embeddings
null_emb = emb_tbl.filter(F.col("embedding").isNull()).count()
results.append({
    "check": "no_null_embeddings",
    "status": "PASS" if null_emb == 0 else "FAIL",
    "detail": f"{null_emb} null embeddings"
})

# V3: embedding dimension consistent
if rc_emb > 0:
    dim_check = emb_tbl.filter(F.col("embedding").isNotNull()).withColumn(
        "dim", F.size(F.col("embedding"))
    ).agg(
        F.min("dim").alias("min_dim"),
        F.max("dim").alias("max_dim")
    ).collect()[0]
    dim_consistent = dim_check["min_dim"] == dim_check["max_dim"]
    results.append({
        "check": "embedding_dim_consistent",
        "status": "PASS" if dim_consistent else "FAIL",
        "detail": f"min={dim_check['min_dim']} max={dim_check['max_dim']}"
    })

# V4: retrieval index row count matches embeddings
rc_idx = idx_tbl.count()
results.append({
    "check": "retrieval_index_matches_embeddings",
    "status": "PASS" if rc_idx == rc_emb else "FAIL",
    "detail": f"embeddings={rc_emb} index={rc_idx}"
})

# V5: both chunk types present
chunk_types = [row["chunk_type"] for row in emb_tbl.select("chunk_type").distinct().collect()]
has_both_types = "external_event" in chunk_types and "risk_explanation" in chunk_types
results.append({
    "check": "both_chunk_types_present",
    "status": "PASS" if has_both_types else "WARN",
    "detail": f"found: {chunk_types}"
})

# V6: supplier-matched chunks exist
supplier_chunks = emb_tbl.filter(F.col("supplier_id").isNotNull()).count()
results.append({
    "check": "supplier_matched_chunks_exist",
    "status": "PASS" if supplier_chunks > 0 else "FAIL",
    "detail": f"{supplier_chunks} supplier-matched chunks"
})

print("\nValidation results:")
for r in results:
    icon = "✅" if r["status"] == "PASS" else ("⚠️ " if r["status"] == "WARN" else "❌")
    print(f"  {icon} [{r['status']}] {r['check']} — {r['detail']}")

from datetime import datetime as _dt
val_df = spark.createDataFrame(results).withColumn(
    "notebook", F.lit("25_gold_rag_embeddings")
).withColumn("run_at", F.lit(_dt.utcnow().isoformat()))
(
    val_df.write.format("delta").mode("append")
    .option("mergeSchema", "true")
    .saveAsTable("supplysage_gold.gold_transform_validation_results")
)

failures = [r for r in results if r["status"] == "FAIL"]
assert len(failures) == 0, f"Validation failures: {failures}"
print("\n✅ All validations passed. RAG embedding pipeline complete.")
print(f"\nSummary:")
print(f"  Total chunks embedded:      {rc_emb}")
print(f"  Supplier-matched chunks:    {supplier_chunks}")
print(f"  Embedding backend:          {EMBEDDING_BACKEND}")
print(f"  Embedding dim:              {EMBEDDING_DIM}")
print(f"  Vector Search registered:   {ENABLE_VECTOR_SEARCH}")
print(f"\nNext step: Notebook 25b — gold_chat_context_snapshots")
print(f"Then:      Notebook 25_langgraph — LangGraph agent setup")
