# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 24 — gold_rag_evidence_chunks
# MAGIC **Sources:**
# MAGIC   - `supplysage_gold.gold_external_risk_event_mart`
# MAGIC   - `supplysage_silver.silver_external_evidence_documents`
# MAGIC   - `supplysage_gold.gold_supplier_risk_explanation_log`
# MAGIC **Target:** `supplysage_gold.gold_rag_evidence_chunks`
# MAGIC **Grain:** One row per evidence chunk (one per external event + one per supplier explanation)
# MAGIC **Purpose:** Retrieval index for the chatbot RAG pipeline.
# MAGIC NOTE: Embeddings are populated by a separate embedding job (not in this notebook).
# MAGIC This notebook writes the chunk text and metadata. The embedding step reads this table,
# MAGIC generates vectors, and writes them back as an additional column or to a vector store.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

events      = spark.table("supplysage_gold.gold_external_risk_event_mart")
evidence    = spark.table("supplysage_silver.silver_external_evidence_documents")
explanations = spark.table("supplysage_gold.gold_supplier_risk_explanation_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk type 1: External event evidence chunks

# COMMAND ----------

event_chunks = (
    events
    .filter(F.col("matched_supplier_id").isNotNull())
    .withColumn(
        "chunk_text",
        F.concat_ws(" | ",
            F.col("source_name"),
            F.col("event_title"),
            F.coalesce(F.col("event_summary"), F.lit("")),
            F.concat(F.lit("Severity: "), F.col("severity")),
            F.concat(F.lit("Matched supplier: "), F.col("matched_supplier_id")),
            F.concat(F.lit("Match type: "), F.col("match_type")),
            F.coalesce(F.col("event_country"), F.lit("")),
            F.coalesce(F.col("event_region"), F.lit(""))
        )
    )
    .withColumn(
        # Fresher events score higher in retrieval
        "freshness_weight",
        F.when(
            F.col("event_date") >= F.date_sub(F.current_date(), 1), F.lit(1.0)
        ).when(
            F.col("event_date") >= F.date_sub(F.current_date(), 7), F.lit(0.8)
        ).when(
            F.col("event_date") >= F.date_sub(F.current_date(), 30), F.lit(0.5)
        ).otherwise(F.lit(0.2))
    )
    .select(
        F.concat(F.lit("EVT_"), F.col("external_event_id")).alias("chunk_id"),
        F.lit("external_event").alias("chunk_type"),
        F.col("matched_supplier_id").alias("supplier_id"),
        F.lit(None).cast("string").alias("sku_id"),
        F.col("source_name"),
        F.col("risk_category"),
        F.col("event_date"),
        F.col("severity"),
        F.col("chunk_text"),
        F.col("freshness_weight"),
        F.col("source_url"),
        F.col("evidence_doc_id"),
        F.lit(None).cast("array<float>").alias("embedding")  # populated by embedding job
    )
)

print(f"Event chunks: {event_chunks.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk type 2: Supplier risk explanation chunks

# COMMAND ----------

explanation_chunks = (
    explanations
    .withColumn(
        "chunk_text",
        F.concat_ws(" | ",
            F.concat(F.lit("Supplier: "), F.col("supplier_name")),
            F.concat(F.lit("Risk score: "), F.col("overall_risk_score").cast("string")),
            F.concat(F.lit("Risk band: "), F.col("risk_band")),
            F.concat(F.lit("Score delta 24h: "), F.col("score_delta_24h").cast("string")),
            F.concat(F.lit("Top driver: "), F.col("top_risk_driver")),
            F.concat(F.lit("Driver 1: "), F.coalesce(F.col("driver_1_detail"), F.lit(""))),
            F.concat(F.lit("Driver 2: "), F.coalesce(F.col("driver_2_detail"), F.lit(""))),
            F.concat(F.lit("Recommended action: "), F.col("recommended_action"))
        )
    )
    .select(
        F.concat(F.lit("EXP_"), F.col("supplier_id"), F.lit("_"), F.col("score_date")).alias("chunk_id"),
        F.lit("risk_explanation").alias("chunk_type"),
        F.col("supplier_id"),
        F.lit(None).cast("string").alias("sku_id"),
        F.lit("internal_risk_engine").alias("source_name"),
        F.lit("supplier_risk").alias("risk_category"),
        F.col("score_date").cast("date").alias("event_date"),
        F.col("risk_band").alias("severity"),
        F.col("chunk_text"),
        F.lit(1.0).alias("freshness_weight"),  # always fresh — computed today
        F.lit(None).cast("string").alias("source_url"),
        F.lit(None).cast("string").alias("evidence_doc_id"),
        F.lit(None).cast("array<float>").alias("embedding")
    )
)

print(f"Explanation chunks: {explanation_chunks.count()}")

# COMMAND ----------

rag_chunks = event_chunks.union(explanation_chunks).withColumn(
    "gold_created_at", F.lit(datetime.utcnow().isoformat())
).withColumn("gold_source_notebook", F.lit("24_gold_rag_evidence_chunks"))

total = rag_chunks.count()
print(f"Total RAG chunks: {total}")

(
    rag_chunks
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_rag_evidence_chunks")
)
print(f"✅ gold_rag_evidence_chunks written: {spark.table('supplysage_gold.gold_rag_evidence_chunks').count()} rows")
print("NOTE: Embeddings column is NULL — run your embedding job to populate it.")

# COMMAND ----------

# Notebook 24 validation
results = []
chunks = spark.table("supplysage_gold.gold_rag_evidence_chunks")
rc = chunks.count()
results.append({"check": "row_count_gt_0", "status": "PASS" if rc > 0 else "FAIL", "detail": str(rc)})
null_chunk = chunks.filter(F.col("chunk_text").isNull() | (F.col("chunk_text") == "")).count()
results.append({"check": "no_empty_chunk_text", "status": "PASS" if null_chunk == 0 else "FAIL", "detail": str(null_chunk)})
for r in results:
    print(f"  [{r['status']}] {r['check']} — {r['detail']}")
val_df = spark.createDataFrame(results).withColumn("notebook", F.lit("24_gold_rag_evidence_chunks")).withColumn("run_at", F.lit(datetime.utcnow().isoformat()))
val_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("supplysage_gold.gold_transform_validation_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Notebook 26 — gold_alert_benchmarks + gold_alert_events
# MAGIC **Sources:**
# MAGIC   - `supplysage_gold.gold_supplier_risk_scores`
# MAGIC   - `supplysage_gold.gold_sku_stockout_risk_scores`
# MAGIC **Targets:**
# MAGIC   - `supplysage_gold.gold_alert_benchmarks` (seed config table)
# MAGIC   - `supplysage_gold.gold_alert_events` (triggered alerts)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, BooleanType, IntegerType
import uuid
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seed gold_alert_benchmarks config table

# COMMAND ----------

benchmarks = [
    {
        "benchmark_id": "SUP_SCORE_CRITICAL",
        "entity_type": "supplier",
        "metric_name": "overall_risk_score",
        "threshold_value": 75.0,
        "comparison": "gte",
        "severity": "critical",
        "active": True,
        "description": "Supplier composite risk score >= 75"
    },
    {
        "benchmark_id": "SUP_SCORE_HIGH",
        "entity_type": "supplier",
        "metric_name": "overall_risk_score",
        "threshold_value": 55.0,
        "comparison": "gte",
        "severity": "high",
        "active": True,
        "description": "Supplier composite risk score >= 55"
    },
    {
        "benchmark_id": "SUP_DELTA_24H",
        "entity_type": "supplier",
        "metric_name": "score_delta_24h",
        "threshold_value": 15.0,
        "comparison": "gte",
        "severity": "critical",
        "active": True,
        "description": "Supplier score spiked >= 15 pts in 24h — indicates new event"
    },
    {
        "benchmark_id": "SKU_STOCKOUT_CRITICAL",
        "entity_type": "sku",
        "metric_name": "stockout_probability",
        "threshold_value": 0.75,
        "comparison": "gte",
        "severity": "critical",
        "active": True,
        "description": "SKU stockout probability >= 75%"
    },
    {
        "benchmark_id": "SKU_STOCKOUT_HIGH",
        "entity_type": "sku",
        "metric_name": "stockout_probability",
        "threshold_value": 0.50,
        "comparison": "gte",
        "severity": "high",
        "active": True,
        "description": "SKU stockout probability >= 50%"
    },
    {
        "benchmark_id": "SKU_COVER_CRITICAL",
        "entity_type": "sku",
        "metric_name": "days_of_cover",
        "threshold_value": 7.0,
        "comparison": "lte",
        "severity": "critical",
        "active": True,
        "description": "SKU days of cover <= 7 days"
    },
    {
        "benchmark_id": "SKU_NO_ALTERNATE",
        "entity_type": "sku",
        "metric_name": "alternate_status",
        "threshold_value": 0.0,   # flag-style, checked separately
        "comparison": "eq_none",
        "severity": "high",
        "active": True,
        "description": "High-risk SKU with no approved alternate supplier"
    },
]

bench_df = spark.createDataFrame(benchmarks).withColumn(
    "gold_created_at", F.lit(datetime.utcnow().isoformat())
)

if not spark.catalog.tableExists("supplysage_gold.gold_alert_benchmarks"):
    bench_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("supplysage_gold.gold_alert_benchmarks")
    print(f"✅ gold_alert_benchmarks seeded: {spark.table('supplysage_gold.gold_alert_benchmarks').count()} benchmarks")
else:
    print("gold_alert_benchmarks already exists — skipping seed to preserve customizations.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate alert candidates from score tables

# COMMAND ----------

supplier_scores = spark.table("supplysage_gold.gold_supplier_risk_scores")
sku_scores      = spark.table("supplysage_gold.gold_sku_stockout_risk_scores")

# Supplier alerts: score >= threshold
sup_critical = supplier_scores.filter(F.col("overall_risk_score") >= 75.0).select(
    F.col("supplier_id").alias("entity_id"),
    F.col("supplier_name").alias("entity_name"),
    F.lit("supplier").alias("entity_type"),
    F.lit("SUP_SCORE_CRITICAL").alias("benchmark_id"),
    F.col("overall_risk_score").alias("actual_value"),
    F.lit(75.0).alias("threshold_value"),
    (F.col("overall_risk_score") - 75.0).alias("breach_amount"),
    F.lit("critical").alias("severity"),
    F.concat(
        F.lit("Score "), F.col("overall_risk_score").cast("string"),
        F.lit(" — "), F.col("top_risk_driver")
    ).alias("trigger_description")
)

sup_delta = supplier_scores.filter(F.col("score_delta_24h") >= 15.0).select(
    F.col("supplier_id").alias("entity_id"),
    F.col("supplier_name").alias("entity_name"),
    F.lit("supplier").alias("entity_type"),
    F.lit("SUP_DELTA_24H").alias("benchmark_id"),
    F.col("score_delta_24h").alias("actual_value"),
    F.lit(15.0).alias("threshold_value"),
    (F.col("score_delta_24h") - 15.0).alias("breach_amount"),
    F.lit("critical").alias("severity"),
    F.concat(
        F.lit("Score Δ +"), F.col("score_delta_24h").cast("string"),
        F.lit(" in 24h — "), F.col("top_risk_driver")
    ).alias("trigger_description")
)

# SKU alerts: stockout probability >= threshold
sku_critical = sku_scores.filter(F.col("stockout_probability") >= 0.75).select(
    F.col("canonical_sku_id").alias("entity_id"),
    F.col("canonical_sku_id").alias("entity_name"),
    F.lit("sku").alias("entity_type"),
    F.lit("SKU_STOCKOUT_CRITICAL").alias("benchmark_id"),
    F.col("stockout_probability").alias("actual_value"),
    F.lit(0.75).alias("threshold_value"),
    (F.col("stockout_probability") - 0.75).alias("breach_amount"),
    F.lit("critical").alias("severity"),
    F.concat(
        F.lit("Stockout prob "),
        F.round(F.col("stockout_probability") * 100, 0).cast("int").cast("string"),
        F.lit("% — "), F.col("days_of_cover").cast("string"), F.lit("d cover remaining")
    ).alias("trigger_description")
)

sku_cover = sku_scores.filter(
    (F.col("days_of_cover") <= 7.0) & (F.col("stockout_risk_band").isin("critical", "high"))
).select(
    F.col("canonical_sku_id").alias("entity_id"),
    F.col("canonical_sku_id").alias("entity_name"),
    F.lit("sku").alias("entity_type"),
    F.lit("SKU_COVER_CRITICAL").alias("benchmark_id"),
    F.col("days_of_cover").alias("actual_value"),
    F.lit(7.0).alias("threshold_value"),
    (F.lit(7.0) - F.col("days_of_cover")).alias("breach_amount"),
    F.lit("critical").alias("severity"),
    F.concat(
        F.lit("Only "), F.col("days_of_cover").cast("string"),
        F.lit("d of cover — supplier "), F.col("supplier_name")
    ).alias("trigger_description")
)

# COMMAND ----------

# Union all alert candidates
all_candidates = sup_critical.union(sup_delta).union(sku_critical).union(sku_cover)

# Add alert metadata
alert_events = (
    all_candidates
    .withColumn("alert_id", F.concat(F.lit("ALT_"), F.abs(F.hash(F.col("entity_id"), F.col("benchmark_id"))).cast("string")))
    .withColumn("status", F.lit("triggered"))
    .withColumn("triggered_at", F.lit(datetime.utcnow().isoformat()))
    .withColumn("sent_at", F.lit(None).cast("string"))
    .withColumn("investigation_id", F.lit(None).cast("string"))
    .withColumn("email_subject", F.lit(None).cast("string"))
    .withColumn("email_body", F.lit(None).cast("string"))
    .withColumn("gold_created_at", F.lit(datetime.utcnow().isoformat()))
    .withColumn("gold_source_notebook", F.lit("26_gold_alert_events"))
    .dropDuplicates(["alert_id"])
)

alert_count = alert_events.count()
print(f"Alert events generated: {alert_count}")
display(alert_events.select("alert_id", "entity_name", "entity_type", "benchmark_id", "severity", "trigger_description").orderBy("severity"))

# COMMAND ----------

(
    alert_events
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_alert_events")
)
print(f"✅ gold_alert_events written: {spark.table('supplysage_gold.gold_alert_events').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Notebook 27 — Dashboard marts
# MAGIC **Targets:**
# MAGIC   - `supplysage_gold.gold_dashboard_supplier_risk_summary` (Tab 1 + Tab 2)
# MAGIC   - `supplysage_gold.gold_dashboard_sku_stockout_summary` (Tab 3)
# MAGIC These are pre-joined, pre-aggregated marts for fast dashboard queries.
# MAGIC No heavy joins at query time in the React app.

# COMMAND ----------

# MAGIC %md
# MAGIC ## gold_dashboard_supplier_risk_summary

# COMMAND ----------

dim_sup   = spark.table("supplysage_gold.gold_dim_suppliers")
sup_scores = spark.table("supplysage_gold.gold_supplier_risk_scores")
ext_events = spark.table("supplysage_gold.gold_external_risk_event_mart")
exp_log    = spark.table("supplysage_gold.gold_supplier_risk_explanation_log")
dep_mart_s = spark.table("supplysage_gold.gold_supplier_sku_dependency_mart")

# Count active external events per supplier
active_events_count = (
    ext_events
    .filter(
        F.col("matched_supplier_id").isNotNull() &
        (F.col("event_date") >= F.date_sub(F.current_date(), 30))
    )
    .groupBy(F.col("matched_supplier_id").alias("supplier_id"))
    .agg(
        F.count("external_event_id").alias("active_event_count_30d"),
        F.sum(F.when(F.col("severity") == "critical", 1).otherwise(0)).alias("critical_event_count"),
        F.max("event_date").alias("latest_event_date")
    )
)

# Count impacted SKU stats per supplier
sku_exposure = (
    dep_mart_s
    .filter(F.col("is_primary_supplier") == True)
    .groupBy("supplier_id")
    .agg(
        F.count("canonical_sku_id").alias("impacted_sku_count"),
        F.sum(F.when(F.col("alternate_status") == "none", 1).otherwise(0)).alias("no_alternate_sku_count")
    )
)

supplier_dash = (
    dim_sup
    .join(sup_scores.select(
        "supplier_id", "overall_risk_score", "risk_band",
        "score_delta_24h", "score_delta_7d", "top_risk_driver",
        "recommended_action", "operational_score", "dependency_score",
        "external_event_score", "logistics_score", "sanctions_score", "cyber_score",
        "deterioration_flag", "score_date"
    ), on="supplier_id", how="left")
    .join(active_events_count, on="supplier_id", how="left")
    .join(sku_exposure, on="supplier_id", how="left")
    .join(exp_log.select("supplier_id", "driver_1_detail", "driver_2_detail", "evidence_count"), on="supplier_id", how="left")
    .withColumn("active_event_count_30d", F.coalesce(F.col("active_event_count_30d"), F.lit(0)))
    .withColumn("impacted_sku_count", F.coalesce(F.col("impacted_sku_count"), F.lit(0)))
    .withColumn("no_alternate_sku_count", F.coalesce(F.col("no_alternate_sku_count"), F.lit(0)))
    .withColumn("gold_created_at", F.lit(datetime.utcnow().isoformat()))
    .withColumn("gold_source_notebook", F.lit("27_gold_dashboard_supplier_risk_summary"))
)

supplier_dash_count = supplier_dash.count()
print(f"gold_dashboard_supplier_risk_summary: {supplier_dash_count} rows")
assert supplier_dash_count == 48, f"Expected 48, got {supplier_dash_count}"

(
    supplier_dash
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_dashboard_supplier_risk_summary")
)
print("✅ gold_dashboard_supplier_risk_summary written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## gold_dashboard_sku_stockout_summary

# COMMAND ----------

sku_scores_tbl = spark.table("supplysage_gold.gold_sku_stockout_risk_scores")
dep_mart_k     = spark.table("supplysage_gold.gold_supplier_sku_dependency_mart")
dim_skus_tbl   = spark.table("supplysage_gold.gold_dim_products_skus")

sku_dash = (
    sku_scores_tbl
    .join(dim_skus_tbl.select("canonical_sku_id", "m5_item_id", "retail_product_id"), on="canonical_sku_id", how="left")
    .withColumn("gold_created_at", F.lit(datetime.utcnow().isoformat()))
    .withColumn("gold_source_notebook", F.lit("27_gold_dashboard_sku_stockout_summary"))
)

sku_dash_count = sku_dash.count()
print(f"gold_dashboard_sku_stockout_summary: {sku_dash_count} rows")

(
    sku_dash
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_dashboard_sku_stockout_summary")
)
print("✅ gold_dashboard_sku_stockout_summary written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final validation summary across all Gold tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT notebook, COUNT(*) AS check_count,
# MAGIC        SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END) AS passed,
# MAGIC        SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) AS failed,
# MAGIC        SUM(CASE WHEN status = 'WARN' THEN 1 ELSE 0 END) AS warnings
# MAGIC FROM supplysage_gold.gold_transform_validation_results
# MAGIC GROUP BY notebook
# MAGIC ORDER BY notebook

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Final Gold table row count inventory
# MAGIC SELECT 'gold_dim_suppliers'                    AS table_name, COUNT(*) AS row_count FROM supplysage_gold.gold_dim_suppliers
# MAGIC UNION ALL SELECT 'gold_dim_products_skus',                    COUNT(*) FROM supplysage_gold.gold_dim_products_skus
# MAGIC UNION ALL SELECT 'gold_supplier_sku_dependency_mart',         COUNT(*) FROM supplysage_gold.gold_supplier_sku_dependency_mart
# MAGIC UNION ALL SELECT 'gold_inventory_stockout_feature_mart',      COUNT(*) FROM supplysage_gold.gold_inventory_stockout_feature_mart
# MAGIC UNION ALL SELECT 'gold_supplier_performance_mart',            COUNT(*) FROM supplysage_gold.gold_supplier_performance_mart
# MAGIC UNION ALL SELECT 'gold_external_risk_event_mart',             COUNT(*) FROM supplysage_gold.gold_external_risk_event_mart
# MAGIC UNION ALL SELECT 'gold_supplier_risk_scores',                 COUNT(*) FROM supplysage_gold.gold_supplier_risk_scores
# MAGIC UNION ALL SELECT 'gold_supplier_risk_explanation_log',        COUNT(*) FROM supplysage_gold.gold_supplier_risk_explanation_log
# MAGIC UNION ALL SELECT 'gold_sku_stockout_risk_scores',             COUNT(*) FROM supplysage_gold.gold_sku_stockout_risk_scores
# MAGIC UNION ALL SELECT 'gold_rag_evidence_chunks',                  COUNT(*) FROM supplysage_gold.gold_rag_evidence_chunks
# MAGIC UNION ALL SELECT 'gold_alert_benchmarks',                     COUNT(*) FROM supplysage_gold.gold_alert_benchmarks
# MAGIC UNION ALL SELECT 'gold_alert_events',                         COUNT(*) FROM supplysage_gold.gold_alert_events
# MAGIC UNION ALL SELECT 'gold_dashboard_supplier_risk_summary',      COUNT(*) FROM supplysage_gold.gold_dashboard_supplier_risk_summary
# MAGIC UNION ALL SELECT 'gold_dashboard_sku_stockout_summary',       COUNT(*) FROM supplysage_gold.gold_dashboard_sku_stockout_summary
# MAGIC ORDER BY table_name
