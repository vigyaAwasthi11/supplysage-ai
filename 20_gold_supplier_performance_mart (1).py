# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 20 — gold_supplier_performance_mart
# MAGIC **Sources:**
# MAGIC   - `supplysage_silver.silver_supplier_scorecards` (576 rows — 48 suppliers × 12 months)
# MAGIC   - `supplysage_silver.silver_purchase_orders` (1,707 rows)
# MAGIC   - `supplysage_silver.silver_shipment_routes` (97 rows)
# MAGIC   - `supplysage_silver.silver_domain_supplier_network` (48 rows, for summary fields)
# MAGIC **Target:** `supplysage_gold.gold_supplier_performance_mart`
# MAGIC **Grain:** One row per supplier × scorecard_month (576 rows expected)
# MAGIC **Purpose:** Monthly supplier KPIs with 3-month trend slopes and deterioration detection.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime
import numpy as np

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

scorecards  = spark.table("supplysage_silver.silver_supplier_scorecards")
pos         = spark.table("supplysage_silver.silver_purchase_orders")
routes      = spark.table("supplysage_silver.silver_shipment_routes")
network     = spark.table("supplysage_silver.silver_domain_supplier_network")

print(f"silver_supplier_scorecards:       {scorecards.count()} rows")
print(f"silver_purchase_orders:           {pos.count()} rows")
print(f"silver_shipment_routes:           {routes.count()} rows")
print(f"silver_domain_supplier_network:   {network.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate PO metrics per supplier per month

# COMMAND ----------

po_monthly = (
    pos
    .withColumn("po_month", F.trunc(F.col("order_date"), "month"))
    .groupBy("supplier_id", "po_month")
    .agg(
        F.count("po_id").alias("po_count"),
        F.count("po_line_id").alias("po_line_count"),
        # silver_purchase_orders already computes is_po_late — use it directly
        F.sum(F.when(F.col("is_po_late") == True, 1).otherwise(0)).alias("late_po_line_count"),
        # po_fill_rate is already precomputed in Silver — just average it
        F.avg("po_fill_rate").alias("avg_po_fill_rate"),
        # delivery_delay_days is already precomputed in Silver
        F.avg("delivery_delay_days").alias("avg_delivery_delay_days")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute 3-month rolling slope for fill_rate and on_time_delivery_rate
# MAGIC Use LAG functions to compute a simple slope proxy:
# MAGIC slope ≈ (current - value_3_months_ago) / 3
# MAGIC Negative slope = deteriorating trend.

# COMMAND ----------

# Sort window for trend calculation
trend_window = Window.partitionBy("supplier_id").orderBy("scorecard_month")
lag3_window  = Window.partitionBy("supplier_id").orderBy("scorecard_month").rowsBetween(-2, 0)

scorecards_with_trend = (
    scorecards
    .withColumn(
        "fill_rate_3mo_slope",
        (F.col("fill_rate") - F.lag("fill_rate", 2).over(trend_window)) / 2.0
    )
    .withColumn(
        "otd_3mo_slope",
        (F.col("on_time_delivery_rate") - F.lag("on_time_delivery_rate", 2).over(trend_window)) / 2.0
    )
    .withColumn(
        "defect_rate_3mo_slope",
        (F.col("defect_rate") - F.lag("defect_rate", 2).over(trend_window)) / 2.0
    )
    .withColumn(
        "deterioration_flag",
        F.when(
            (F.col("fill_rate_3mo_slope") < -0.02) | (F.col("otd_3mo_slope") < -0.02),
            F.lit(True)
        ).otherwise(F.lit(False))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join scorecards + PO monthly + route counts

# COMMAND ----------

route_counts = routes.groupBy("supplier_id").agg(
    F.count("route_id").alias("route_count"),
    F.countDistinct("origin_country").alias("origin_country_count"),
    F.countDistinct("origin_port").alias("origin_port_count"),
    F.countDistinct("risk_region").alias("risk_region_count")
)

performance_mart = (
    scorecards_with_trend
    .join(
        po_monthly,
        on=[scorecards_with_trend.supplier_id == po_monthly.supplier_id,
            F.trunc(scorecards_with_trend.scorecard_month, "month") == po_monthly.po_month],
        how="left"
    )
    .join(route_counts, on="supplier_id", how="left")
    .select(
        scorecards_with_trend.supplier_id,
        F.col("scorecard_month"),
        F.col("fill_rate"),
        F.col("on_time_delivery_rate"),
        F.col("quality_issue_rate"),
        F.col("avg_lead_time_days"),
        F.col("lead_time_variance"),
        F.col("defect_rate"),
        F.col("fill_rate_3mo_slope"),
        F.col("otd_3mo_slope"),
        F.col("defect_rate_3mo_slope"),
        F.col("deterioration_flag"),
        F.coalesce(F.col("po_count"), F.lit(0)).alias("po_count"),
        F.coalesce(F.col("po_line_count"), F.lit(0)).alias("po_line_count"),
        F.coalesce(F.col("late_po_line_count"), F.lit(0)).alias("late_po_line_count"),
        F.col("avg_po_fill_rate"),
        F.col("avg_delivery_delay_days"),
        F.coalesce(F.col("route_count"), F.lit(0)).alias("route_count"),
        F.coalesce(F.col("origin_country_count"), F.lit(0)).alias("origin_country_count"),
        F.coalesce(F.col("origin_port_count"), F.lit(0)).alias("origin_port_count"),
        F.coalesce(F.col("risk_region_count"), F.lit(0)).alias("risk_region_count"),
        F.lit(datetime.utcnow().isoformat()).alias("gold_created_at"),
        F.lit("20_gold_supplier_performance_mart").alias("gold_source_notebook")
    )
)

row_count = performance_mart.count()
print(f"gold_supplier_performance_mart row count: {row_count}")
assert row_count == 576, f"Expected 576 rows (48 suppliers × 12 months), got {row_count}"

# COMMAND ----------

(
    performance_mart
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_supplier_performance_mart")
)

print(f"✅ written: {spark.table('supplysage_gold.gold_supplier_performance_mart').count()} rows")

# COMMAND ----------

results = []
mart = spark.table("supplysage_gold.gold_supplier_performance_mart")

rc = mart.count()
results.append({"check": "row_count_576", "status": "PASS" if rc == 576 else "FAIL", "detail": str(rc)})

deteriorating = mart.filter(F.col("deterioration_flag") == True).select("supplier_id").distinct().count()
results.append({"check": "deterioration_detected_gt0", "status": "PASS" if deteriorating > 0 else "WARN", "detail": f"{deteriorating} suppliers deteriorating"})

bad_fill = mart.filter((F.col("fill_rate") < 0) | (F.col("fill_rate") > 1)).count()
results.append({"check": "fill_rate_in_range", "status": "PASS" if bad_fill == 0 else "FAIL", "detail": str(bad_fill)})

for r in results:
    print(f"  [{r['status']}] {r['check']} — {r['detail']}")

val_df = spark.createDataFrame(results).withColumn("notebook", F.lit("20_gold_supplier_performance_mart")).withColumn("run_at", F.lit(datetime.utcnow().isoformat()))
val_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("supplysage_gold.gold_transform_validation_results")

failures = [r for r in results if r["status"] == "FAIL"]
assert len(failures) == 0, f"Validation failures: {failures}"
print("✅ All validations passed.")
