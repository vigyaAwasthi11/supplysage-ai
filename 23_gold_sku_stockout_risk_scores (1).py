# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 23 — gold_sku_stockout_risk_scores
# MAGIC **Sources (all Gold):**
# MAGIC   - `supplysage_gold.gold_inventory_stockout_feature_mart`
# MAGIC   - `supplysage_gold.gold_supplier_sku_dependency_mart`
# MAGIC   - `supplysage_gold.gold_supplier_risk_scores`
# MAGIC   - `supplysage_gold.gold_dim_products_skus`
# MAGIC   - `supplysage_silver.silver_m5_calendar` (for SNAP/holiday signals)
# MAGIC **Target:** `supplysage_gold.gold_sku_stockout_risk_scores`
# MAGIC **Grain:** One row per canonical_sku_id (latest score per SKU)
# MAGIC **Purpose:** Stockout probability per SKU combining inventory, demand, supplier risk, and lead time.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import math
from datetime import datetime, date

spark = SparkSession.builder.getOrCreate()

SCORE_DATE = date.today().isoformat()

# COMMAND ----------

inv_features  = spark.table("supplysage_gold.gold_inventory_stockout_feature_mart")
dep_mart      = spark.table("supplysage_gold.gold_supplier_sku_dependency_mart")
risk_scores   = spark.table("supplysage_gold.gold_supplier_risk_scores")
dim_skus      = spark.table("supplysage_gold.gold_dim_products_skus")
calendar      = spark.table("supplysage_silver.silver_m5_calendar")

print(f"gold_inventory_stockout_feature_mart: {inv_features.count()} rows")
print(f"gold_supplier_sku_dependency_mart:    {dep_mart.count()} rows")
print(f"gold_supplier_risk_scores:            {risk_scores.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get primary supplier per SKU with their risk scores

# COMMAND ----------

primary_supplier_risk = (
    dep_mart
    .filter(F.col("is_primary_supplier") == True)
    .join(
        risk_scores.select(
            "supplier_id",
            F.col("overall_risk_score").alias("linked_supplier_risk_score"),
            F.col("risk_band").alias("linked_supplier_risk_band"),
            F.col("logistics_score").alias("supplier_logistics_score")
        ),
        on="supplier_id", how="left"
    )
    .select(
        "canonical_sku_id",
        "supplier_id",
        "supplier_name",
        "criticality_tier",
        "dependency_percent",
        "std_lead_time_days",
        "transport_mode",
        "alternate_status",
        "estimated_switch_days",
        "linked_supplier_risk_score",
        "linked_supplier_risk_band",
        "supplier_logistics_score"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute adjusted lead time
# MAGIC Adjust standard lead time based on supplier logistics risk score.
# MAGIC A supplier with logistics_score=80 adds 40% to lead time.

# COMMAND ----------

# logistics disruption factor: score/100 * 0.5 = max +50% lead time extension
primary_supplier_risk = primary_supplier_risk.withColumn(
    "logistics_disruption_factor",
    F.coalesce(F.col("supplier_logistics_score"), F.lit(0.0)) / 100.0 * 0.5
).withColumn(
    "adjusted_lead_time_days",
    F.round(
        F.col("std_lead_time_days") * (1.0 + F.col("logistics_disruption_factor")),
        1
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join inventory features + supplier risk

# COMMAND ----------

sku_risk_base = (
    inv_features
    .join(primary_supplier_risk, on="canonical_sku_id", how="left")
    # NOTE: inv_features (gold_inventory_stockout_feature_mart) already carries
    # category, department, and m5_item_id from Notebook 19's own join to
    # gold_dim_products_skus. Joining dim_skus again here would just
    # re-introduce the same columns and cause AMBIGUOUS_REFERENCE errors.
    # No second join to dim_skus is needed.
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute stockout probability using sigmoid function
# MAGIC
# MAGIC Core formula:
# MAGIC   cover_gap = adjusted_lead_time_days - days_of_cover
# MAGIC   If cover_gap > 0, we WILL run out before the next shipment arrives.
# MAGIC   stockout_probability = sigmoid(0.3 × cover_gap + 0.02 × supplier_risk_score + no_alternate_penalty)
# MAGIC
# MAGIC Sigmoid: 1 / (1 + exp(-x))  — we use a Spark approximation.

# COMMAND ----------

# Register sigmoid as a UDF
import math as _math

def sigmoid(x):
    if x is None:
        return 0.0
    try:
        return 1.0 / (1.0 + _math.exp(-float(x)))
    except OverflowError:
        return 0.0 if float(x) < 0 else 1.0

sigmoid_udf = F.udf(sigmoid)
spark.udf.register("sigmoid_udf", sigmoid)

# COMMAND ----------

stockout_scores = (
    sku_risk_base
    .withColumn(
        "cover_gap",
        F.col("adjusted_lead_time_days") - F.col("days_of_cover")
    )
    .withColumn(
        "no_alternate_penalty",
        F.when(F.col("alternate_status") == "none", F.lit(1.5))
         .when(F.col("alternate_status") == "pending", F.lit(0.5))
         .otherwise(F.lit(0.0))
    )
    .withColumn(
        "demand_spike_multiplier",
        F.when(F.col("demand_spike_in_cover_window") == True, F.lit(1.3))
         .otherwise(F.lit(1.0))
    )
    .withColumn(
        "sigmoid_input",
        (
            F.col("cover_gap") * 0.30 +
            F.coalesce(F.col("linked_supplier_risk_score"), F.lit(0.0)) * 0.02 +
            F.col("no_alternate_penalty")
        ) * F.col("demand_spike_multiplier")
    )
    .withColumn(
        "stockout_probability",
        F.round(sigmoid_udf(F.col("sigmoid_input")), 3)
    )
    .withColumn(
        "expected_stockout_date",
        F.when(
            F.col("days_of_cover") < 90,
            F.date_add(F.current_date(), F.col("days_of_cover").cast("int"))
        ).otherwise(F.lit(None).cast("date"))
    )
    .withColumn(
        "sales_exposure_30d",
        F.coalesce(F.col("sales_exposure_30d"), F.lit(0.0))
    )
    .withColumn(
        "stockout_risk_band",
        F.when(F.col("stockout_probability") >= 0.75, F.lit("critical"))
         .when(F.col("stockout_probability") >= 0.50, F.lit("high"))
         .when(F.col("stockout_probability") >= 0.25, F.lit("medium"))
         .otherwise(F.lit("low"))
    )
    .withColumn("score_date", F.lit(SCORE_DATE))
    .withColumn("gold_created_at", F.lit(datetime.utcnow().isoformat()))
    .withColumn("gold_source_notebook", F.lit("23_gold_sku_stockout_risk_scores"))
    .select(
        "canonical_sku_id",
        "category",
        "department",
        "score_date",
        "supplier_id",
        "supplier_name",
        "criticality_tier",
        "dependency_percent",
        "std_lead_time_days",
        "adjusted_lead_time_days",
        "transport_mode",
        "inventory_level",
        "effective_daily_demand",
        "days_of_cover",
        "cover_gap",
        "stockout_probability",
        "stockout_risk_band",
        "expected_stockout_date",
        "sales_exposure_30d",
        "linked_supplier_risk_score",
        "linked_supplier_risk_band",
        "alternate_status",
        "estimated_switch_days",
        "demand_spike_in_cover_window",
        "snap_days_next_14d",
        "no_alternate_penalty",
        "gold_created_at",
        "gold_source_notebook"
    )
)

row_count = stockout_scores.count()
print(f"gold_sku_stockout_risk_scores: {row_count} rows")

critical_skus = stockout_scores.filter(F.col("stockout_risk_band") == "critical").count()
print(f"Critical stockout risk SKUs: {critical_skus}")
print(f"High stockout risk SKUs: {stockout_scores.filter(F.col('stockout_risk_band') == 'high').count()}")

display(
    stockout_scores
    .filter(F.col("stockout_risk_band").isin("critical", "high"))
    .orderBy(F.col("stockout_probability").desc())
    .select("canonical_sku_id", "category", "days_of_cover", "stockout_probability",
            "supplier_name", "alternate_status", "sales_exposure_30d", "stockout_risk_band")
    .limit(20)
)

# COMMAND ----------

(
    stockout_scores
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_sku_stockout_risk_scores")
)
print(f"✅ gold_sku_stockout_risk_scores written: {spark.table('supplysage_gold.gold_sku_stockout_risk_scores').count()} rows")

# COMMAND ----------

results = []
scores_tbl = spark.table("supplysage_gold.gold_sku_stockout_risk_scores")

rc = scores_tbl.count()
results.append({"check": "row_count_gt_0", "status": "PASS" if rc > 0 else "FAIL", "detail": str(rc)})

bad_prob = scores_tbl.filter((F.col("stockout_probability") < 0) | (F.col("stockout_probability") > 1)).count()
results.append({"check": "stockout_probability_in_range", "status": "PASS" if bad_prob == 0 else "FAIL", "detail": str(bad_prob)})

bad_band = scores_tbl.filter(~F.col("stockout_risk_band").isin("critical", "high", "medium", "low")).count()
results.append({"check": "risk_band_valid_vocab", "status": "PASS" if bad_band == 0 else "FAIL", "detail": str(bad_band)})

for r in results:
    print(f"  [{r['status']}] {r['check']} — {r['detail']}")

val_df = spark.createDataFrame(results).withColumn("notebook", F.lit("23_gold_sku_stockout_risk_scores")).withColumn("run_at", F.lit(datetime.utcnow().isoformat()))
val_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("supplysage_gold.gold_transform_validation_results")

failures = [r for r in results if r["status"] == "FAIL"]
assert len(failures) == 0, f"Validation failures: {failures}"
print("✅ All validations passed.")
