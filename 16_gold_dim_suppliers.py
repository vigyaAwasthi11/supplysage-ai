# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 16 — gold_dim_suppliers
# MAGIC **Source:** `supplysage_silver.silver_domain_supplier_network`
# MAGIC **Target:** `supplysage_gold.gold_dim_suppliers`
# MAGIC **Grain:** One row per supplier (48 rows expected)
# MAGIC **Purpose:** Conformed supplier dimension. Stable attributes + operational summary fields.
# MAGIC All fact tables FK into this table via supplier_id.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# Create Gold schema if not exists
spark.sql("CREATE SCHEMA IF NOT EXISTS supplysage_gold")

print("Schema ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read silver_domain_supplier_network

# COMMAND ----------

supplier_network = spark.table("supplysage_silver.silver_domain_supplier_network")

print(f"silver_domain_supplier_network row count: {supplier_network.count()}")
supplier_network.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build gold_dim_suppliers

# COMMAND ----------

gold_dim_suppliers = supplier_network.select(
    F.col("supplier_id"),
    F.col("supplier_name"),
    F.coalesce(F.col("parent_company"), F.lit(None).cast(StringType())).alias("parent_company"),
    F.col("country"),
    F.col("region"),
    F.col("supplier_category"),
    F.col("criticality_tier"),
    F.col("annual_spend"),
    F.col("single_source_flag"),
    F.col("default_lead_time_days"),
    # Operational summary from domain view
    F.col("alias_count"),
    F.col("mapped_sku_count"),
    F.col("primary_sku_count"),
    F.col("sku_with_alternate_count"),
    F.col("avg_dependency_percent"),
    F.col("avg_standard_lead_time"),
    F.col("po_count"),
    F.col("avg_po_fill_rate"),
    F.col("avg_actual_lead_time"),
    F.col("avg_delivery_delay"),
    F.col("late_po_line_count"),
    F.col("latest_fill_rate"),
    F.col("latest_on_time_delivery_rate"),
    F.col("latest_quality_issue_rate"),
    F.col("latest_defect_rate"),
    F.col("deterioration_flag"),
    F.col("route_count"),
    F.col("origin_country_count"),
    F.col("origin_port_count"),
    F.col("risk_region_count"),
    # Metadata
    F.lit(datetime.utcnow().isoformat()).alias("gold_created_at"),
    F.lit("16_gold_dim_suppliers").alias("gold_source_notebook")
)

# Validate
row_count = gold_dim_suppliers.count()
print(f"gold_dim_suppliers row count: {row_count}")
assert row_count == 48, f"Expected 48 suppliers, got {row_count}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta

# COMMAND ----------

(
    gold_dim_suppliers
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_dim_suppliers")
)

print(f"✅ gold_dim_suppliers written: {spark.table('supplysage_gold.gold_dim_suppliers').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

results = []

# V1: row count
rc = spark.table("supplysage_gold.gold_dim_suppliers").count()
results.append({"check": "row_count_48", "status": "PASS" if rc == 48 else "FAIL", "detail": str(rc)})

# V2: no null supplier_id
nulls = spark.table("supplysage_gold.gold_dim_suppliers").filter(F.col("supplier_id").isNull()).count()
results.append({"check": "no_null_supplier_id", "status": "PASS" if nulls == 0 else "FAIL", "detail": str(nulls)})

# V3: criticality_tier values are valid
valid_tiers = spark.table("supplysage_gold.gold_dim_suppliers").filter(
    ~F.col("criticality_tier").isin("Tier 1", "Tier 2", "Tier 3")
).count()
results.append({"check": "valid_criticality_tiers", "status": "PASS" if valid_tiers == 0 else "FAIL", "detail": str(valid_tiers)})

# V4: supplier_id is unique
distinct = spark.table("supplysage_gold.gold_dim_suppliers").select("supplier_id").distinct().count()
results.append({"check": "supplier_id_unique", "status": "PASS" if distinct == rc else "FAIL", "detail": f"distinct={distinct} total={rc}"})

for r in results:
    print(f"  [{r['status']}] {r['check']} — {r['detail']}")

# Write validation results
val_df = spark.createDataFrame(results).withColumn("notebook", F.lit("16_gold_dim_suppliers")).withColumn("run_at", F.lit(datetime.utcnow().isoformat()))
(val_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("supplysage_gold.gold_transform_validation_results"))

failures = [r for r in results if r["status"] == "FAIL"]
assert len(failures) == 0, f"Validation failures: {failures}"
print("✅ All validations passed.")
