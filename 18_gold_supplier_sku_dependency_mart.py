# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 18 — gold_supplier_sku_dependency_mart
# MAGIC **Sources:**
# MAGIC   - `supplysage_silver.silver_supplier_sku_map` (355 rows)
# MAGIC   - `supplysage_silver.silver_alternate_suppliers` (155 rows)
# MAGIC   - `supplysage_silver.silver_product_crosswalk` (3,387 rows)
# MAGIC   - `supplysage_silver.silver_suppliers` (48 rows)
# MAGIC **Target:** `supplysage_gold.gold_supplier_sku_dependency_mart`
# MAGIC **Grain:** One row per supplier × canonical_sku_id pair
# MAGIC **Purpose:** Powers the "Impacted SKUs" drill-down. Answers which SKUs a given supplier
# MAGIC supplies, dependency weight, alternate options, and switching difficulty.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Silver sources

# COMMAND ----------

sku_map      = spark.table("supplysage_silver.silver_supplier_sku_map")
alternates   = spark.table("supplysage_silver.silver_alternate_suppliers")
crosswalk    = spark.table("supplysage_silver.silver_product_crosswalk")
suppliers    = spark.table("supplysage_silver.silver_suppliers")

print(f"silver_supplier_sku_map:      {sku_map.count()} rows")
print(f"silver_alternate_suppliers:   {alternates.count()} rows")
print(f"silver_product_crosswalk:     {crosswalk.count()} rows")
print(f"silver_suppliers:             {suppliers.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve canonical_sku_id from product crosswalk
# MAGIC silver_supplier_sku_map uses sku_id which is the supplier_internal SKU.
# MAGIC Resolve to canonical_sku_id via silver_product_crosswalk.

# COMMAND ----------

# Get canonical mapping for supplier internal SKUs
canonical_map = crosswalk.filter(
    F.col("source_system").isin("supplier_internal", "m5")
).select(
    F.col("source_product_id").alias("sku_id"),
    F.col("canonical_sku_id")
).dropDuplicates(["sku_id"])

# If sku_id already IS the canonical, join will cover it
# Self-mapping rows added during silver relationship validation have canonical_sku_id = source_product_id
sku_map_resolved = sku_map.join(canonical_map, on="sku_id", how="left").withColumn(
    "canonical_sku_id",
    F.coalesce(F.col("canonical_sku_id"), F.col("sku_id"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve best alternate per SKU
# MAGIC Pick the best approved alternate per primary_supplier_id × sku_id:
# MAGIC best = approved AND highest capacity_available_pct

# COMMAND ----------

alt_window = Window.partitionBy("sku_id", "primary_supplier_id").orderBy(
    F.col("approved_flag").desc(),
    F.col("capacity_available_pct").desc()
)

best_alternate = (
    alternates
    .withColumn("rn", F.row_number().over(alt_window))
    .filter(F.col("rn") == 1)
    .select(
        F.col("sku_id"),
        F.col("primary_supplier_id"),
        F.col("alternate_supplier_id").alias("best_alternate_supplier_id"),
        F.col("approved_flag").alias("alternate_approved_flag"),
        F.col("switching_cost_level"),
        F.col("estimated_switch_days"),
        F.col("capacity_available_pct")
    )
)

# Derive alternate_status label
best_alternate = best_alternate.withColumn(
    "alternate_status",
    F.when(F.col("alternate_approved_flag") == True, F.lit("approved"))
     .when(F.col("best_alternate_supplier_id").isNotNull(), F.lit("pending"))
     .otherwise(F.lit("none"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join all sources together

# COMMAND ----------

# Cell patch: Build gold supplier-SKU dependency mart with explicit aliases
# Fix:
# - Avoid ambiguous sku_id after joining supplier_sku_map and alternate_suppliers
# - Rename alternate supplier sku_id to alt_sku_id before joining
# - Use fully qualified aliases in SQL
# - Serverless-safe: no cache(), no persist(), no CACHE TABLE

from datetime import datetime
from pyspark.sql import functions as F

# -------------------------------------------------------------------
# 1. Build clean supplier-SKU map input
# -------------------------------------------------------------------

sku_map_for_join = (
    sku_map_resolved
    .select(
        F.col("supplier_id").alias("supplier_id"),
        F.col("sku_id").alias("sku_id"),
        F.col("canonical_sku_id").alias("canonical_sku_id"),
        F.col("dependency_percent").alias("dependency_percent"),
        F.col("is_primary_supplier").alias("is_primary_supplier"),
        F.col("standard_lead_time_days").alias("standard_lead_time_days"),
        F.col("origin_country").alias("origin_country"),
        F.col("origin_port").alias("origin_port"),
        F.col("destination_dc").alias("destination_dc"),
        F.col("transport_mode").alias("transport_mode"),
        F.col("minimum_order_quantity").alias("minimum_order_quantity"),
        F.col("alternate_supplier_available").alias("alternate_supplier_available")
    )
    .dropDuplicates(["supplier_id", "sku_id", "canonical_sku_id"])
)

# -------------------------------------------------------------------
# 2. Build clean supplier dimension input
# -------------------------------------------------------------------

suppliers_for_join = (
    suppliers
    .select(
        F.col("supplier_id").alias("supplier_id"),
        F.col("supplier_name").alias("supplier_name"),
        F.col("country").alias("country"),
        F.col("region").alias("region"),
        F.col("supplier_category").alias("supplier_category"),
        F.col("criticality_tier").alias("criticality_tier"),
        F.col("annual_spend").alias("annual_spend"),
        F.col("single_source_flag").alias("single_source_flag")
    )
    .dropDuplicates(["supplier_id"])
)

# -------------------------------------------------------------------
# 3. Build clean alternate supplier input
# -------------------------------------------------------------------
# Important:
# The alternate supplier table also has a sku_id column.
# Rename it to alt_sku_id so Spark never sees two unqualified sku_id columns.

best_alternate_for_join = (
    best_alternate
    .select(
        F.col("sku_id").alias("alt_sku_id"),
        F.col("primary_supplier_id").alias("alt_primary_supplier_id"),
        F.col("best_alternate_supplier_id").alias("best_alternate_supplier_id"),
        F.col("alternate_status").alias("alternate_status"),
        F.col("switching_cost_level").alias("switching_cost_level"),
        F.col("estimated_switch_days").alias("estimated_switch_days"),
        F.col("capacity_available_pct").alias("capacity_available_pct")
    )
    .dropDuplicates(["alt_sku_id", "alt_primary_supplier_id"])
)

# -------------------------------------------------------------------
# 4. Register temp views for fully qualified SQL joins
# -------------------------------------------------------------------

sku_map_for_join.createOrReplaceTempView("tmp_18_sku_map_for_join")
suppliers_for_join.createOrReplaceTempView("tmp_18_suppliers_for_join")
best_alternate_for_join.createOrReplaceTempView("tmp_18_best_alternate_for_join")

# -------------------------------------------------------------------
# 5. Build dependency mart
# -------------------------------------------------------------------

gold_created_at = datetime.utcnow().isoformat()

dependency_mart = spark.sql(f"""
SELECT
    m.supplier_id,
    s.supplier_name,
    s.country,
    s.region,
    s.supplier_category,
    s.criticality_tier,
    s.annual_spend,
    s.single_source_flag,

    m.sku_id,
    m.canonical_sku_id,
    m.dependency_percent,
    m.is_primary_supplier,
    m.alternate_supplier_available,
    m.standard_lead_time_days AS std_lead_time_days,
    m.origin_country,
    m.origin_port,
    m.destination_dc,
    m.transport_mode,
    m.minimum_order_quantity,

    a.best_alternate_supplier_id,
    COALESCE(a.alternate_status, 'none') AS alternate_status,
    a.switching_cost_level,
    a.estimated_switch_days,
    a.capacity_available_pct,

    '{gold_created_at}' AS gold_created_at,
    '18_gold_supplier_sku_dependency_mart' AS gold_source_notebook

FROM tmp_18_sku_map_for_join m

LEFT JOIN tmp_18_suppliers_for_join s
    ON m.supplier_id = s.supplier_id

LEFT JOIN tmp_18_best_alternate_for_join a
    ON m.sku_id = a.alt_sku_id
   AND m.supplier_id = a.alt_primary_supplier_id
""")

row_count = dependency_mart.count()
print(f"gold_supplier_sku_dependency_mart row count: {row_count}")

display(dependency_mart.limit(20))

# COMMAND ----------

(
    dependency_mart
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_supplier_sku_dependency_mart")
)

print(f"✅ written: {spark.table('supplysage_gold.gold_supplier_sku_dependency_mart').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

results = []
mart = spark.table("supplysage_gold.gold_supplier_sku_dependency_mart")

rc = mart.count()
results.append({"check": "row_count_gt_355", "status": "PASS" if rc >= 355 else "FAIL", "detail": str(rc)})

null_sup = mart.filter(F.col("supplier_id").isNull()).count()
results.append({"check": "no_null_supplier_id", "status": "PASS" if null_sup == 0 else "FAIL", "detail": str(null_sup)})

null_sku = mart.filter(F.col("canonical_sku_id").isNull()).count()
results.append({"check": "no_null_canonical_sku_id", "status": "PASS" if null_sku == 0 else "FAIL", "detail": str(null_sku)})

# Each SKU must have exactly one primary supplier
multi_primary = mart.filter(F.col("is_primary_supplier") == True).groupBy("canonical_sku_id").count().filter(F.col("count") > 1).count()
results.append({"check": "one_primary_per_sku", "status": "PASS" if multi_primary == 0 else "FAIL", "detail": str(multi_primary)})

# dependency_percent between 0 and 1
bad_dep = mart.filter((F.col("dependency_percent") < 0) | (F.col("dependency_percent") > 1)).count()
results.append({"check": "dependency_percent_valid_range", "status": "PASS" if bad_dep == 0 else "FAIL", "detail": str(bad_dep)})

# alternate_status values
bad_alt = mart.filter(~F.col("alternate_status").isin("approved", "pending", "none")).count()
results.append({"check": "alternate_status_valid_vocab", "status": "PASS" if bad_alt == 0 else "FAIL", "detail": str(bad_alt)})

for r in results:
    print(f"  [{r['status']}] {r['check']} — {r['detail']}")

val_df = spark.createDataFrame(results).withColumn("notebook", F.lit("18_gold_supplier_sku_dependency_mart")).withColumn("run_at", F.lit(datetime.utcnow().isoformat()))
val_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("supplysage_gold.gold_transform_validation_results")

failures = [r for r in results if r["status"] == "FAIL"]
assert len(failures) == 0, f"Validation failures: {failures}"
print("✅ All validations passed.")
