# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 17 — gold_dim_products_skus
# MAGIC **Source:** `supplysage_silver.silver_product_crosswalk`
# MAGIC **Target:** `supplysage_gold.gold_dim_products_skus`
# MAGIC **Grain:** One row per canonical_sku_id (3,387 rows expected)
# MAGIC **Purpose:** Conformed product/SKU dimension. Maps canonical SKU back to all source system IDs.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read silver_product_crosswalk

# COMMAND ----------

crosswalk = spark.table("supplysage_silver.silver_product_crosswalk")

print(f"silver_product_crosswalk row count: {crosswalk.count()}")
crosswalk.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build one row per canonical_sku_id
# MAGIC The crosswalk has multiple rows per canonical_sku_id (one per source system).
# MAGIC Pivot to get one row with all source IDs as separate columns.

# COMMAND ----------

# Get distinct canonical SKUs as the spine
sku_spine = crosswalk.select("canonical_sku_id").distinct()

# Pivot source system IDs
m5_ids = crosswalk.filter(F.col("source_system") == "m5").select(
    F.col("canonical_sku_id"),
    F.col("source_product_id").alias("m5_item_id")
)

retail_ids = crosswalk.filter(F.col("source_system") == "retail_inventory").select(
    F.col("canonical_sku_id"),
    F.col("source_product_id").alias("retail_product_id")
)

dataco_ids = crosswalk.filter(F.col("source_system") == "dataco").select(
    F.col("canonical_sku_id"),
    F.col("source_product_id").alias("dataco_product_id")
)

supplier_ids = crosswalk.filter(F.col("source_system") == "supplier_internal").select(
    F.col("canonical_sku_id"),
    F.col("source_product_id").alias("supplier_internal_sku_id")
)

# Extract category / department from M5 item_id where available
# M5 item format: CATEGORY_DEPT_NNN e.g. FOODS_1_001
sku_with_attrs = crosswalk.filter(F.col("source_system") == "m5").select(
    F.col("canonical_sku_id"),
    F.col("source_product_id").alias("m5_item_id"),
    # Parse category and dept from M5 item_id naming convention
    F.split(F.col("source_product_id"), "_").getItem(0).alias("category"),
    F.split(F.col("source_product_id"), "_").getItem(1).alias("department")
).dropDuplicates(["canonical_sku_id"])

# Build the dimension
gold_dim_products_skus = (
    sku_spine
    .join(sku_with_attrs.select("canonical_sku_id", "category", "department"), on="canonical_sku_id", how="left")
    .join(m5_ids, on="canonical_sku_id", how="left")
    .join(retail_ids, on="canonical_sku_id", how="left")
    .join(dataco_ids, on="canonical_sku_id", how="left")
    .join(supplier_ids, on="canonical_sku_id", how="left")
    .withColumn("gold_created_at", F.lit(datetime.utcnow().isoformat()))
    .withColumn("gold_source_notebook", F.lit("17_gold_dim_products_skus"))
)

row_count = gold_dim_products_skus.count()
print(f"gold_dim_products_skus row count: {row_count}")
assert row_count == 3387, f"Expected 3387 SKUs, got {row_count}"

# COMMAND ----------

(
    gold_dim_products_skus
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_dim_products_skus")
)

print(f"✅ gold_dim_products_skus written: {spark.table('supplysage_gold.gold_dim_products_skus').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

results = []

rc = spark.table("supplysage_gold.gold_dim_products_skus").count()
results.append({"check": "row_count_3387", "status": "PASS" if rc == 3387 else "FAIL", "detail": str(rc)})

nulls = spark.table("supplysage_gold.gold_dim_products_skus").filter(F.col("canonical_sku_id").isNull()).count()
results.append({"check": "no_null_canonical_sku_id", "status": "PASS" if nulls == 0 else "FAIL", "detail": str(nulls)})

distinct = spark.table("supplysage_gold.gold_dim_products_skus").select("canonical_sku_id").distinct().count()
results.append({"check": "canonical_sku_id_unique", "status": "PASS" if distinct == rc else "FAIL", "detail": f"{distinct}/{rc}"})

for r in results:
    print(f"  [{r['status']}] {r['check']} — {r['detail']}")

val_df = spark.createDataFrame(results).withColumn("notebook", F.lit("17_gold_dim_products_skus")).withColumn("run_at", F.lit(datetime.utcnow().isoformat()))
val_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("supplysage_gold.gold_transform_validation_results")

failures = [r for r in results if r["status"] == "FAIL"]
assert len(failures) == 0, f"Validation failures: {failures}"
print("✅ All validations passed.")
