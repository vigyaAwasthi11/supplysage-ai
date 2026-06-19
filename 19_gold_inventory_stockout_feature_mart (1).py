# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 19 — gold_inventory_stockout_feature_mart
# MAGIC **Sources:**
# MAGIC   - `supplysage_silver.silver_domain_inventory_stockout_snapshot` (view on silver_retail_inventory, 73,100 rows)
# MAGIC   - `supplysage_silver.silver_domain_retail_sales_price_daily` (view on 58M-row sales — aggregated, NOT copied raw)
# MAGIC   - `supplysage_silver.silver_m5_calendar` (1,969 rows)
# MAGIC   - `supplysage_gold.gold_dim_products_skus` (3,387 rows)
# MAGIC **Target:** `supplysage_gold.gold_inventory_stockout_feature_mart`
# MAGIC **Grain:** One row per canonical_sku_id × snapshot_date
# MAGIC **IMPORTANT:** We aggregate the 58M sales table to SKU-week level before joining.
# MAGIC We never copy raw daily sales rows into Gold.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load sources

# COMMAND ----------

inv_snapshot   = spark.table("supplysage_silver.silver_domain_inventory_stockout_snapshot")
calendar       = spark.table("supplysage_silver.silver_m5_calendar")
dim_skus       = spark.table("supplysage_gold.gold_dim_products_skus")

print(f"silver_domain_inventory_stockout_snapshot: {inv_snapshot.count()} rows")
print(f"silver_m5_calendar: {calendar.count()} rows")
print(f"gold_dim_products_skus: {dim_skus.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate sales to SKU-week grain from domain view
# MAGIC Use the domain view (which already joins sales + prices) to get weekly demand velocity.
# MAGIC This avoids touching the 58M-row table directly.

# COMMAND ----------

# silver_domain_retail_sales_price_daily has item_id, not canonical_sku_id.
# Resolve canonical_sku_id by joining through the crosswalk on the M5 item_id.
crosswalk_m5 = spark.table("supplysage_silver.silver_product_crosswalk").filter(
    F.col("source_system") == "m5"
).select(
    F.col("source_product_id").alias("item_id"),
    F.col("canonical_sku_id")
).dropDuplicates(["item_id"])

# IMPORTANT: the M5 dataset's calendar runs through ~2016, not the present day.
# Using current_date() here would return zero rows since no sales data exists
# within 30 days of "today". Instead, find the latest calendar_date actually
# present in the sales data and use THAT as the reference point for "last 30 days".
max_date_row = spark.sql("""
    SELECT MAX(calendar_date) AS max_date
    FROM supplysage_silver.silver_domain_retail_sales_price_daily
""").collect()[0]
reference_date = max_date_row["max_date"]
print(f"Using reference_date = {reference_date} (latest date in sales data) instead of current_date()")

sales_raw = spark.table("supplysage_silver.silver_domain_retail_sales_price_daily")

sales_agg = (
    sales_raw
    .filter(F.col("calendar_date") >= F.date_sub(F.lit(reference_date), 30))
    .join(crosswalk_m5, on="item_id", how="inner")
    .groupBy("canonical_sku_id")
    .agg(
        F.avg("units_sold").alias("avg_daily_units_sold"),
        F.avg("sell_price").alias("avg_sell_price"),
        F.sum(F.col("units_sold") * F.col("sell_price")).alias("total_revenue_30d"),
        F.countDistinct("calendar_date").alias("days_with_sales"),
        F.sum("units_sold").alias("total_units_30d")
    )
)

print(f"Sales 30-day aggregation: {sales_agg.count()} SKUs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get latest inventory snapshot per SKU
# MAGIC Use the most recent snapshot_date per canonical_sku_id

# COMMAND ----------

latest_window = Window.partitionBy("canonical_sku_id").orderBy(F.col("inventory_date").desc())

inv_latest = (
    inv_snapshot
    .withColumn("rn", F.row_number().over(latest_window))
    .filter(F.col("rn") == 1)
    .select(
        "canonical_sku_id",
        F.col("inventory_date").alias("snapshot_date"),
        "inventory_level",
        "units_sold",
        "units_ordered",
        "demand_forecast",
        "demand_forecast_raw",
        "demand_forecast_was_negative",
        "inventory_position",
        "stockout_gap_units",
        "is_stockout_risk",
        "inventory_coverage_ratio",
        "sell_through_rate",
        "price_vs_competitor"
    )
)

print(f"Latest inventory snapshot per SKU: {inv_latest.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check next 14 days for SNAP events (demand spike signals)

# COMMAND ----------

# IMPORTANT: silver_m5_calendar covers the M5 dataset's historical date range
# (through ~2016), not the present day. Use the calendar's own max date as the
# reference point for "next 14 days" rather than current_date().
calendar_max_date = calendar.agg(F.max("calendar_date").alias("max_date")).collect()[0]["max_date"]
print(f"Using calendar reference date = {calendar_max_date} instead of current_date()")

snap_upcoming = calendar.filter(
    (F.col("calendar_date") >= F.lit(calendar_max_date)) &
    (F.col("calendar_date") <= F.date_add(F.lit(calendar_max_date), 14)) &
    (F.col("is_snap_any_state") == True)
).agg(F.count("*").alias("snap_days_next_14d")).collect()[0]["snap_days_next_14d"]

is_event_upcoming = calendar.filter(
    (F.col("calendar_date") >= F.lit(calendar_max_date)) &
    (F.col("calendar_date") <= F.date_add(F.lit(calendar_max_date), 14)) &
    (F.col("is_event_day") == True)
).agg(F.count("*").alias("event_days_next_14d")).collect()[0]["event_days_next_14d"]

print(f"SNAP days in next 14 days (from data's latest date): {snap_upcoming}")
print(f"Event days in next 14 days (from data's latest date): {is_event_upcoming}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build feature mart

# COMMAND ----------

feature_mart = (
    inv_latest
    .join(sales_agg, on="canonical_sku_id", how="left")
    .join(dim_skus.select("canonical_sku_id", "category", "department", "m5_item_id"), on="canonical_sku_id", how="left")
    .withColumn(
        # Use actual demand forecast from retail_inventory; fall back to 30d sales average
        "effective_daily_demand",
        F.coalesce(
            F.col("demand_forecast"),
            F.col("avg_daily_units_sold"),
            F.lit(1.0)
        )
    )
    .withColumn(
        "days_of_cover",
        F.when(F.col("effective_daily_demand") > 0,
               F.col("inventory_level") / F.col("effective_daily_demand"))
         .otherwise(F.lit(999.0))  # no demand = no stockout risk
    )
    .withColumn(
        "sales_exposure_30d",
        F.coalesce(F.col("total_revenue_30d"), F.lit(0.0))
    )
    .withColumn("snap_days_next_14d", F.lit(snap_upcoming))
    .withColumn("event_days_next_14d", F.lit(is_event_upcoming))
    .withColumn(
        "demand_spike_in_cover_window",
        F.when(
            (F.col("days_of_cover") >= 0) & (F.col("days_of_cover") <= 14) &
            ((F.lit(snap_upcoming) > 0) | (F.lit(is_event_upcoming) > 0)),
            F.lit(True)
        ).otherwise(F.lit(False))
    )
    .withColumn("gold_created_at", F.lit(datetime.utcnow().isoformat()))
    .withColumn("gold_source_notebook", F.lit("19_gold_inventory_stockout_feature_mart"))
    .select(
        "canonical_sku_id",
        "category",
        "department",
        "m5_item_id",
        "snapshot_date",
        "inventory_level",
        "units_ordered",
        "demand_forecast",
        "effective_daily_demand",
        "avg_daily_units_sold",
        "avg_sell_price",
        "total_units_30d",
        "sales_exposure_30d",
        "days_of_cover",
        "inventory_position",
        "stockout_gap_units",
        "is_stockout_risk",
        "inventory_coverage_ratio",
        "sell_through_rate",
        "price_vs_competitor",
        "snap_days_next_14d",
        "event_days_next_14d",
        "demand_spike_in_cover_window",
        "gold_created_at",
        "gold_source_notebook"
    )
)

row_count = feature_mart.count()
print(f"gold_inventory_stockout_feature_mart row count: {row_count}")

# COMMAND ----------

(
    feature_mart
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("supplysage_gold.gold_inventory_stockout_feature_mart")
)

print(f"✅ written: {spark.table('supplysage_gold.gold_inventory_stockout_feature_mart').count()} rows")

# COMMAND ----------

results = []
mart = spark.table("supplysage_gold.gold_inventory_stockout_feature_mart")

rc = mart.count()
results.append({"check": "row_count_gt_0", "status": "PASS" if rc > 0 else "FAIL", "detail": str(rc)})

null_sku = mart.filter(F.col("canonical_sku_id").isNull()).count()
results.append({"check": "no_null_canonical_sku_id", "status": "PASS" if null_sku == 0 else "FAIL", "detail": str(null_sku)})

neg_cover = mart.filter(F.col("days_of_cover") < 0).count()
results.append({"check": "days_of_cover_non_negative", "status": "PASS" if neg_cover == 0 else "FAIL", "detail": str(neg_cover)})

neg_inv = mart.filter(F.col("inventory_level") < 0).count()
results.append({"check": "inventory_level_non_negative", "status": "PASS" if neg_inv == 0 else "FAIL", "detail": str(neg_inv)})

for r in results:
    print(f"  [{r['status']}] {r['check']} — {r['detail']}")

val_df = spark.createDataFrame(results).withColumn("notebook", F.lit("19_gold_inventory_stockout_feature_mart")).withColumn("run_at", F.lit(datetime.utcnow().isoformat()))
val_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("supplysage_gold.gold_transform_validation_results")

failures = [r for r in results if r["status"] == "FAIL"]
assert len(failures) == 0, f"Validation failures: {failures}"
print("✅ All validations passed.")
